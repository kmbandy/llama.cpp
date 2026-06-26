# MAD-305 — ml8 fp8 GEMM dyn-VGPR Gauntlet: Results & Repro

**GPU:** AMD R9700 (gfx1201, RDNA4, wave32), KFD node 1, PCI 0000:42:00.0. NOT isolated from the
Hyprland compositor (any GPU wedge → MODE1 reset → desktop reboot).
**Toolchain:** ROCm `/opt/rocm`; assembler `/opt/rocm/llvm/bin/clang`; harness `occ_dispatch` (raw PM4 / libhsakmt KFD queue).
**GPU clock counter:** ~100 MHz (the in-kernel REALTIME timer base; all TF derived from it).
**Branch:** sync/upstream-2026-06-09. Dir: `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`.

> rocprof / rocprofv3 **cannot** hook these PM4 kernels (they run under the HSA layer rocprof intercepts).
> Bottleneck instrumentation = in-kernel `PROFILE=1` phase timers + RGA (static) + `dram_meter`
> (rocprofiler-sdk device-counting, agent-level GL2C/GRBM/SQ counters over a wrapped subprocess).

---

## 1. The ml8 shapes (REAL — no artificial/square shapes)

Dense model (hidden 2560 / FFN 9216). M = tokens. K multiple of ml8 group size 64. Source: KG `3caaf612`.

| name     | regime        | M    | K    | N    |
|----------|---------------|------|------|------|
| down     | training-fwd  | 2048 | 9216 | 2560 |
| gate/up  | training-fwd  | 2048 | 2560 | 9216 |
| attn_q   | training-fwd  | 2048 | 2560 | 4096 |
| attn_kv  | training-fwd  | 2048 | 2560 | 1024 |
| attn_o   | training-fwd  | 2048 | 4096 | 2560 |
| down_pf  | prefill       | 512  | 9216 | 2560 |
| gtup_pf  | prefill       | 512  | 2560 | 9216 |
| q_pf     | prefill       | 512  | 2560 | 4096 |
| kv_pf    | prefill       | 512  | 2560 | 1024 |
| o_pf     | prefill       | 512  | 4096 | 2560 |

(Big-batch variant used only in `--mbml8long`: same K/N, M=131072.)

---

## 2. Kernel & configs

**Kernel:** `occ_kernel_mbgemm.s` — persistent single-wave-per-WG, atomic work-queue claim, real
per-K feed (A direct `global_load_b64`, B `global_load_tr_b64` from preshuffled Bshuf), oracle-gated.

**Assemble defsyms** (`-Wa,-defsym,NAME=VAL`):
- `FM`, `FN` — per-wave accumulator tile = (16·FM)×(16·FN). reuse = FM·FN/(FM+FN).
- `BATCH` — tiles claimed per atomic grab (amortizes the queue counter + dyn grow/shrink).
- `DYNVGPR` — 0 = static (reserve full footprint); 1 = lean launch (32 VGPR) + `s_alloc_vgpr` grow.
- `GENDIV` — 1 = magic-reciprocal tile decode `row=mul_hi(ti,magic); col=ti-row*NTL` (s12=magic
  =ceil(2³²/NTL), s13=NTL). REQUIRED for non-pow2 NTL (the real ml8 N-dims). 0 = pow2 shift/mask.
- `DEFERGROW` — 1 = dyn-PROPER variant: A/B frags single-buffered in the LEAN block (v11+), grow
  to **accumulators-only ~96-VGPR footprint** (vs 128 prefetch), single-buffer occupancy-hidden
  feed. **GROW FIRST then load** (load-then-grow WEDGES the GPU — see §6).
- `NAIVEFEED` — 1 = exposed single-buffer feed (load→wait→compute, no prefetch). 0 = double-buffer prefetch.
- `NOFEED` — 1 = operands loaded once, reused (isolation probe; oracle BAD by design).
- `PROFILE` — 1 = in-kernel REALTIME phase timers (COMPUTE/ATOMIC/GROW/STORE/SETUP → occ[24..44], wg0).

**VGPR footprint** (dyn-able = ≤128, no umr): `(32 + FM·FN·8 + FM·4 + FN·4)` rounded to 16
(prefetch path); DEFERGROW = `(32 + FM·FN·8)` rounded (accumulators only).
- 1x1=48, 2x2=80, 1x4/4x1=96, 2x4/4x2=128 (prefetch) / 96 (defergrow). Static-only fat: 4x4=192, 8x2/2x8=208, 8x1=144.
- Tile must divide the shape: FM | (M/16), FN | (N/16). M=2048→FM∈{1,2,4,8}; M=512→FM∈{1,2,4,8} (≤512/16=32).

**Bin naming:** `occ_mbgemm_{FM}x{FN}_b{BATCH}_d{0|1}_{SUFFIX}.bin`
where d0=static, d1=dyn; SUFFIX: `gd`=GENDIV+prefetch, `dg`=GENDIV+DEFERGROW, `ndgd`=GENDIV+NAIVEFEED.

**Build scripts:** `build_gauntlet.sh` (all stage-1 tiles, gd), `build_dynproper.sh` (2x4/4x2 ×
BATCH{1,8,32} × {gd,dg}), `build_mbg_gendiv.sh`, `build_mbg_naive.sh`.

---

## 3. Harness (occ_dispatch) — modes

Compile (must use bash; fish breaks word-splitting):
```
ROCM=/opt/rocm; PM4=../dvgpr_pm4
clang++ -std=c++17 -O2 -Wall -Wno-unused -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
  occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
  "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch
```
(IDE hsakmt.h diagnostics are pre-existing noise — the -I flags supply the headers.)

| mode | what it does |
|------|--------------|
| `--mbml8gaunt` | STAGE 1: per shape, sustained sweep of every dividing tile × {static,dyn}, ~12s/cell, ranks → top-4 static + top-4 dyn |
| `--mbml8dyn`   | FAIR RACE: per shape, static refs (2x8,4x4) vs dyn-PROPER variants (2x4/4x2 × {pf b1/b8/b32, dg b1/b8/b32}), ~12s/cell, oracle-gated |
| `--mbml8gate`  | SAFE 512³ oracle gate (single <1s dispatch each) for new bins — confirms correctness, isolates brick path |
| `--mbml8long`  | SUSTAINED steady-state at M=131072, REPS back-to-back (run_mbgemm `reps`/`targetSecs`), per-rep min/max spread |

`run_mbgemm(node, bin, dynvgpr, pool, M,N,K, FM,FN, fullCheck, useGenDiv, reps, targetSecs)`:
fullCheck → 512³-style oracle (chained wmma_ref vs C); targetSecs>0 → loop reps until that much wall
(steady-state); per-rep GPU-clock span min/max in MbgResult.wall{Min,Max}.

**TF formula:** `TF = TOTAL·FM·FN·KT · 2·16³ · freq_hz / span_cycles / 1e12`, TOTAL=(M/16FM)(N/16FN), KT=K/16.

---

## 4. STAGE 1 results (`--mbml8gaunt`, M real, ~12s/cell sustained, oracle 256/256 all)

Top-4 static / top-4 dyn per shape (TF):

| shape   | TOP-4 STATIC                          | TOP-4 DYN (prefetch B1)            |
|---------|---------------------------------------|------------------------------------|
| down    | 2x8 18.8 · 2x4 17.1 · 4x4 16.4 · 1x4 14.2 | 2x4 12.5 · 1x4 10.7 · 4x2 10.5 · 2x2 10.0 |
| gate/up | 2x8 15.9 · 4x4 15.6 · 8x2 13.5 · 2x4 11.1 | 2x4 9.3 · 4x2 9.2 · 2x2 6.2 · 1x4 6.2 |
| attn_q  | 2x8 9.2 · 4x4 9.0 · 8x2 8.5 · 2x4 7.5  | 2x4 5.7 · 4x2 5.6 · 2x2 4.4 · 1x4 4.4 |
| attn_kv | 2x8 2.9 · 4x4 2.9 · 8x2 2.8 · 2x4 2.7  | 4x2 1.9 · 2x4 1.9 · 1x4 1.7 · 2x2 1.7 |
| attn_o  | 2x8 9.8 · 4x4 9.4 · 2x4 8.6 · 4x2 8.1  | 2x4 6.3 · 4x2 5.9 · 1x4 5.3 · 2x2 5.3 |
| down_pf | 2x8 6.6 · 4x4 6.6 · 2x4 6.2 · 4x2 5.9  | 2x4 4.3 · 4x2 4.2 · 1x4 3.9 · 2x2 3.8 |
| gtup_pf | 2x8 5.9 · 4x4 5.8 · 8x2 5.7 · 2x4 5.1  | 2x4 3.7 · 4x2 3.7 · 2x2 3.1 · 4x1 3.1 |
| q_pf    | 4x4 2.9 · 2x8 2.9 · 8x2 2.9 · 4x2 2.7  | 2x4 1.9 · 4x2 1.9 · 2x2 1.7 · 1x4 1.7 |
| kv_pf   | 2x8 0.8 · 8x2 0.8 · 4x4 0.8 · 4x2 0.8  | 4x2 0.5 · 2x4 0.5 · 1x4 0.5 · 4x1 0.5 |
| o_pf    | 4x4 3.0 · 2x8 3.0 · 8x2 2.9 · 2x4 2.8  | 4x2 1.9 · 2x4 1.9 · 1x4 1.8 · 2x2 1.8 |

**Findings:** Static wins every shape. The fat **2x8 / 4x4** tiles dominate (2x8 = FM2 FN8: only 2
A-streams/wave → best A-feed locality on the single-wave kernel; 8x2 loses here because FM=8 scatters
8 A-streams w/ no shared-LDS amortization). dyn (on the prefetch kernel, B1) trails by ~30%.
NOTE: absolute TF understated at real M=2048 (small GEMM, launch overhead under-amortized) — see §5
for amortized steady-state.

---

## 5. SUSTAINED steady-state (`--mbml8long`, M=131072, REPS back-to-back, oracle OK)

Reading is trustworthy when the per-rep [min..max] spread is tight (<1%). Corrects the ms-scale
under-statement (~2.4× higher than the short runs).

| shape (M=131072) | tile | static TF | dyn TF | note |
|------------------|------|-----------|--------|------|
| down   K9216 | 2x2 | 29.4 | 18.7 | dyn −36% |
| down   K9216 | 4x2 | 18.5 | 18.3 | tie |
| down   K9216 | 2x4 | **42.2** | 32.8 | dyn −22% (BEST static) |
| down   K9216 | 4x4 | 34.3 | — (static only) | |
| gateup K2560 | 2x2 | 9.0 | 9.0 | tie |
| gateup K2560 | 4x2 | 18.0 | 17.9 | tie |
| gateup K2560 | 2x4 | 18.0 | 17.9 | tie |
| gateup K2560 | 4x4 | **33.8** | — | BEST static |

**Finding:** dyn ties at best, loses up to −36%; never beats static. static glass-flat (±0.2 TF/80 reps).

---

## 6. THE FAIR RACE (`--mbml8dyn`, M real, dyn on its PROPER kernel vs static)

dyn-PROPER variants: `pf b{1,8,32}` (prefetch + BATCH amortization), `dg b{1,8,32}` (DEFERGROW:
96-VGPR footprint, single-buffer, grow-first). Static refs: 2x8, 4x4.

### down (M=2048 K=9216 N=2560) — DONE, oracle 256/256 all, NO BRICK
| config | TF | maxlive |
|--------|----|---------|
| **stat 2x8** | **18.8** | 768 |
| stat 4x4 | 16.6 | 768 |
| dyn 2x4 **pf b8** | **14.6** | 1152 |  ← best dyn (BATCH amortization +17% vs b1)
| dyn 2x4 dg b8 | 13.9 | 1152 |
| dyn 2x4 pf b1 | 12.5 | 1152 |
| dyn 2x4 pf b32 | 12.4 | 1152 |
| dyn 2x4 dg b1 | 11.8 | 1152 |
| dyn 2x4 dg b32 | 10.8 | 1152 |
| dyn 4x2 pf b1 | 10.5 | 1152 |
| dyn 4x2 dg b1 | 11.0 | 1152 |

**down findings:** (1) BATCH amortization is the real dyn lever — `pf b8` (14.6) > `pf b1` (12.5),
+17%; the per-tile grow tax was the handicap. (2) DEFERGROW did NOT help — `dg b8` (13.9) < `pf b8`
(14.6); the 96-VGPR footprint gain < the single-buffer exposed-feed cost. (3) Best dyn 14.6 still
LOSES to best static 18.8 (−22%), even on dyn's properly-tuned kernel.

### THE FUNNEL FIELD — top-4 static + top-4 dyn per shape (COMPLETE, every cell oracle-OK)
Gauntlet Stage-1 output. Static top-4 = full tile sweep (§4, `--mbml8gaunt`). Dyn top-4 = the **COMPLETE** field: ALL 8 dyn-able tiles ≤128 VGPR `{1x1,1x2,2x1,2x2,1x4,4x1,2x4,4x2}` × `{pf,dg}` × `BATCH{1,8,32}` = **48 dyn/shape** (`build_dynfull.sh`, grow-targets disasm-verified ≤128, 512³ oracle-gated 48/48 PASS). Raw: shapes 1-9 in `dynfull_results.txt`, o_pf in `o_pf_yield_test.txt` (re-run with the compositor-yield fix — see §7). **Bold = top-2 of each group → the 40 cells (10 shapes × 4) that advance to the rocprof/RGA bottleneck stage.**

| shape | TOP-4 STATIC (TF) | TOP-4 DYN — complete field (TF) |
|-------|-------------------|----------------------------------|
| down    | **2x8 18.8** · **4x4 16.5** | **2x4 pf b8 14.8** · **1x4 pf b32 13.9** · 2x4 dg b8 13.6 · 1x4 pf b8 13.3 |
| gate/up | **2x8 15.8** · **4x4 15.6** | **2x4 pf b32 14.5** · **2x4 dg b32 13.7** · 2x4 pf b8 12.8 · 1x4 pf b32 12.8 |
| attn_q  | **2x8 9.2** · **4x4 9.0** | **2x4 pf b8 7.3** · **1x4 pf b32 7.3** · 2x4 pf b32 7.2 · 2x4 dg b32 7.0 |
| attn_kv | **2x8 2.9** · **4x4 2.9** | **2x4 pf b8 2.1** · **4x2 pf b8 2.1** · 4x1 pf b8 2.0 · 2x2 pf b32 2.0 |
| attn_o  | **2x8 9.7** · **4x4 9.3** | **2x4 pf b8 7.6** · **2x4 dg b8 7.3** · 1x4 pf b32 7.3 · 1x4 pf b8 7.2 |
| down_pf | **2x8 6.6** · **4x4 6.6** | **2x2 pf b8 4.3** · **2x4 pf b1 4.3** · 1x4 pf b8 4.3 · 2x4 pf b8 4.3 |
| gtup_pf | **2x8 5.9** · **4x4 5.8** | **2x4 pf b8 4.5** · **2x4 dg b8 4.4** · 4x2 pf b8 4.4 · 2x4 pf b32 4.3 |
| q_pf    | **2x8 3.0** · **4x4 3.0** | **2x4 pf b8 2.1** · **4x2 pf b8 2.1** · 4x1 pf b8 2.0 · 2x2 pf b32 2.0 |
| kv_pf   | **2x8 0.8** · **4x4 0.8** | **2x4 dg b8 0.5** · **2x2 dg b1 0.5** · 2x2 pf b8 0.5 · 4x2 pf b1 0.5 |
| o_pf    | **2x8 3.0** · **4x4 3.0** | **2x4 pf b8 2.1** · **4x1 pf b8 2.0** · 1x4 dg b8 2.0 · 2x2 pf b8 2.0 |

**Field note (the complete field changes the read):** the dyn top-4 is **NOT** a `2x4` monoculture once the full tile×batch grid runs — `1x4`, `2x2`, `4x1`, `4x2` all reach the top-4, especially on the **small / prefill** shapes (attn_kv, q_pf, kv_pf, o_pf) where the smaller-footprint tiles get more concurrent fat waves and `pf b8` amortizes the grow tax. `2x4 pf b8` is still the most common dyn winner (6/10), but the runners-up are genuinely tile-diverse — so the dyn rocprof dives now compare **tile shape AND feed AND batch**, not just feed×batch on one tile. (The thin earlier field — 2x4 only at full batch — was an artifact, not the truth.) Still: best dyn loses to best static on every shape (gap −12%…−38%, same as the top-1 table below).

### Top-1 reference (winner per shape, from the field above)
Static refs in the fair race = 2x8 / 4x4. Dyn = 2x4/4x2 × {pf,dg} × {b1,b8,b32}, on its PROPER kernel.

| shape | M | K | N | **top static** | TF | **top dyn** | TF | dyn gap |
|-------|---|---|---|----------------|----|-------------|----|---------|
| down    | 2048 | 9216 | 2560 | 2x8 | **18.8** | 2x4 pf b8  | 14.6 | −22% |
| gate/up | 2048 | 2560 | 9216 | 2x8 | **15.8** | 2x4 pf b32 | 13.9 | −12% |
| attn_q  | 2048 | 2560 | 4096 | 2x8 | **9.2**  | 2x4 pf b32 | 7.3  | −21% |
| attn_kv | 2048 | 2560 | 1024 | 2x8 | **2.9**  | 2x4 pf b8  | 2.1  | −28% |
| attn_o  | 2048 | 4096 | 2560 | 2x8 | **9.8**  | 2x4 pf b8  | 7.6  | −22% |
| down_pf | 512  | 9216 | 2560 | 4x4 | **6.6**  | 2x4 pf b8  | 4.4  | −33% |
| gtup_pf | 512  | 2560 | 9216 | 2x8 | **5.9**  | 2x4 pf b8  | 4.5  | −24% |
| q_pf    | 512  | 2560 | 4096 | 2x8 | **3.0**  | 2x4 pf b8  | 2.1  | −30% |
| kv_pf   | 512  | 2560 | 1024 | 2x8 | **0.8**  | 2x4 pf b8  | 0.5  | −38% |
| o_pf    | 512  | 4096 | 2560 | 2x8 | **3.0**  | 2x4 pf b1  | 2.0  | −33% |

| shape | M | K | N | **top static** | TF | **top dyn** | TF | dyn gap |
|-------|---|---|---|----------------|----|-------------|----|---------|
| down    | 2048 | 9216 | 2560 | 2x8 | **18.8** | 2x4 pf b8  | 14.6 | −22% |
| gate/up | 2048 | 2560 | 9216 | 2x8 | **15.8** | 2x4 pf b32 | 13.9 | −12% |
| attn_q  | 2048 | 2560 | 4096 | 2x8 | **9.2**  | 2x4 pf b32 | 7.3  | −21% |
| attn_kv | 2048 | 2560 | 1024 | 2x8 | **2.9**  | 2x4 pf b8  | 2.1  | −28% |
| attn_o  | 2048 | 4096 | 2560 | 2x8 | **9.8**  | 2x4 pf b8  | 7.6  | −22% |
| down_pf | 512  | 9216 | 2560 | 4x4 | **6.6**  | 2x4 pf b8  | 4.4  | −33% |
| gtup_pf | 512  | 2560 | 9216 | 2x8 | **5.9**  | 2x4 pf b8  | 4.5  | −24% |
| q_pf    | 512  | 2560 | 4096 | 2x8 | **3.0**  | 2x4 pf b8  | 2.1  | −30% |
| kv_pf   | 512  | 2560 | 1024 | 2x8 | **0.8**  | 2x4 pf b8  | 0.5  | −38% |
| o_pf    | 512  | 4096 | 2560 | 2x8 | **3.0**  | 2x4 pf b1  | 2.0  | −33% |

**Across-the-board findings (all 10 shapes):**
- **Static wins every shape, every time.** Gap ranges −12% (gate/up) to −38% (kv_pf, the tiny K=2560 N=1024 prefill). Dyn never ties, never wins.
- **`2x8` is the static champion in 9/10** (4x4 only edges it on down_pf, 6.6 vs 6.5 — noise-close). Confirms the single-wave A-feed-locality story (2 A-streams beat 8 scattered).
- **`pf` (prefetch double-buffer) beats `dg` (DEFERGROW single-buffer) in all 10.** The smaller 96-VGPR footprint never pays for its exposed single-buffer feed. DEFERGROW is a dead lever.
- **BATCH `b8` is the dyn sweet spot in 8/10.** Exception: the two **big-N (N=9216)** shapes — gate/up and (slightly) — prefer `b32` (gate/up pf b32 13.9 > b8 12.8): more tiles ⇒ deeper batching amortizes the grow tax further. `b1` only wins on o_pf by rounding.
- The gap **widens as the GEMM shrinks** (prefill M=512 and small-N shapes): −30%…−38% on the prefill tail vs −12%…−28% on the M=2048 dense bodies. dyn's per-batch grow + single-wave occupancy overhead is fixed cost; the smaller the GEMM, the worse it amortizes.
- `dg b32` cells show occasional wild max-spread (gate/up [12..25], down_pf [4..17]) — unreliable, another mark against DEFERGROW.

**Top-1 per shape, locked:** static = **2x8** universally (down/gate/up/attn_*/all prefill); dyn = **2x4 pf b8**, except big-N gate/up & attn_q = **2x4 pf b32**.

---

## 6.5 NO-FEED CEILING — feed-bound vs framework-bound attribution (`--mbml8nf`, 2026-06-23)

**What:** each shape's top-4 static + top-4 dyn re-run on its OWN kernel with the per-K feed REMOVED
(operands loaded once, reused for all KT WMMAs — `NOFEED=1`). Isolates each config's **compute ceiling**
with DRAM bandwidth off the table → reveals whether dyn's loss is a **feed** problem or a **framework**
problem. Build `build_nofeed.sh` (20 bins; the `dg` path got a new `NOFEED` guard — feed removed, grow-tax
intact, grow-target disasm-verified 96 VGPR). Harness `--mbml8nf`. Raw: `nofeed_ceiling.txt`. NO BRICK
(compositor yield on). Oracle BAD by design (operands garbage) — this is a perf probe only.

**Best of each group, no-feed TF, vs the fed TF (§6 top-1):**

| shape | stat fed→nofeed | Δfeed | dyn fed→nofeed | Δfeed | **nofeed dyn vs nofeed stat** |
|-------|------|------|------|------|------|
| down    | 18.8→24.1 | **+28%** | 14.6→18.3 | **+25%** | dyn 18.3 < stat 24.1 (−24%) |
| gate/up | 15.8→16.3 | +3% | 13.9→**18.0** | **+29%** | **dyn 18.0 > stat 16.3 (+10%)** ← only shape dyn's ceiling wins |
| attn_q  | 9.2→9.7 | +5% | 7.3→8.3 | +14% | dyn 8.3 < stat 9.7 (−14%) |
| attn_kv | 2.9→3.0 | +3% | 2.1→2.1 | 0% | dyn 2.1 < stat 3.0 (−30%) |
| attn_o  | 9.8→10.7 | +9% | 7.6→8.3 | +9% | dyn 8.3 < stat 10.7 (−22%) |
| down_pf | 6.6→7.0 | +6% | 4.4→4.8 | +9% | dyn 4.8 < stat 7.0 (−31%) |
| gtup_pf | 5.9→6.2 | +5% | 4.5→4.7 | +4% | dyn 4.7 < stat 6.2 (−24%) |
| q_pf    | 3.0→3.0 | 0% | 2.1→2.1 | 0% | dyn 2.1 < stat 3.0 (−30%) |
| kv_pf   | 0.8→0.8 | 0% | 0.5→0.5 | 0% | dyn 0.5 < stat 0.8 (−38%) |
| o_pf    | 3.0→3.1 | +3% | 2.0→2.1 | +5% | dyn 2.1 < stat 3.1 (−32%) |

**The verdict (settles the Stage-2 question — dyn is framework-capped, not feed-capped):**
1. **Only the two big-K/big-N shapes are meaningfully feed-bound** (down +25–28%, gate/up-dyn +29%).
   The entire prefill tail + small shapes are framework-bound — removing the feed buys ~0–9%. Feed
   amortization has a NARROW payoff window: 2 shapes.
2. **dyn's extra occupancy does NOT convert to throughput — the smoking gun.** dyn runs **maxlive 1152
   vs static 768** (1.5× live waves) and burns **1.9–2.3 GPU-s vs 1.4–1.8**, yet on **9/10 shapes its
   no-feed ceiling is BELOW static's**. More waves + more GPU time + less work done = the grow/shrink/
   atomic tax eating the occupancy. The single-wave grow-tax is the wall, not memory.
3. **gate/up is the lone exception and a real lever.** It's the ONLY shape where dyn's compute ceiling
   (18.0) exceeds static's (16.3). In the fed race dyn LOSES (13.9 vs 15.8) **purely because dyn is
   feed-bound there (−29%) and static is not (−3%)** — dyn has the higher engine but is starved at the
   pump. Feeding it better (shared-A/B LDS reuse across pipelines) would unlock 18.0 and beat static
   by ~+14%. This is the one multi-wave/feed-sharing experiment the data endorses.

**Top-2 of each group per shape (no-feed ranking → the 40 rocprof candidates):** static = 2x8/4x4 on
every shape (8x2 on gtup_pf/kv_pf); dyn = 2x4 pf b8 + 2x4 dg b8 (down/attn_o/gtup_pf), 1x4/2x4 pf b32
(gate/up/attn_q big-N), small-tile pf b8 (prefill tail). Full field in `nofeed_ceiling.txt`.

> **Note (superseded by §6.6):** the "gate/up is the one shape dyn's ceiling beats static" reading above
> was a **static-b1 confound** — it compared dyn at b8/b32 against static at b1. Once static is also
> batched (§6.6), static reclaims gate/up. Keep the framework-vs-feed split; discard the dyn-beats-static-
> on-gate/up claim.

---

## 6.6 THE ATOMIC-CLAIM WALL → batched-static baseline + the reuse cap (2026-06-23)

Three measurements, in sequence. **Framing first (corrected):** none of this kills dyn-VGPR. dyn runs at
0% overhead (measured); it is *reuse-capped on the single-wave vehicle*. These runs (a) localize the
framework wall to ONE phase, (b) bank a free static win, (c) lock the honest baseline, and (d) show why
the **cooperative hybrid (§6.7)** is the vehicle where dyn's mechanism finally carries the load.

### (1) Phase split — the framework wall is ~entirely the atomic claim (`--mbml8prof`, `mbml8_phase_split.txt`)
In-kernel realtime timers (read as RATIOS — timers carry a ~70× perturbation), static 2x8 (b1) + dyn 2x4 (b8), fed:

| shape | config | ATOMIC | GROW | SETUP | COMPUTE | STORE | SHRINK |
|-------|--------|--------|------|-------|---------|-------|--------|
| down    | static 2x8 | **40%** | 0 | 0 | 60% | 0 | 0 |
| gate/up | static 2x8 | **36%** | 0 | 0 | 64% | 0 | 0 |
| q_pf    | static 2x8 | **36%** | 0 | 0 | 64% | 0 | 0 |

SETUP / STORE / GROW / SHRINK are all **~0%**. The bookkeeping wall is the **one device-scope `global_atomic`
work-queue claim** ("16384 grabs serialize the machine"), nothing else. (dyn's lower ATOMIC% on its rows is
the BATCH confound — dyn ran b8, 8× fewer claims — not dyn being better. GROW/SHRINK = 0% confirms dyn-VGPR
overhead is zero.)

### (2) Atomic-claim batch sweep — confirms the wall + a free static win (`--mbml8batch`, `mbml8_atomic_batch.txt`)
Best tile 2x8 + worst tile 1x1 (max claim density), BATCH{1,8,32}, real TF:
- **1x1 (worst) multiplies under batching** — gate/up **+566%** (2.1→14.0), attn_q +272%, attn_o +205%, down +100%.
  The atomic wall is real and huge; the lift lands by **b8** (b32 plateaus — atomic no longer binds).
- **2x8 (best): a free, brick-free win on the M=2048 bodies** — best batch per shape: down **21.0** (b8, +11%),
  gate/up **20.4** (b32, +28%), attn_q **11.0** (b8, +18%), attn_o **11.3** (b8, +15%), gtup_pf **6.7** (b8, +13%).
  Prefill tail (down_pf/q_pf/kv_pf/o_pf) keeps **b1** — batching coarsens work → load-imbalance when total tiles are few.
- **The tradeoff:** batching cuts atomics but coarsens distribution → shape-dependent sweet spot. This is the
  ceiling batching can't pass, and it's exactly what a run-ahead claim wave hides (atomic relief w/o coarsening).

### (3) Matched-batch fair race — the honest baseline; the gap is REUSE, not dyn (`--mbml8match`, `mbml8_matched_batch.txt`)
static 2x8 vs dyn 2x4, **both** batch-tuned (removes the b1 confound):

| shape | static best | dyn best | gap |
|-------|-------------|----------|-----|
| down | 20.96 | 14.36 | −31% |
| gate/up | 20.38 | 13.60 | −33% |
| attn_q | 10.96 | 7.49 | −32% |
| (all 10) | — | — | **−30…−34%, FLAT** |

**Read it right:** the gap is the **tile-size / reuse** difference, NOT a dyn-mechanism deficit. static 2x8 =
reuse 1.6 @ **208 VGPR**; dyn 2x4 = reuse 1.33, and it's 2x4 *because the no-umr ~128 VGPR dyn cap won't host
anything fatter*. The dead-flat −32% across all 10 is the signature of a structural reuse gap (same law as the
square data: tile size dominates, even feed-free). **dyn loses here because the single-wave vehicle confines it
to a smaller-reuse tile — not because s_alloc_vgpr underperforms.** That's the box, and breaking it is §6.7.

**Locked baseline (the number the hybrid must beat):** batched static 2x8 — down 21.0, gate/up 20.4, attn_q 11.0,
attn_o 11.3, gtup_pf 6.7; prefill tail at b1 (down_pf 6.6, q_pf 3.0, kv_pf 0.8, o_pf 3.0).

---

## 7. THE BRICKS + FIXES (2026-06-22)

### Brick #2 — COMPOSITOR STARVATION (the dynfull at-scale run, ~100 min in, on o_pf)
The complete-field `--mbml8dyn` run completed 9 shapes + o_pf through `2x4 pf b32`, then took down the
desktop. **It was NOT a kernel bug.** Kernel log (boot -2): `amdgpu 42:00.0 [gfxhub] page fault ...
Faulty UTCL2 client ID: SQC (data) PERMISSION_FAULTS:0x3` on `ring gfx_0.0.0 timeout`, Process
**Hyprland** → ring reset → "device wedged, but recovered through reset" → 2 of 3 monitors' compositor
died → reboot. Decisive evidence it's the compositor, not us: (1) fault client **SQC = scalar cache**,
but our kernel has **ZERO scalar memory loads** (all A/B/C are vector `global_load`/`global_store`;
queue counter is a vector `global_atomic`) — a vector fault is `TCP`/`TA`, never `SQC`; (2) the
timed-out ring is **gfx_0.0.0 (graphics)**, our PM4 work is on a **compute** queue; (3) **card1 (R9700)
drives 2 of the 3 monitors** (DP-1+DP-4; they can't move to the 6900XT — Thunderbolt login order).
ROOT CAUSE: the ~100-min dyn-VGPR grow/shrink storm starved the per-SIMD VGPR pool that the co-resident
Hyprland graphics waves need → a compositor shader got stale registers → bad scalar address → SQC fault.
It tipped on **o_pf 4x2 b32** because that's peak pressure: 4x2 grows to the full **128-VGPR** footprint
(fattest dyn tile) and b32 holds it for 32 tiles straight = longest sustained pool occupancy. The cell
wasn't buggy (its addressing is provably in-bounds; the same cell ran clean on 5 other shapes).

**FIX (the production-correct one — mirrors the torch `ML8_YIELD_*` fix, ported to PM4):** a **wall-time
compositor yield** in `run_mbgemm`'s rep loop. Between reps the dispatch is fully DRAINED (fence
signaled, waves exited, dyn-VGPR pool released), so a short host `nanosleep` there hands the gfx ring an
unconditional render+VGPR window. Env: `ML8_YIELD_MS` (sleep, default 5) / `ML8_YIELD_EVERY_MS`
(cadence, default 100) / `ML8_YIELD_DISABLE=1` (genuinely headless). **Zero TF impact** — throughput is
measured from per-rep IN-KERNEL gpu-clock spans, so a host sleep *between* reps never enters the number;
cost is ~5% wall. This is the right fix for OSS too (single-GPU users hit the identical starvation;
"don't compute on the display GPU" is a non-starter).

**VERIFIED:** re-ran o_pf only (`ML8_ONLY=o_pf ML8_YIELD_MS=5 ML8_YIELD_EVERY_MS=100 ./occ_dispatch
--mbml8dyn`) — all 50 cells incl. the exact ones that bricked (`2x4 dg b1/b8/b32`, `4x2 pf b8/b32`,
`4x2 dg b8/b32`) completed oracle-OK, **Hyprland alive throughout**, done-marker written, 0 problem
cells. Fix confirmed against the precise failure condition; o_pf data filled in (`o_pf_yield_test.txt`).

### Brick #1 — DEFERGROW load-then-grow race (earlier, fixed pre-dynfull)
`--mbml8dyn` v1 **bricked gfx1201** (MODE1 reset). Root cause: the original DEFERGROW path
**loaded operands then grew** (`LOADBUF` → `s_alloc_vgpr`). That races `s_alloc_vgpr` against the
VGPR write-back of the just-loaded data — the exact hazard banked from Phase-2
(*"s_alloc_vgpr races the in-flight VMEM and the grown registers come up wrong"*). Every working
path (Phase-2 occ_kernel.s, the prefetch dyn that ran clean all session) **grows first, then loads.**

**FIX:** reordered DEFERGROW to grow-first (frags still in lean block → keeps 96-VGPR footprint,
drops the unsafe load-while-lean). Verified: disasm shows `s_alloc_vgpr 0x60` BEFORE the v[11:12]
loads; `--mbml8gate` 512³ oracle PASS 128/128 on all 4 new bins (dg b1, 4x2 dg b1, pf b8, pf b32), no brick.

**SAFETY PROTOCOL (mandatory before any at-scale dispatch of a NEW kernel):**
1. Assemble + RGA gate offline. Disasm-verify grow ordering and register layout.
2. `--mbml8gate` 512³ oracle (sub-1s dispatch — too short to wedge) — confirms correctness + isolates path.
3. Only then escalate; watch one shape clear before the full sweep. Results tee'd to a persistent
   file (NOT /tmp — a reboot wipes /tmp).
- single-wave is NOT a brick-safety guarantee (L4 bricked single-wave via GPUVM page-fault at scale).
- Brick axes: load-then-grow race, GPUVM page-fault (addressing bug at scale), long/hung dispatch,
  multi-wave co-residency deadlock. SCC-retry guard on every `s_alloc_vgpr` (a failed grow retries,
  never runs on unallocated VGPRs). No umr (dyn tiles ≤128 VGPR).

---

## 8. STANDING CONCLUSION (dyn-VGPR throughput on ml8 fp8 GEMM)

dyn-VGPR runs correctly on gfx1201 under the real ml8 GEMMs (oracle-clean across **all 10 shapes**,
no brick once grow-first) — the proven-works trophy; HIP cannot arm `s_alloc_vgpr`, we do (PM4 RSRC2
bit6, MAD-304). As a THROUGHPUT lever it **loses to static on every one of the 10 ml8 shapes** — even
on its properly-tuned kernel (BATCH-amortized, DEFERGROW). The fair race is now decisive, not a
single-test call: gap −12% (gate/up) to −38% (kv_pf), median ≈ −26%. Deployment kernel for ml8
training/prefill stays **static**; the single-wave static champion is the fat **2x8** (9/10 shapes;
down 42 TF / gateup 34 TF steady-state @ M=131072). Best dyn config, if ever needed, is **2x4 pf b8**
(pf b32 on big-N gate/up/attn_q). The cooperative wggemm2 champion (112 TF real ml8 / 165 square)
remains the overall winner — a different (multi-wave LDS-cooperative) kernel.

**Stage-1 of the gauntlet is DONE** (tile sweep + sustained + fair race, all 10 shapes). **OPEN
(Stage-2):** RGA + dram_meter (GL2C_HIT / GRBM_TA_BUSY) on the top-2 each per shape to (a) attribute
*why* static wins — confirm the 2x8>8x2 single-wave A-feed-locality hypothesis, and (b) classify dyn's
bottleneck (grow-tax vs single-wave occupancy) to settle whether dyn is fundamentally capped here or
just under-tuned. dram_meter built (codex), not yet wired with GRBM/SQ counters.
