# MAD-305 Hybrid Cooperative fp8 GEMM — STATUS & RESUME (gfx1201 / R9700, wave32)

**Single source of truth. Read this first before touching the cooperative kernel. Updated 2026-06-24 — STATIC ORACLE GREEN.**
Companion design: `HYBRID_DESIGN.md` (Steps 0–8, locked). This doc = build state + debug journey + resolution.

---
## 0. TL;DR

The cooperative shared-B fp8 GEMM **STATIC oracle is GREEN** on silicon: `oracle 128×512×512: CLEAN ok=256 bad=0`,
dispatch **EXIT 0**, real EOP fence completes (no workaround). The full cooperative protocol works end-to-end.

**THE BUG (root cause, 2026-06-24):** ONE addressing slip in `occ_kernel_coop.s` prologue caused BOTH the
"24-hour store-hang" AND a numerical error. Lines 147/149 computed `v9` (LDS B-read vaddr) and `v10` (C-store
vaddr) from the **flat workgroup tid `v0`** instead of the **per-wave lane `v2`**. The compute wave (wid 1) has
`v0 = 32..63`, so `v0*8` shifted its LDS B-read by +256 B (read the wrong B column → only `(mi,0)` frags wrong,
2/tile, deterministic) and `v0*32` shifted its C-store by +1024 B (last frag landed at the buffer edge and
**never drained → s_endpgm/EOP fence hung**). Fix: `v9,3,v2` and `v10,5,v2`. That single change → oracle CLEAN
AND fence completes. **Diagnosis was Claude (localize: "ni=0 column, deterministic, settle-invariant") + Codex
(`codex:codex-rescue` pattern-matched the `v0`/`v2` slip), Claude verified vs symptom before applying.**

Key unlock that exposed the bug: the store-hang masked the oracle (it never ran). Forcing completion on the
kernel's own done-signal (`ML8_COOP_NOFENCE`, ignore the stuck EOP fence) let the oracle run → revealed the
numerical error → led to the addressing fix → which fixed the hang too. NOFENCE is now **unnecessary** (kept as
a harmless diagnostic env). The `global_wb` / terminal-store-wait experiments were dead ends (reverted/kept-benign).

**NEXT:** the **dyn (DYNVGPR=1) oracle** — the strategic moat (HIP can't arm `s_alloc_vgpr`). BRICK#4 class;
arm with care (SCC-retry guard, ≤128 VGPR, never touch umr BLOCK_SIZE). Static is GREEN, so dyn is unblocked.

---
## 1. Files & exact commands

| File | What |
|------|------|
| `occ_kernel_coop.s` | THE cooperative kernel. defsyms: `FM,FN,P,RINGD,BATCH,DYNVGPR,DIAG,RGADESC`. |
| `occ_dispatch.cpp` | harness. `run_mbcoop()` = the (1+P)-wave dispatch primitive; `--mbml8coop` mode. |
| `build_coop.sh` | builds `occ_coop_<FM>x<FN>_p<P>_r<RINGD>_b<BATCH>_d<dyn>_gd.bin` (+ `_rga.o`). |
| `HYBRID_DESIGN.md` | the locked design (Steps 0–8). |
| `coop_static_*.log` | tee'd dispatch logs from the debug session. |

**Build kernel cell (offline, safe):**
```
./build_coop.sh                 # builds the d0 (static) + d1 (dyn) bins, non-DIAG
# manual (with DIAG markers for debugging):
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,P=1 -Wa,-defsym,RINGD=2 -Wa,-defsym,BATCH=1 \
  -Wa,-defsym,DYNVGPR=0 -Wa,-defsym,DIAG=1 -c occ_kernel_coop.s -o X.o
/opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=.text X.o occ_coop_2x4_p1_r2_b1_d0_gd.bin
```
**Build harness:** `ROCM=/opt/rocm; PM4=../dvgpr_pm4; clang++ -std=c++17 -O2 -Wall -Wno-unused -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch`
(hsakmt IDE diagnostics are spurious; the .a links fine.)

**Dispatch (GPU — kmbandy gates EACH one):**
```
ML8_DYN=0 ML8_ONLY=down ML8_P=1 ML8_POOL=1 ML8_ORACLE_ONLY=1 timeout 60 ./occ_dispatch --mbml8coop | tee LOG
```
Env knobs: `ML8_P/RINGD/FM/FN/BATCH/POOL/DYN/ONLY/TGT/ORACLE_ONLY`. RGA: real rga at
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga` (NOT /usr/bin/rga=ripgrep).

---
## 2. The cooperative design (as built)

WG = `1 feed + P compute` waves (`(1+P)·32` threads). For the oracle: P=1, FM=2, FN=4, RINGD=2, BATCH=1.
Shape 128×512×512 → WG tile `TM=P·FM·16=32` M-rows × `TN=FN·16=64` shared N-cols, KT=Ko/16=32, TOTAL=32 tiles.

- **Feed wave (wid 0):** persistent atomic-claim (`global_atomic_add` on occ[5], offset 20) → per tile:
  publish `ti` to LDS + bump `epoch`; per K16-step `global_load_tr_b64` the FN B-frags → `ds_store` into a
  depth-RINGD ring → publish `prod` (MONOTONIC). Per-tile **drain**: wait `min_cons >= s55` (cumulative).
- **Compute wave (wid 1..P):** per tile read `ti` via epoch handshake (drain-enforced), SCC-retry grow (dyn),
  zero ACC; per K16-step busy-wait `prod > s56` → `ds_load_b64` shared B + `global_load_b64` direct A →
  `cons = s56+1` release → WMMA×(FM·FN); then full-frag fp32 store. **Terminal: count-to-TOTAL** (pool=1 the
  single WG processes exactly TOTAL tiles → exit at count==TOTAL; no cross-wave terminal signal).
- **LDS layout:** `B_ring[RINGD·FN·256] + prod_count(u32) + cons_count[P](u32) + ti(u32) + epoch(u32)`.
- **Init:** leader zeroes prod/cons/epoch, then ONE **symmetric pre-grow `s_barrier`** publishes it (SAFE: all
  waves lean-32 = same alloc = NOT the brick condition; wavespec uses the same idiom).
- **Counters are MONOTONIC (global step), never reset per tile** — this is the design's Step-3 intent. (My
  earlier per-tile reset was a deviation that caused a stale-read race; fixed.) Feed `s55`=global prod step,
  compute `s56`=global cons step; ring slot = step % RINGD.
- userdata ABI = identical to `occ_kernel_mbgemm.s` (A plain / B preshuffled-for-TR).

---
## 3. WHAT WORKS (proven on silicon, STATIC)

- ✅ **The cooperative shared-B consume completes ALL 32 tiles** (`occ[10]=1024` = 32·KT global cons steps).
  The barrier-free prod/cons B-ring + per-tile epoch/drain are CORRECT. This is the whole hard part.
- ✅ Address **BOUNDS GATE** in `run_mbcoop` passes every run (A 65535/65536, B 262143/262144, C 262143/262144
  — all land exactly at buffer-end; NO formula bug like the 2026-06-19 FM*FN hardcode).
- ✅ Feed claims all 32 tiles + terminal (`claim=33`), reaches its exit, `live--`, writes t1.
- ✅ RGA livereg (static): peak-live ~81 VGPR, no spills. (dyn cell: `s_alloc_vgpr 0x70`=112, off the 128 edge.)
- ✅ Bounds-gated + tiny-oracle + per-dispatch-gated = the static path never bricked across ~17 dispatches.

---
## 4. THE ONE OPEN BUG — last-tile C-store hang

**Symptom:** compute consumes all 32 tiles' B (`occ[10]=1024`) but completes only 31 (`occ[11]=31`).
`occ[12]=1024` proves it **exited the last tile's K-loop** → stuck in the **C-store region**; the only spin
there is `s_wait_storecnt 0x0`. The C address for ti=31 is in-bounds (bounds gate green). Clean timeout, GPU
survives. Manifests identically static & non-DIAG. **Only the LAST tile hangs; tiles 0..30 store fine.**

**RULED OUT (each by a dedicated dispatch):**
1. ❌ **DIAG markers** — non-DIAG build (no markers) hangs identically. Markers innocent.
2. ❌ **Feed-wave retirement stalling the survivor's store** — added a final symmetric `s_barrier` so the feed
   PARKS (alive) instead of `s_endpgm`; `live` went 0→1 (feed parked) but compute STILL hangs in the store.
   So it's not the feed retiring. (Barrier reverted; didn't help.)
3. ❌ **Cooperative sync / B-handoff** — consume fully completes (occ[10]=1024).
4. ❌ **Per-tile counter reset race** — replaced with monotonic counters; symptom unchanged (occ[11]=31), i.e.
   the race wasn't the cause of THIS hang (but monotonic is still the correct form, kept).
5. ❌ **Address formula / OOB** — bounds gate green; not a brick.

**REMAINING SUSPECTS (for tomorrow):**
- storecnt accounting specific to the **32nd** tile.
- the C-store landing at the **very END of the C buffer** (bytes 253952..262143 of a 262144 buffer).
- a static **fat-launch (128 VGPR)** interaction with the final store.

**NEXT MOVES (sharpest first):**
1. Add `occ[13] = s56` **right after** `s_wait_storecnt` → confirm that wait is the exact stall (occ[13]=992
   ⇒ hangs in the wait; 1024 ⇒ stall is after).
2. Add an env to **shrink the oracle tile count** (e.g. `ML8_OTILES`): if **TOTAL=1** also hangs in the single
   tile's store, it's "store after the feed claims terminal," NOT "last of many" — sharpest cut.
3. Check whether the feed's **terminal claim/reset** touches anything the compute's store path needs.
4. If localized to the store + feed-terminal interaction: try deferring the last store, or a store fence
   (`s_waitcnt`/scope), or have the feed not claim terminal until compute signals (count-based, pool=1).

---
## 5. Hard facts learned (don't re-derive)

- **gfx1201 gives NO release/acquire ordering for barrier-free cross-wave LDS.** Two LDS words written by one
  wave can become visible to another wave **out of order** (epoch visible before ti). And **LDS vs global are
  not cross-ordered**: the compute saw the feed's LDS epoch bump (33) BEFORE the feed's global claim (33),
  even though the feed did claim→epoch in that order (`occ[10]=32` evidence). ⇒ a global-claim terminal check
  is unreliable when routed by an LDS epoch. Termination must NOT depend on cross-memory ordering.
- The **per-tile drain** is what makes the epoch/ti handshake reliable for real tiles (the feed waits for the
  compute's cons = a forced round-trip that enforces visibility). The **terminal** has no drain → its LDS
  signal is unreliable → use **count-to-TOTAL** (pool=1) instead. This is the working pattern.
- **Monotonic prod/cons (never reset per tile)** is the correct counter form (design Step-3). Per-tile reset
  introduces a stale-read race in the drain.
- atomic-MAX DIAG markers are reliable for "furthest point reached" but **ambiguous across tiles** (can't
  isolate the last tile). Use a value-carrying marker (e.g. global step `s56`) to disambiguate.

---
## 6. GPU SAFETY (non-negotiable — caused a brick this session)

- **THE R9700 (42:00.0) drives 2 of 3 desktop monitors. A bad GPU address wedges the desktop via the shared
  amdgpu driver.** The 2026-06-23 brick: `sq_intr` shader exception → `GCVM_L2_PROTECTION_FAULT` (TCP client,
  OOB global access) → MES unrecoverable → MODE1 reset → VRAM lost → reboot. Root cause that time: a garbage
  value flowed into a global address (compute read garbage `ti` from a racy LDS read → `mul_hi` → OOB A/C addr).
- **RULE: kmbandy gates EVERY individual GPU dispatch.** One "go" ≠ a session of auto-firing. A hang on an
  unproven kernel = FULL STOP + report, do NOT auto-try the next variant.
- **RULE (2026-06-19, re-confirmed): mirror the kernel's load/store offset formulas in a pure-CPU bounds check
  and REFUSE to dispatch if any access exceeds its buffer.** This is the `run_mbcoop` ADDRESS BOUNDS GATE.
- Static (DYNVGPR=0) is brick-safe (only hangs). Validate on static first; layer dyn after, with the
  BRICK#4 discriminating ladder (RGA-gated, headless).
- NEVER pass `--gl2c` (MES landmine). NEVER touch umr `SQ_DYN_VGPR.BLOCK_SIZE` (hard-wedges). dyn tiles ≤128 VGPR.

---
## 7. dyn brick (DEFERRED, separate problem)

Earlier dyn (DIAG) run bricked; the static run only hung. Either the brick is dyn-specific (BRICK#4 class) OR
the nondeterministic garbage→OOB landed on a dyn run by luck. CANNOT distinguish yet. Either way: get the
STATIC oracle GREEN first, THEN isolate the dyn brick with the BRICK#4 ladder (H1 prologue-at-lean / H2 SCC
grow / H3 mixed-alloc barrier), headless, RGA-gated.

---
## 8. Build order (HYBRID_DESIGN.md Step 8) — where we are

B0 ✅ SIMD probe ([1,1,1,1] confirmed). B1 ✅ harness. B2a ✅ kernel authored. B3 ✅ RGA. **B4 = oracle (HERE —
one store-hang bug from GREEN on static).** B2b (P=2/3) / B5 (supervised escalation) / B6 (sweep + dyn) after.
