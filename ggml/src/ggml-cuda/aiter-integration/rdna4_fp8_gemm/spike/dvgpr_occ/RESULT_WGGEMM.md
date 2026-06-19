# RESULT — RDNA4 dyn-VGPR wave-group fp8 WMMA GEMM

> ## ⚠️ CONTAMINATION MARKER — read before trusting any perf number below
> **All PERFORMANCE/BANDWIDTH data recorded before the "ROOT CAUSE FOUND" section used operands in
> SYSTEM/GTT RAM, read by the GPU over PCIe (~25 GB/s), NOT device-local VRAM (640 GB/s).** `AllocGpu` had
> `NonPaged=0` (inherited verbatim from the MAD-304 dyn-VGPR probe, where bandwidth never mattered). The
> entire "FED wall" — the 1.4 TF / 2.7 GB/s feed, rungs 1–9, the stack ladder, the lockstep hypothesis,
> the bandwidth hunt — was an artifact of PCIe-fed operands. **Treat all of it as INVALID for
> bandwidth/feed conclusions.** Correctness work (oracle/acc00/equivalence) still stands. The fix
> (`NonPaged=1` device-local VRAM) took the real GEMM 1.4 → 79 TF (56×). Honest baselines start at the
> "VRAM RE-BASELINE" section. A `run_wggemm_perf` guard now aborts if perf operands aren't VRAM.

**HW:** AMD Radeon AI PRO R9700 / gfx1201 (RDNA4, wave32), ROCm 7.2 (HIP 7.2.53211), clang 22.
**Note:** box also has an RX 6900 XT (gfx1030). The bench is compiled `--offload-arch=gfx1201`
only, so it can *only* load on the R9700 — confirmed by the fp8-WMMA oracle gate passing
(err=0.0038). `rocm-smi` default-device display showing the 6900 XT is cosmetic.

Ceiling reference (from `bench/wmma_peak.hip`, prior work): **307 TF = 15.9 WMMA/cycle** fp8 e4m3,
feed-free matrix-issue ceiling. hipBLASLt on these shapes ≈ 143 TF.

---

## Phase 0 — Anchor the Reference (DONE)

### 0.1 HIP winner baseline numbers (the structure the PM4 kernel must clone)

Winner = `gemm_fp8_lever<TBM=128,TBN=128,TWM=2,TWN=2,TBK=32,AFEED=0,DBUF=1>`:
4 waves/block (128 threads), logical **128×128** C tile, **per-wave 4×4** frag accumulator
(`TFRAGS_M=TFRAGS_N=4` → 16 frags → 128 acc VGPR), A staged in **LDS** (double-buffered),
B fed direct via **`global_load_tr_b64`** from a preshuffled tile-major buffer.

Measured this session via `ISSUE_PROBE=1 ./bench/gemm_fp8_levers` (30-iter avg):

| shape | kernel | TF | % of 307 | WMMA/cyc | VGPR | occ |
|---|---|---:|---:|---:|---:|---|
| 4096² × K16384 | **lever 4×4 dbuf (winner)** | **161.1** | 52.5% | 8.34 | 183 | 5 blk / 20 wv |
| 4096² × K16384 | pipe A+B deep | 63.4 | 20.6% | 3.28 | 219 | 5 blk / 20 wv |
| 4096² × K16384 | NOFEED (feed-free, same struct) | 272.0 | 88.6% | 14.09 | 151 | 5 blk / 20 wv |
| 4096×14336 × K16384 | **lever 4×4 dbuf (winner)** | **145.4** | 47.4% | 7.53 | 183 | 5 blk / 20 wv |
| 4096×14336 × K16384 | pipe A+B deep | 73.6 | 24.0% | 3.81 | 219 | 5 blk / 20 wv |
| 4096×14336 × K16384 | NOFEED (feed-free, same struct) | 278.0 | 90.6% | 14.40 | 151 | 5 blk / 20 wv |

Occupancy sweep `minWaves = 4→8` is **flat** (~160 TF on the square shape): the wall is the shared
VALU issue port, **not** latency / occupancy. `minW=10` spills (VGPR 144, spill 248 → 24 TF) and is
confounded — excluded from the verdict.

**The gap to close:** NOFEED 272 vs fed 161 ⇒ operand feed + address math + waitcnts steal **~41%**
of throughput by competing for the single VALU issue port. The wave-group dyn-VGPR thesis is to cut
*feed issues per useful WMMA* (bigger logical reuse tile) while keeping resident waves up.

### 0.1a G2 parity gate (set against these measured numbers; GPT 5.5-confirmed)

The G2 HIP-parity gate uses the **K16384** numbers above (not the older "155" headline, which was the
K=8192 peak). Run **square 4096² × K16384 first** (cleaner tile-grid decode), then 4096×14336 second
once square static parity is close.

| shape | parity bar | hard pass (≤5%) | acceptable first pass (≤10%) | below |
|---|---:|---:|---:|---|
| 4096² × K16384 | 161.1 TF | **≥153 TF** | ≥145 TF | hand-asm scheduling/codegen debt, *not* a dyn-VGPR result |
| 4096×14336 × K16384 | 145.4 TF | ≥138 TF | ≥131 TF | (run as 2nd gate) |

### 0.2 The winner's steady-state hot loop (ISA, for transcription)

Extracted from `hipcc -O3 --offload-arch=gfx1201 -save-temps=obj`. Full kernel saved to
`/tmp/cg/winner_kernel_4x4_full.s`; isolated 32-WMMA window to `/tmp/cg/winner_hotloop_4x4.s`.

One tile iteration = **2 kk-steps × 16 frags = 32 WMMAs**, fully unrolled, in **16 independent
accumulator chains** (`v[1:8]` … `v[121:128]` = 128 VGPR of live accumulator):

```asm
        s_wait_loadcnt_dscnt 0x700            ; wait LDS-A (dscnt=7) for this tile
        v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[149:150], v[158:159], v[121:128]
        s_wait_loadcnt 0x6                    ; release B frag, one at a time …
        v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[149:150], v[160:161], v[113:120]
        s_wait_loadcnt 0x5
        v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[149:150], v[162:163], v[105:112]
        s_wait_loadcnt 0x4
        v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[149:150], v[172:173], v[97:104]
        v_wmma ... (12 more, no waits — operands already resident)   ; finish kk=0 group
        s_wait_loadcnt 0x3
        v_wmma ... v[151:152] x v[170:171]                           ; kk=1 group
        s_wait_loadcnt 0x2
        v_wmma ...
        s_wait_loadcnt 0x1
        v_wmma ...
        s_wait_loadcnt 0x0
        v_wmma ... (12 more, no waits)
        v_add_nc_u32_e32 v181, 32, v181      ; A LDS ptr += 32  (pointer increment, hoisted)
        v_add_nc_u32_e32 v157, s10, v157     ; B ptr += stride
        s_addk_co_i32 s16, 0x100             ; scalar tile counters
        s_addk_co_i32 s17, 0x1000
        s_cmp_eq_u32 s1, s18
        s_cbranch_scc1 .LBB_exit             ; clean counted back-edge
```

The B `global_load_tr_b64` (8/tile) and A `ds_load` (for tile t+1) are **issued in the prefetch
block ahead of the next compute window** — they do not appear inside the 32-WMMA window, which stays
contiguous and shows only the wait ladder. The loop block order is: **prefetch block → branch → fine
wait ladder + 32-WMMA run (contiguous) → pointer increments → loop control.** Loads are hidden in
*time* but still consume *issue slots* (the source of the issue-port bound).

**Codegen target for the PM4 wave-group kernel (the transcription contract; GPT 5.5-reviewed):**
1. **Preserve the block structure** — prefetch block, branch, fine wait ladder, the **contiguous**
   32-WMMA run, pointer increments, loop control. Do **not** literally interleave loads inside the
   back-to-back WMMA run; keep the WMMA window contiguous and let the prefetch sit in its own block.
2. 16 WMMAs back-to-back per kk in independent acc chains — no branch inside the issue run.
3. A from LDS, gated by `dscnt`; B from `global_load_tr_b64`, gated by a **fine descending
   `s_wait_loadcnt` ladder** (`0x6…0x0`), not one coarse barrier. (PM4 `WMMABUF_WAIT` already does this.)
   Keep A on `dscnt` and B on `loadcnt` for G2 — **no B-LDS, no b128, no dynamic tiles before parity**;
   mixing counters/feeds makes failures undiagnosable.
4. Treat `s_wait_loadcnt_dscnt 0x700` and the `0x6→0x0` ladder as **observed scheduling constants**,
   not fully-understood semantic labels — the waitcnt subfields are easy to misread. Copy the pattern
   *structurally* for G2 and validate by oracle + perf, then refine.
5. **Honor `s_wait_alu` hazards.** The compiler inserts `s_wait_alu` around VALU-produced addresses
   before the loads/branches that consume them; hand-PM4 has no compiler to do this. Prefer hoisting
   address-gen far ahead of the consuming `global_load_tr`/`ds_load`; otherwise insert the matching
   `s_wait_alu`, or risk silent stalls / bad scheduling.
6. Operand addresses as **pointer increments**, all address math hoisted out of the WMMA run.
7. **Preserve accumulator reuse spacing** — a given acc chain (e.g. `v[121:128]`) is reused only after
   a full 15-WMMA gap (it starts kk0 and kk1 16 WMMAs apart). That spacing is part of the latency
   hiding, not incidental; keep the 16-chain stride.
8. Counted loop via scalar compare + single back-edge.

### 0.3 Steady-state feed density (per tile iteration)

| class | count / tile | gated by |
|---|---:|---|
| `v_wmma_f32_16x16x16_fp8_fp8` | 32 | — |
| `global_load_tr_b64` (B) | 8 | loadcnt |
| A `ds_load` (LDS) | ~8 | dscnt |
| `s_wait_loadcnt` / `_dscnt` | 9 | — |
| ptr-advance `v_add` + scalar ctrl | ~6 | — |

≈ 31 non-WMMA issues / 32 WMMAs. Halving feed/WMMA (the dyn-VGPR fat-tile lever) is what could move
161 → toward the 272 feed-free ceiling, **if** residency can be held — the wave-group hypothesis.

**Exit criteria met:** concrete 4×4 HIP ISA snippet available for transcription; baseline TFLOPS and
WMMA/cycle recorded for both wgrad shapes.

---

## Phase 1 — 4-wave cooperative workgroup skeleton (DONE, pivoted)

### 1.0 PIVOT: raw-PM4 TGID is unavailable; tiles are distributed by atomic-claim + LDS-broadcast

The first Phase-1 attempt used a **grid-stride over `TGID_X`** tile claim (cheaper, no LDS). It failed:
the SGPR probe (`occ_kernel_wgdiag.s`, `--sgpr-probe`) proved **`TGID_X` is delivered to NO SGPR** on
this raw-PM4 `DISPATCH_DIRECT` path — 8 distinct workgroups ran (atomic counter=8) but `s8..s23`
showed `s8-s15`=0 and `s16-s23` constant across all 8 ordinals; no SGPR carried a `{0..7}`
permutation. The RSRC2 encoding is **correct** (a minimal KD with `.amdhsa_system_sgpr_workgroup_id_x 1`
assembles to `RSRC2=0x80` = bit7 `TGID_X_EN`, which our `0x289e` sets) — so it is a raw-PM4 delivery
gap, not a bit error. **Latent** because the entire existing harness distributes work via a global
atomic work-queue and never relied on TGID (mbgemm reads `s15` only for `STAGGER`, default off).

**Canonical decision (GPT 5.5 + user approved):** grid-stride scheduling is **abandoned**;
**atomic-claim + LDS-broadcast** is the canonical PM4 wave-group tile distribution. The leader
atomic-claims `ti`, broadcasts it to the workgroup's waves via **LDS + barrier**, and all waves
process the same tile — exactly the LDS+barrier machinery Phase 2's A-sharing needs anyway.

### 1.1 LDS config in raw PM4 (solved)

`COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE` = bits **[15:23]** (9-bit), per `amd_hsa_kernel_code.h:122`
and LLVM `AMDHSAKernelDescriptor.h`. The assembler leaves it 0 in the KD (the HSA loader fills it at
load time) — so for raw PM4 the host must compute and program it: `units = ceil(ldsBytes/512)`
(gfx10/11/12 granule = 512 B), `RSRC2 |= (units & 0x1FF) << 15`. (`occ_dispatch.cpp:ldsRsrc2Bits`.)
Host prints request/granule/units/alloc/RSRC2 before every dispatch. **TODO:** re-confirm the 512 B
granule before Phase 2's ~12 KB A-tile (the smoke uses 16 B → `units=1`, safe under any granule).

### 1.2 Smoke result (`occ_kernel_wglds.s`, `--wglds-smoke`)

Persistent 4-wave (128-thread) workgroups; leader `atomic_add`-claims a tile → `ds_store` to LDS[0]
→ `s_barrier` → every wave `ds_load`s `ti` → lane-0 of each wave writes
`mark=(tile_row<<20)|(tile_col<<8)|(wave_m<<4)|wave_n` to `C[ti*4+wid]`. No TGID, no compute, no dyn-VGPR.

| shape | workgroups (maxlive) | tile claims | wave marks | verdict |
|---|---:|---:|---:|---|
| 512² (16 tiles), nWG=4 | 4 | 20 (=16+4) | 64/64, bad=0, miss=0 | **PASS** |
| 1024² (64 tiles), nWG=16 | 16 | 80 (=64+16) | 256/256, bad=0, miss=0 | **PASS** |
| 4096² (1024 tiles), nWG=64 | 64 | 1088 (=1024+64) | 4096/4096, bad=0, miss=0 | **PASS** |

`RSRC2=0xa89e` (LDS bits `0x8000`, 512 B). Claims = TOTAL + nWG (each WG does one final drain-detect
claim ≥ TOTAL). Bit-exact mark verification IS the coverage proof: every tile claimed exactly once,
all 4 waves wrote the correct `(wave_m, wave_n)`, GPU healthy throughout.

**Proves:** (a) raw-PM4 multi-wave workgroups form correctly (4 waves/WG); (b) workgroup **LDS
allocates + reads back**; (c) the **workgroup barrier works**; (d) **atomic-claim** distribution covers
each tile exactly once; (e) the lane/wave mapping is correct. The earlier "barrier hang" was an
uninitialized-VGPR atomic-vaddr bug (OOB fault), not a workgroup-config limit — multi-wave PM4 is fine.

**Exit criteria met:** workgroup launches with 4 waves; tile decode covers every tile exactly once;
smoke proves all 4 waves active. Phase 2's LDS+barrier prerequisites are de-risked.

## Phase 2 — A-LDS fill + per-wave 4×4 static compute (DONE, bit-exact)

`occ_kernel_wggemm2.s` + `--wggemm-compute`. The full 4-wave cooperative fp8 GEMM on raw PM4, on the
proven claim/broadcast/LDS/barrier foundation. **No dyn-VGPR, no B-LDS, no b128** (static G2 first).

Per claimed 128×128 tile: leader atomic-claims `ti` → LDS-broadcasts it; cooperative A[128][32] fill
into LDS (each of 128 lanes 2× `global_load_b128` → `ds_store_b128`; LDS dst = `tid*16`, `tid*16+2048`);
barrier; per kk∈{0,1}: 4 A-frags from LDS (`ds_load_b64`, `ldsbase = wave_m*2048 + (lane&15)*32 +
colhi*8`, +`mi*512+kk*16`) + 4 B-frags (`global_load_tr_b64`, `nt_base = tile_col*8 + wave_n*4`) →
16 WMMAs; barrier; store each wave's 16 frags FLAT to `C[ti*65536 + wid*16384 + frag*1024]`.

Instruction mix (per K-tile iter): 32 `v_wmma`, 8 A `ds_load_b64`, 8 B `global_load_tr_b64`, 2 A-fill
`global_load_b128`+`ds_store_b128`, 32 C-store `b128`, 3 barriers. Static 192 VGPR, LDS 4100 B (A at 0,
ti at 4096; units=9, boundary-tested). Coarse `s_wait_loadcnt/dscnt 0x0` before WMMA (fine ladder = Phase 3).

Oracle (`run_wggemm_compute`): chained `wmma_ref` per frag + `unpack_D`, full-check every frag:

| shape | tiles | claims | frags | verdict |
|---|---:|---:|---:|---|
| 256×256×256 | 4 | 8 | 256/256, bad=0 | **PASS** |
| 512×512×512 | 16 | 24 | 1024/1024, bad=0 | **PASS** |
| 512×512×2048 (big-K) | 16 | 24 | 1024/1024, bad=0 | **PASS** |

GPT 5.5-reviewed (kernel + oracle + disasm) before the run. **Exit criteria met:** oracle bit-exact on
small + big-K shapes; objdump structure correct. The wave-group fp8 GEMM is a correct PM4 vehicle.

## Phase 3 — G2 HIP-parity gate

### 3.0 BASELINE (un-tuned: single-buffer A, coarse waits, STORE=0 minimal store)

`occ_kernel_wggemm2.s` built `-defsym STORE=0` (acc[0][0]-only store → 16× less store traffic than the
diagnostic layout, so store traffic doesn't mask compute). `--wggemm-perf`, in-kernel realtime span,
nWG=256 (maxlive=192 WGs resident):

| shape | TF | % of 307 | WMMA/cyc | maxlive | acc[0][0] |
|---|---:|---:|---:|---:|---|
| 1024² × K2048 (sanity) | 1.1 | 0.4% | 0.06 | 192 WGs | 64/64 ✓ |
| 2048² × K4096 (medium) | 1.4 | 0.4% | 0.07 | 192 WGs | 64/64 ✓ |
| **4096² × K16384 (TARGET)** | **1.1** | **0.4%** | **0.06** | 192 WGs | 64/64 ✓ |

**Correct but ~140× below the 161.1 TF bar — structural, not waitcnt polish** (the "catastrophically
low" decision branch). Diagnosis: single-buffer A + coarse `s_wait_loadcnt/dscnt 0x0` before *every*
kk-step ⇒ each kk pays the full B `global_load_tr` latency (~400 cyc) **unhidden**; the **2 barriers /
K-tile (≈1024 / output tile at K16384)** compound it — waves stall on loads *before* each barrier, so
the barrier waits for the slowest, and the workgroup marches in lockstep so occupancy (192 WGs) can't
hide the latency. This is the **expected un-tuned baseline**, NOT a wall: the HIP winner reaches 161 TF
at the *same* barrier frequency because it **double-buffers** (prefetch tile t+1's A into a 2nd LDS
buffer + B into registers during tile t's WMMAs), hiding the load latency and letting waves reach each
barrier without stalling.

**Next (Phase 3 tuning):** A double-buffer (ping-pong LDS) + B register prefetch + the fine descending
`s_wait_loadcnt` ladder (the `WMMABUF_WAIT` pattern) — the HIP-winner structure — re-measuring each
increment. Then judge against the 153 TF hard-pass / 145 acceptable bar.

## Phase 3 tuning

### 3.1 Step 1: A double-buffer only (DBUF=1) — DID NOT MOVE THE NUMBER

A ping-pong LDS (`As[2][128×32]`, 8 KB) + prefetch A(t+1) during compute(t) + 1 barrier/K-tile
(vs 2). B kept just-in-time, coarse waits. Oracle re-verified bit-exact (256/256, 1024/1024,
1024/1024). `STORE=0` perf @ target = **1.4 TF (0.4%)** — **unchanged** from the 1.1 TF baseline.
(Implementation note: the first cut used `s_wait_loadcnt 0x2` to keep the A-prefetch in flight; that
left B frags stale on the last K-tile → oracle 247/256. Fixed by waiting B `0x0` *before* issuing the
prefetch, so the prefetch still overlaps the 32 WMMAs but B is exact.)

### 3.2 NOFEED compute-ceiling probe — localizes the wall to FEED, not the vehicle

`NOFEED=1`: fill A once, read 8 A + 8 B frags once, K-loop = 32 WMMAs only (no per-K feed, no
barriers). Result is garbage (acc00 0/64) — perf-only:

| shape | NOFEED TF | % of 307 | WMMA/cyc | (fed, for ref) |
|---|---:|---:|---:|---:|
| 1024² × K2048 | 3.3 | 1.1% | 0.17 | 1.1 (small/underfilled) |
| 2048² × K4096 | 16.6 | 5.4% | 0.86 | 1.4 |
| **4096² × K16384** | **104.0** | **33.9%** | **5.39** | **1.4** |

**Reading:** the wave-group **compute path is sound** (104 TF resident-operand), so A-dbuf-not-moving
is NOT "the vehicle is broken." The wall is the **feed + barriers**: 104 → 1.4 TF = a **~74× feed
penalty** (HIP winner's is ~1.7×). Cause = **B `global_load_tr` latency unhidden** (coarse JIT wait)
+ per-K-tile barriers amplifying it (waves stall variably on B → barrier waits for the slowest). A-dbuf
can't fix this (A wasn't the bottleneck). **Next lever = B register prefetch + fine `s_wait_loadcnt`
ladder (Step 2)** to hide B latency so waves reach the barrier without stalling — exactly the
HIP-winner structure. (Secondary: NOFEED 104 vs HIP-NOFEED 272 ⇒ the WMMA scheduling/ILP also has
~2.6× headroom — the fine-ladder back-to-back run addresses part of that too.)

## Phase 3 Step 2 — B prefetch + barrier attribution (all flat; fine ladder is the remaining lever)

`occ_kernel_wggemm2.s` defsyms `BDBUF` (B register ping-pong, unroll-by-2, prefetch B(t+1) one K-tile
ahead) and `NOBAR` (skip per-K-tile barriers, perf-only). All oracle-gated where correct.

| variant @ 4096²×K16384 | TF | %307 | WMMA/cyc | removes |
|---|---:|---:|---:|---|
| baseline (single-buf, coarse) | 1.1 | 0.4% | 0.06 | — |
| A double-buffer (DBUF=1) | 1.4 | 0.4% | 0.07 | A-latency (prefetch) |
| B double-buffer (BDBUF=1) | 1.3 | 0.4% | 0.07 | B-latency (prefetch) |
| NOBAR (DBUF=0, barriers off) | 1.3 | 0.4% | 0.07 | the per-K-tile barriers |
| **NOFEED** (no feed, no barriers) | **104** | **33.9%** | **5.39** | ALL per-K-tile feed |

**Attribution:** prefetching A, prefetching B, and removing the barriers **each do nothing** (~1.4),
yet removing the *entire* per-K-tile feed jumps to 104 TF. ⇒ The kernel is **feed-latency-bound and
occupancy is not hiding it**, and coarse-wait prefetch doesn't reduce the *effective* per-K-tile stall
(each K-tile still waits ~400 cyc for the 8 B frags / A reads via `s_wait …0x0`). At ~6 waves/SIMD
(192 WGs × 4 / 64 CU / 2), the SIMD can overlap ~384 cyc — just short of the ~400 cyc all-frags wait,
so the latency is exposed and compounds over NTILES.

**The one untried lever — the fine descending `s_wait_loadcnt` ladder** (the HIP winner's
`WMMABUF_WAIT`): release B frags individually (`0x6→0x0`) so the first WMMA fires when frag-0 lands
(~50 cyc) instead of waiting all 8 (~400 cyc). That cuts the effective per-K-tile wait ~8×, which the
existing occupancy *can* cover — the mechanism by which the HIP winner reaches 161 at ~8 waves/SIMD.

## Phase 3 Step 2.2 — fine `s_wait_loadcnt` ladder (BLADDER): FLAT → B was never the bottleneck

`occ_kernel_wggemm2.s` defsym `BLADDER` (build with `DBUF=0`). Single-buffer A (JIT each K-tile);
**all 8 B frags issued up front** (loadcnt→8) right after the A-fill barrier, then the A-frag `ds_load`
runs (waited on `dscnt` **only** — never mixed with B's loadcnt), then a **fine descending ladder**
`s_wait_loadcnt 0x7→0x0` releases B frag-by-frag, each step followed by its 4 WMMA (two back-to-back
16-WMMA `kk` groups, ni-outer/mi-inner). Disasm-verified: `0x7,0x6,…,0x0`; same 16-WMMA accumulator
reuse spacing as the proven baseline. **Oracle: PASS bit-exact** (256³, 512³, 512²×K2048 — 2304/2304
frags, 0 bad).

| variant @ 4096²×K16384 | TF | %307 | WMMA/cyc | acc00 |
|---|---:|---:|---:|---|
| coarse B-dbuf (Step 2.1) | 1.3 | 0.4% | 0.07 | ok |
| **fine ladder (BLADDER)** | **1.3** | **0.4%** | **0.07** | ok |
| (sanity 1024²×K2048 / medium 2048²×K4096) | 0.9 / 1.3 | — | — | ok |

**Read (not a wall — the ladder worked):** the ladder is correct and *does* hide B, but B feed was
never the wave-group bottleneck. Because the 8 B loads are issued before the A-frag `ds_load` + `dscnt`
wait, frag-0 has essentially landed by the first `0x7` and `0x6→0x0` hide behind WMMA work — so B is
fully covered, hence flat. The exposed cost is the **per-K-tile A round-trip** (A-fill global load →
`ds_store` → barrier → A-frag `ds_load` → barrier), a serial **latency chain** the B ladder cannot
touch. This matches the attribution: A-prefetch, B-prefetch, B-ladder, and NOBAR each ~flat; only
NOFEED (removes the whole chain) reaches 104. At ~6 waves/SIMD (192 VGPR static) the SIMD is just short
of hiding that chain.

## Phase 3 Step 2.3 — 2×2 + NOFEED@2×2: OCCUPANCY FALSIFIED; the ceiling is WMMA-issue-density bound

`occ_kernel_wggemm2.s` parameterized by `FM`/`FN` (derived shifts `AROW_SH/TROW_SH/TCOL_SH/WN_SH`,
tile-aware `TI_OFF`, compacted frag bases `FA`/`FB`). 2×2 = per-wave 32×32 quadrant, 64×64 claimed
tile, acc v32-63 + fa v64-71 + fb v72-79 → **max v79 (~96-VGPR reservation) → all 256 launched WGs
resident** (maxlive 256 vs 192 at 4×4). Harness `--wggemm-2x2` (`run_wggemm_*` parameterized by
`FMt`/`ldsBytes`/`vgprField`). **2×2 oracle PASS bit-exact** (256³ 256/256, 512²×K2048 1024/1024).

| @ 4096²×K16384 | TF | %307 | WMMA/cyc | maxlive | acc00 |
|---|---:|---:|---:|---:|---|
| 2×2 BLADDER **fed** | 0.9 | 0.3% | 0.05 | 256 WGs | 64/64 ✓ |
| **NOFEED @ 2×2** | **32.8** | 10.7% | 1.70 | 256 WGs | (garbage) |
| **NOFEED @ 4×4** (control) | **97.9** | 31.9% | 5.07 | 192 WGs | (garbage) |

**Decisive result — the occupancy hypothesis is FALSIFIED.** 2×2 *raised* occupancy (256 resident WGs
vs 192) yet the NOFEED compute ceiling *fell* 3× (98 → 33 TF). More waves did not lift the ceiling — it
dropped. The ceiling tracks **WMMA run length / issue density**: 4×4 runs 32 back-to-back WMMA per
K-tile (5.07 WMMA/cyc); 2×2 runs only 8 before per-tile loop+claim overhead (1.70). (2×2 also pays 4×
the atomic-claim/broadcast/barrier overhead — 4096 vs 1024 tiles for the same M·N — which compounds the
same direction.) **Every change that raised occupancy lowered throughput; the only thing that raised
the ceiling was the *larger* tile (longer WMMA runs).**

**Re-orientation (two walls, both pointing the same way):**
1. **WMMA-issue ceiling** (98 TF @ 4×4, < G2 161) is *issue-density bound*, not occupancy bound. HIP's
   ~272 NOFEED at the same 4×4 ⇒ hipcc keeps the WMMA stream dense **across** K-tiles (deep K-unroll,
   no per-K-tile branch/claim breaking the run). Lever = **deeper K-unroll + cheaper tile boundaries**,
   i.e. *bigger* uninterrupted WMMA windows — the opposite of shrinking the tile.
2. **Feed latency** (the per-K-tile A round-trip) still crushes the fed path to ~1 TF, and occupancy
   cannot hide it (2×2 made it worse). HIP hides it by **software-pipelining the feed across the
   unrolled K-window** so tile t+1's loads overlap tile t's long WMMA run.

Both walls are solved by the *same* structural change HIP uses: **software-pipelined multi-K-tile
unroll** — chain N K-tiles of WMMA back-to-back (raises the ceiling) while prefetching the next
window's A/B during the run (hides feed). That, not occupancy, is the next lever.

**Codegen gotcha logged:** the `FM`-guard refactor briefly moved `ds_store v12,v[16:19]` *before* the
`s_wait_loadcnt 0x0`, creating a RAW hazard on the A-fill VGPRs (LDS got pre-load garbage). 4×4 oracle
dropped to ~half-bad until all A-fill loads were waited *before* any `ds_store`. Wait the whole A-fill
group, then store.

> **⚠️ FRAMING UPDATE (read §2.4c/d first).** The root cause was found downstream of these probes: the
> **global atomic tile-claim** (the dead-TGID workaround). Demote the intermediate conclusions:
> - the **~100 TF "ceiling" was NOT a WMMA-issue ceiling** (the stream is HIP-identical; coarse claiming
>   lifts it to 201.6);
> - the **192-WG "wall" was a *symptom*** of the 4-wave/barrier shape + atomic contention, not a primary
>   limit (1-wave WGs admit 896);
> - **occupancy "falsification" (§2.3)** holds only as "shrinking the tile doesn't help" — the real
>   occupancy story is the atomic contention that *rises* with resident waves.
> Current root cause: **global atomic-claim caused by dead-TGID workaround** (§2.4c/d).

## Phase 3 Step 2.4 — why NOFEED@4×4 caps at ~99: it is the vehicle's 192-WG residency wall

Three probes to localize the ~98 TF NOFEED ceiling (`KUNROLL` defsym = N copies of the 32-WMMA block
before the backedge; harness `--nofeed-unroll`, `--nofeed-occ`):

**(a) NOFEED@KUNROLL sweep (U=1,2,4,8) — FLAT.** 102.7 / 104.7 / 103.2 / 100.5 TF (5.3 WMMA/cyc). 256
back-to-back WMMA (U=8, backedge eliminated) is no faster than 32. ⇒ **not backedge / run-length bound.**

**(b) WMMA-ordering diff vs the HIP winner (`/tmp/cg/winner_hotloop_4x4.s`) — IDENTICAL.** Both are
mi-outer / ni-inner, same A-frag reused for 4 consecutive WMMA, **16 distinct accumulators** per kk
group, kk-reuse at distance 16. My disasm: 32 fully back-to-back WMMA, **zero** assembler-inserted
waits/nops/delays. ⇒ **the issue stream is *not* malformed** (rules out GPT's branch-2 premise).

**(c) NOFEED@4×4 occupancy sweep at the SAME tile — residency is HARD-CAPPED.**

| field / LDS / grid | TF | WMMA/cyc | maxlive |
|---|---:|---:|---:|
| 26 / 8196 / 256 (base) | 93.6 | 4.85 | **192** |
| 24 / 4100 / 256 | 99.3 | 5.14 | **192** |
| 24 / 4100 / 512 | 86.0 | 4.46 | **192** |
| 24 / 4100 / 1024 | 63.5 | 3.29 | **192** |

`maxlive` is pinned at **192 WGs** (= 3 WGs/CU × 64 CU = 768 waves = **6 waves/SIMD**) — *unchanged* by
tighter VGPR (192 vs 208), smaller LDS (4100 vs 8196), or a 4× bigger grid. Over-launching only adds
atomic-claim contention (throughput falls 99 → 64). So the ~99 TF ceiling is the **192-WG residency
wall**, not registers/LDS/scheduling.

**Provenance (resolved):** the HIP 4×4 **lever-kernel** NOFEED was recorded at **284–289 TF** — not just
the standalone `wmma_peak` microbench. So the contradiction is real: HIP 4×4 NOFEED 284–289 vs PM4
wave-group NOFEED ~99–105, and a feed pipeline cannot beat the PM4 NOFEED ceiling. Resolve the ceiling
first (below).

## Phase 3 Step 2.4c/d — ROOT CAUSE: the global atomic-claim (dead-TGID workaround) is the wall

Barrier-free NOFEED residency probe (`occ_kernel_wgnofeed_bf.s`, harness `--nofeed-bf`): same 4×4 WMMA
stream, **no LDS / no `s_barrier_*`**, per-wave independent atomic-claim, operands loaded once into
registers. `CLAIMCHUNK` defsym = work-units grabbed per atomic. Launch 128-thread (4-wave) or 32-thread
(1-wave). claim_ceil = M·N/4096 ⇒ total WMMA = claims·K = M·N·K/4096 (TF stays calibrated).

| config @ 4096²×K16384 | TF | WMMA/cyc | maxlive | reading |
|---|---:|---:|---:|---|
| original 4-wave (barriers+LDS, atomic/tile) | ~99 | 5.1 | 192 WG | the prior "ceiling" |
| barrier-free 4-wave (atomic/tile) | ~135 | 7.0 | 192 WG | barriers+LDS cost ~26% |
| barrier-free **1-wave** c1 nWG=256 | 123–158 | 6.4–8.2 | 256 WG | per-tile atomic |
| barrier-free **1-wave** c1 nWG=896 | 86 | 4.5 | **896 WG** | residency is wave-level (~7/SIMD), **and TF FALLS as waves rise** |
| barrier-free **1-wave c16** nWG=256 | **201.6** | **10.4** | 256 WG | **16× fewer atomics ⇒ +64% — clears G2 161** |
| barrier-free 4-wave c16 nWG=256 | 167 | 8.6 | 196 WG | 4-wave structure costs ~35 TF vs 1-wave |

**Two confirmations that the global atomic-claim is the dominant wall:**
1. **Throughput falls monotonically as resident waves rise** (1-wave c1: 256w→158, 384w→135, 512w→119,
   768w→96, 896w→87) with near-identical claim counts — the signature of contention on the single
   `global_atomic_add` (TH_ATOMIC_RETURN) claim counter scaling with concurrency.
2. **Coarsening the claim 16×** (same total WMMA, 16× fewer atomics) **lifts NOFEED 123 → 201.6 TF** at
   the same occupancy.

Also learned: the residency cap is **wave-level (~768–896 waves ≈ 7/SIMD)**, *not* a 192-WG limit
(1-wave WGs admit 896; 4-wave admit 192 = 768 waves). Run-to-run variance on this vehicle is ~±20%.

**Why this reframes everything:** the atomic claim was adopted in Phase 1 *solely because raw-PM4 TGID
is dead* (no per-WG id). It serializes under contention, and that — not feed latency, not occupancy,
not WMMA scheduling — is the primary gap to HIP's 285 (HIP assigns tiles by workgroup-id, zero atomics).
A 1-wave kernel with coarse chunking already NOFEEDs **201.6 TF, above the 161 bar.**

**Next:** kill / minimize the atomic claim. Options: (a) coarse static chunking (partial — done, +64%);
(b) recover a real per-WG id without a global atomic (revisit the dead-TGID problem — a PM4 path that
delivers WG id, or seeding per-WG via USER_SGPR per-dispatch); (c) DeepSeek's non-persistent dispatch
(one claim per WG lifetime). Then re-introduce feed (software-pipelined) on the winning low-contention
shape and re-measure FED toward 161.

## Phase 3 Step 2.5 — band-claim on the real GEMM: NEGATIVE (atomic was a probe artifact)

`occ_kernel_wggemm2.s` defsym `BAND` (one `atomic_add(counter, BAND)` + one broadcast-barrier per BAND
tiles, then stride the band with zero atomics). **Oracle PASS bit-exact** at BAND=16 (each tile computed
once; bands disjoint). Harness `--band-sweep` @ 4096²×K16384 (TOTAL=1024 tiles):

| | band=1 | band=2 | band=4 | band=8 | band=16 | band=32 |
|---|---:|---:|---:|---:|---:|---:|
| NOFEED TF | 107 | 96 | 101 | 82 | 75 | 46 |
| FED (BLADDER) TF | 1.4 | — | 1.4 | — | 2.0 | — |

**Negative result — the atomic is NOT the real-GEMM wall.** Cutting atomic frequency 4× (band=1→4) left
NOFEED flat (107→101); bigger bands *hurt* (only 1024 tiles ⇒ band≥8 → fewer bands than the 192 resident
WGs → **starvation**). The probe's c16 = 201.6 win was real but **isolated to its stripped regime**:
operands loaded *once* (no per-tile A-fill), 1-wave (no barriers). The real GEMM reloads A and barriers
*per tile* (each output tile needs its own operands; the 4-wave A-share needs sync), so that per-tile feed
structure — not the atomic — is the wall.

### Corrected root cause (Phase-1 decomposition + HIP reference compare)

| stage | TF | the cost between stages |
|---|---:|---|
| bare-compute probe (load once, 1-wave) | ~158–201 | — |
| **NOFEED** real (A-fill once/output-tile + barriers) | ~107 | per-output-tile A-fill + barriers + 4-wave structure (~100 TF) |
| **FED** real (A-fill+B-load+barriers per K-tile ×512) | ~1.4 | **the per-K-tile *serialized* feed (≈76×) — THE dominant wall** |

Comparing to the HIP winner's hot loop (`/tmp/cg/winner_hotloop_4x4.s`): HIP **software-pipelines** —
loads K-tile *t+1* interleaved into K-tile *t*'s WMMA run, so feed latency overlaps compute. Mine
**serializes**: `load(t) → barrier → compute(t) → barrier → load(t+1)`. This — not the atomic, not
occupancy, not WMMA scheduling, not run length — is why FED is 1.4 vs HIP 161.

**Levers that did NOT move real FED off ~1–2 TF: BLADDER fine ladder, 2×2, KUNROLL, band-claim.** Per
systematic-debugging (4 failed point-fixes → question architecture): the wall is the **serialized
per-K-tile feed loop**.

## Phase 3 Step 2.6 — FEED-ONLY probe: compute is FREE, the feed is 100% of the wall

`occ_kernel_wggemm2.s` defsym `FEEDONLY` (BLADDER path: keep the entire per-K-tile feed — A-fill,
`ds_store`, barriers, B `global_load_tr`, A-frag `ds_load`, fine `s_wait_loadcnt` ladder, barrier — but
emit **ZERO WMMA**). @ 4096²×K16384:

| variant | wall (as 2·M·N·K/wall "TF") | acc00 |
|---|---:|---|
| FED = feed + 32 WMMA/K-tile | 1.4 | 64/64 ✓ |
| **FEED-ONLY = same feed, 0 WMMA** | **1.4** | 0/64 (no compute) |

**Removing every WMMA changed the wall by zero.** Compute is fully hidden; the **feed is 100% of the
wall**. Every compute-side lever (ladder, 2×2, KUNROLL, WMMA order, band) optimized something off the
critical path — hence uniformly ~1 TF.

**Magnitude:** the feed moves ≈4.3 GB (8 KB/K-tile · 512 · 1024) in ≈1.57 s ⇒ **≈2.7 GB/s, <0.5% of the
card's ~640 GB/s.** Not bandwidth-bound — the loads are running **nearly fully serialized** (one
round-trip at a time), because every K-tile drains `s_wait_loadcnt/dscnt 0x0` + a workgroup barrier
before the next can start, so the 192 resident WGs barely overlap.

**The remaining decider:** is this *recoverable latency* (deep multi-K-tile async → memory runs at
bandwidth → FED jumps) or a *hard serialization* of this persistent 4-wave/LDS-staged/barrier vehicle?
Probe = a feed-only kernel that issues *many* K-tiles' A/B loads back-to-back with a *single* trailing
wait (no per-K-tile drain). If its effective bytes/s jumps toward bandwidth, deep prefetch is the fix; if
it stays ~2.7 GB/s, the vehicle cannot keep loads in flight and the wave-group/PM4 structure must change.

## Phase 3 Step A — FEED-ONLY depth-P pipeline probe: depth does NOT help; the load stream is NOT the wall

GPT's "multi-stage feed pipeline" angle: keep `P` independent slices in flight against `loadcnt` so the
queue never drains. `occ_kernel_feedpipe.s` — **no WMMA / no LDS / no barrier**, 1-wave (32-thread) WGs,
a depth-`P` register ring of `FRAGS` `global_load_b64`/slice, `s_wait_loadcnt((P-1)·FRAGS)` back-pressure,
xor-consume to force retire. Streams a 64 MiB buffer, `nWG=1024`. `--feedpipe` @ 393 216 slices:

| depth | wait watermark | eff. feed BW | ×(real 2.7 GB/s) | maxlive |
|---|---|---:|---:|---:|
| **P=1** F=8 (drain every slice) | `loadcnt 0` | **123.3 GB/s** | **45.7×** | 1024 |
| P=2 F=8 | `loadcnt 8` | 119.3 GB/s | 44.2× | 1024 |
| P=4 F=8 | `loadcnt 24` | 90.8 GB/s | 33.6× | 1024 |
| P=8 F=8 | `loadcnt 56` | 74.1 GB/s | 27.4× | 1024 |
| P=16 F=4 | `loadcnt 60` | 48.3 GB/s | 17.9× | 1024 |

**Two results, both decisive:**

1. **The decider is answered — and it's neither branch.** A barrier-free / LDS-free load stream already
   runs **123 GB/s even at P=1** — i.e. *draining every single slice* (`s_wait_loadcnt 0x0` per slice,
   the exact per-K-tile drain we blamed) still delivers **45× the real GEMM's feed.** So the 2.7 GB/s is
   **NOT** a property of the load stream and **NOT** caused by per-slice draining. The loads are not the
   wall. Latency is already hidden by **wave-level** parallelism (`maxlive=1024` — tiny 1-wave WGs take
   full residency, ≥1024 resident waves).

2. **Deeper pipelining HURTS — monotonically (123→48 GB/s as P 1→16).** With 1024 waves already
   concurrent, adding per-wave depth oversubscribes the memory system (more outstanding requests → MSHR
   thrash), it does not hide more latency. **GPT's hypothesis ("keep P slices in flight → BW rises toward
   bandwidth") is falsified.** "Build multiple feed pipelines / deep prefetch" is NOT the fix for FED.
   (VGPR/occupancy is *not* the cause of the falloff — `maxlive=1024` held at every P, incl. 160-VGPR P=16.)

   *L2 caveat:* P=1's instantaneous working set (~1024·1·2 KB ≈ 2 MB) may be partly L2-resident, inflating
   123 toward L2 BW; but even P=16 (≈32 MB working set, DRAM-dominated) is 48 GB/s = **18×** the real feed.
   Robust either way.

**Where this points the investigation.** The 45× gap between this probe (P=1) and the real GEMM feed is
exactly what the probe *removed*: the **per-K-tile workgroup barrier**, the **LDS A round-trip**
(`ds_store → barrier → ds_load`), and the **4-wave lock-step coupling**. The real FED wall is one (or
more) of *those*, not load serialization. **Next probe is not deeper pipelines — it is to add the removed
couplings back onto the 123 GB/s baseline one at a time** (4-wave WG + per-slice `s_barrier`; then the LDS
A round-trip) and watch which one collapses 123 → ~2.7. That localizes the true wall. *(STOP — GPT review.)*

### Step A ladder — couplings added back one at a time: NONE collapses the feed

`occ_kernel_feedladder.s` (GPT rungs 1-5). Same load body, toggles `WAVES`/`BARRIER`/`LDSMODE`. Fixed
NSLICES=4096/wave (no atomic-claim → all waves of a WG hit equal barrier counts), **1024 launched waves
held constant** across rungs so GB/s is directly comparable. `--feedladder`:

| rung | coupling added | eff. BW | ×(2.7) | maxlive |
|---|---|---:|---:|---:|
| r1 | 1-wave, none (rebaseline) | 1073 GB/s | 398× | 1024 |
| r2 | 4-wave WG | 1207 GB/s | 447× | 1024 |
| r3 | + `s_barrier_signal/wait` per slice | 1116 GB/s | 414× | 1024 |
| r4 | + LDS round-trip (`ds_store→barrier→ds_load`, no global A) | 2382 GB/s | 882× | 1024 |
| r5 | + global A → LDS → barrier → ds_load (real A-share path) | 1088 GB/s | 403× | 1024 |

**SECOND FALSIFICATION — none of the hypothesized couplings is the wall.** Per-slice barrier costs ~7%
(r2→r3); the LDS publication round-trip costs **nothing** (r4/r5 ≥ r1). 4-wave shape, barrier cadence, and
the full real A-share path all run **400-880× the real GEMM's feed.** The barrier/LDS/4-wave publication
machinery is *not* what pins FED at 2.7 GB/s.

**Two facts reframe the whole problem:**
- **The real feed is stall-bound, not bandwidth-bound.** It moves only 4.3 GB in 1.57 s — **~200× below
  even cold-DRAM bandwidth (~640 GB/s).** So the wall is *loads stalling*, not bytes saturating anything.
  (The ladder's >640 GB/s absolutes confirm the *probe* is L2-resident — it re-reads a ~64 MB footprint
  ~1000×. But cold-DRAM would still be ~640, not 2.7, so L2 is not what hides the real wall.)
- **Everything the probe and FEEDONLY share runs fast; only the real feed stalls.** The remaining untested
  differences between the fast ladder and the slow real feed (§2.6, 2.7 GB/s) are: **(a) B's
  `global_load_tr_b64` transpose load** (never tested — GPT's deferred rung 6, now the prime suspect);
  **(b) the real cold-streaming access pattern / strides** (vs the probe's cache-friendly re-reads); and
  **(c) the real GEMM's lower residency** (~768 waves, capped by its 160-192 VGPR + 8 KB LDS, vs the
  ladder's 1024).

**Next stage (for GPT):** rung 6 = add B `global_load_tr_b64` with real B addressing; plus a *residency
control* (re-run r5 forced to ~768 waves via high VGPR/LDS) and a *cold-stream control* (no L2 reuse) to
separate (a)/(b)/(c). If rung 6 collapses → the transpose load is the wall. *(STOP — GPT review.)*

### Step A rung 6 — B `global_load_tr_b64` is NOT the wall either (THIRD FALSIFICATION)

`occ_kernel_btr.s` (GPT rungs 6a-6d). Replicates the real B recipe exactly: `s9=NT*256` (=65536), 2 kk-groups
of FN=4 frags (kk1 @ `+s9`), `offset:ni*256`, vaddr=`lane*8`, per-slice advance `2·s9`, wave seed
`(wser*2048 + wave_n*1024)`. 1024 launched waves, NSLICES=2048. `--feedbtr`:

| rung | what | eff. BW | ×(2.7) | maxlive |
|---|---|---:|---:|---:|
| 6a | `global_load_tr_b64`, synthetic cache-friendly stride | 274 GB/s | 102× | 1024 |
| 6b | `global_load_tr_b64`, **real Bshuf addressing** | 171 GB/s | 63× | 1024 |
| 6c | 6b + **real residency** (192 VGPR + 8 KB LDS) | 137 GB/s | 51× | 768 |
| 6d | **plain `global_load_b64`**, same Bshuf addressing (neg. control) | 155 GB/s | 57× | 1024 |

**The transpose load is innocent.** 6b (transpose, real addressing) = 63× the real feed and does not
collapse; 6d (plain load, *identical* addresses) ≈ 6b (within noise) → transpose-vs-plain makes no
difference, so `global_load_tr_b64` is not pathological. Real Bshuf addressing costs ~1.6× (6a→6b); the
real 768-wave residency costs ~20% (6b→6c). **None collapses toward 2.7.**
*(Probe bug caught & fixed: the first run had 6b/6c/6d hang because real-Bshuf reads ran ~66 KB past the
64 MiB wrap → OOB fault. Stream buffer extended to 96 MiB; the "timeout" was an artifact, not a result.)*

**Verdict after rungs 1-6 (per GPT's interpretation matrix: "all fast"):** every isolated component —
plain loads, depth pipelining, 4-wave shape, per-slice barrier, LDS A round-trip, B transpose load, real
Bshuf addressing, real residency — runs **50-880× the real GEMM's feed.** The wall is **not any single
operation.** It is in **how the real per-K-tile loop *sequences* them** — a hidden loop-carried dependency
or an over-wait in the real kernel's `s_wait_loadcnt`/`s_wait_dscnt` accounting. Component-bisection from
below is exhausted.

**Next: bisect from the real kernel (top-down).** Take the real FEEDONLY path (2.7 GB/s, §2.6) and neuter
ONE thing at a time until it jumps: (1) replace the per-K-tile B-address progression (`s20 += 2·NT·256`)
with a fixed address (re-read) — tests whether the loop-carried scalar address dep + cold stride is it;
(2) collapse the fine `s_wait_loadcnt` ladder to a single trailing wait — tests over-wait; (3) drop the
per-K-tile A-frag `ds_load`; (4) run it single-wave (no barrier/claim). The component that makes 2.7 jump
is the wall. Alternatively, add in-kernel per-phase realtime timers to the real K-loop (the `mbgemm`
PROFILE mechanism) to read the stall directly. *(STOP — GPT review.)*

### Step A rung 7 — phase timers: an accidental 70× SPEED-UP fingers SYNCHRONIZED LOCKSTEP, not the feed

GPT: timers first. Added sampled realtime phase timers (`PROFILE` defsym, all under `.if PROFILE`,
non-PROFILE byte-identical ✓) to the **real DBUF==1 A-ping-pong K-loop** (the canonical 1.4-TF FED — note
`DBUF` defaults to 1; the perf bin = this path). One global profiler wave accumulates per-phase realtime
tick-sums → `occ[8..15]`. `--feedprof`.

**The dominant result is an anomaly, not the breakdown.** The PROFILE build runs at **96 TF (~22 ms)**,
while the **byte-identical non-PROFILE build is 1.4 TF (~1.57 s)** — adding one slow profiler wave made the
**whole kernel 70× faster** (reproduced: the earlier BLADDER-FEEDONLY profile build also jumped to ~98 TF).
The profiler also counted only 7 K-tiles (inconsistent with barrier-locked execution), so the *absolute*
phase numbers are not trustworthy. Tentative ratios (caveated): parked on **B-load wait 44%**,
**A-prefetch land + ds_store 35%**, **tail barrier 21%**; WMMA **0.1%** (compute hidden — consistent with §2.6).

**What the 70× speed-up most likely means.** Perturbing the *timing* of one wave (extra `s_sendmsg`
instructions + descheduling) **desynchronizes the 4-wave-WG barrier lockstep**. Strong hypothesis: in the
real kernel all ~768 resident waves hit their per-K-tile barriers *in phase*, so their memory stalls are
**correlated** — every wave waits for the feed at the same moment and nothing hides anything (1.4 TF).
Jittering one wave breaks the phase alignment → stalls interleave → latency hides → 96 TF. **The wall is
SYNCHRONIZED LOCKSTEP / correlated stalls, not the feed itself** — fully consistent with rungs 1-6 (every
isolated feed component was fast) and with the prior `mbgemm` lockstep-stagger lead (`STAGGER` defsym;
"feed stalls fire in a synchronized burst; phase-offset so they interleave").

**Next (for GPT):** test the desync hypothesis *directly on the real kernel, no profiler*: add a small
per-wave **stagger** (phase-offset each wave's K-loop start by a few hundred cycles, keyed off wave/lane id)
and see if 1.4 → ~96 TF. If it jumps, the wall is named (correlated lockstep stalls) and the fix is
de-correlation (stagger / fewer barriers / 1-wave-per-tile). The timer path is shelved — it perturbs the
very thing it measures. *(STOP — GPT review.)*

### Step A rung 8 — inert per-WG stagger: the lockstep hypothesis is FALSIFIED (the PROFILE 70× was a side effect)

GPT (correctly cautious): a 70× speed-up from `PROFILE=1` is too large to trust as "just measurement" —
test desync directly, but with an **inert** stagger, not profiler machinery. Recommended: non-PROFILE,
deterministic, **per-WG-varied** (a per-wave stagger inside a 4-wave WG is eaten by the next WG barrier;
what you want is *inter-WG* phase decorrelation), and swept.

Built (`STAGGER` defsym, all under `.if STAGGER`; **STAGGER=0 byte-identical to the perf bin** ✓, gate in
build.sh). At the DBUF==1 K-loop entry (after the t=0 publish barrier, before `.Lkt_loop`), once per claimed
tile: `delay = ((ti*13 + wid*3) & MASK) << SHIFT` busy-loop iters. `ti` (s17) is per-WG (same for all 4
waves, differs across WGs) → a *persistent* inter-WG offset (no cross-WG barrier re-syncs it). No timers, no
atomics, no `s_sendmsg`. Same dispatch/occupancy as `--wggemm-perf`. `--feedstag` runs baseline + the sweep
at 4096²×K16384.

**Result — zero movement across the entire sweep:**

| cell | delay (iters) | TF | maxlive | claims | acc00 |
|---|---|---|---|---|---|
| baseline (no stagger) | — | 1.4 | 192 | 1280 | 64/64 |
| MASK=0  SHIFT=5 (control, delay≡0) | 0 | 1.4 | 192 | 1280 | 64/64 |
| MASK=3  SHIFT=5 | ≤96 | 1.4 | 192 | 1280 | 64/64 |
| MASK=7  SHIFT=5 | ≤224 | 1.4 | 192 | 1280 | 64/64 |
| MASK=15 SHIFT=5 | ≤480 | 1.4 | 192 | 1280 | 64/64 |
| MASK=31 SHIFT=5 | ≤992 | 1.4 | 192 | 1280 | 64/64 |
| MASK=15 SHIFT=4 | ≤240 | 1.4 | 192 | 1280 | 64/64 |
| MASK=15 SHIFT=6 | ≤960 (~2000 cyc) | 1.5 | 192 | 1280 | 64/64 |

Per GPT's pre-registered interpretation, "**no movement** ⇒ the PROFILE speed-up was a side effect, go back
to top-down neuters." The MASK=0 control reproduces baseline exactly (the stagger *code* costs nothing), and
even a ~2000-cycle per-tile offset moves nothing. **The synchronized-lockstep / correlated-stalls hypothesis
is NOT supported.** The 70× from rung 7 was a side effect of the *profiling path* (the extra `global_atomic`
token grab, the `s_sendmsg_rtn`/`s_wait_kmcnt`, or the control-flow), not desync.

**Two caveats this test does NOT close (raise to GPT):** (1) the stagger fired **once per claimed tile** =
once per **512 K-tiles**, whereas the profiler's `s_sendmsg` fired **per K-tile** — 512× more frequent. So
this is not a faithful reproduction of the profiler's *cadence*; a per-K-tile stagger inside the loop would
be the matched test. (2) If the coupling is memory-contention-mediated, a sparse one-time offset can't
*sustain* decorrelation (the contention re-syncs the WGs each K-tile). Both point at the same stronger test.

**The real lead is now the 70× itself.** A tiny code change made the real kernel 70× faster — that is the
prize, whatever the mechanism. **Next (for GPT): bisect the PROFILE path.** Add each PROFILE sub-component
*independently* to the real non-timed kernel and find which one triggers the jump: (a) the per-tile
`global_atomic_add` profiler-token grab alone; (b) the `s_sendmsg_rtn_b64 + s_wait_kmcnt` per-K-tile alone
(this is also the per-K-tile-cadence stagger GPT's caveat #1 asks for); (c) the `s_cmp`/branch control-flow
alone. Whichever reproduces 96 TF *is* the lever. Fallback per GPT: top-down neuters (#2 = collapse the fine
ladder to one trailing wait). *(STOP — GPT review.)*

### Step A rung 9 — bisect the PROFILE 70×: it was a MEASUREMENT ARTIFACT (the PROFILE build computes garbage)

GPT: the 70× is the strongest clue in the campaign and now a finite bisection — add each PROFILE ingredient
*independently* to the real DBUF==1 path (byte-close, non-PROFILE) and find which reproduces 96 TF. Order:
#1 per-K `s_sendmsg_rtn`+`s_wait_kmcnt` (all-wave) → #2 per-tile leader token atomic → #3 per-K cmp/branch
skeleton → #4 per-K inert busy-loop (control). Built `PB` defsym (all under `.if PB`; **PB=0 byte-identical
to the perf bin** ✓). `--feedpb`.

**None of the four ingredients moved it** — every variant is honest at 1.3–1.4 TF with **correct output**:

| variant | cadence | TF | maxlive | claims | acc00 |
|---|---|---|---|---|---|
| baseline | — | 1.4 | 192 | 1280 | 64/64 |
| PB1 sendmsg+kmcnt (all-wave) | per-K | 1.4 | 192 | 1280 | 64/64 |
| PB2 leader token atomic | per-tile | 1.4 | 192 | 1280 | 64/64 |
| PB3 cmp/branch skeleton | per-K | 1.4 | 192 | 1280 | 64/64 |
| PB4 inert busy-loop (control) | per-K | 1.3 | 192 | 1280 | 64/64 |

The decisive tell: every PB variant **completes the GEMM correctly** (acc00 64/64) and is honest at 1.4 TF.
That reframed the question to: *was the 96 TF ever real?* The rung-7 `run_feedprof` had **no correctness
check** and reported an impossible **K-tiles=7**. So I added the same sampled acc[0][0] oracle to
`run_feedprof` and re-ran the exact rung-7 PROFILE dispatch:

```
profiler K-tiles=8  maxlive=192  whole-wall TF=94.96
*** acc00 OK=0/64  <<< OUTPUT WRONG ***
```

**The PROFILE build computes complete garbage (0/64).** The 95 TF is a **bogus wall**: the profiling
machinery corrupts the kernel (every wave's output is wrong, not just the profiler wave's), it races to
completion in ~8 "K-tiles" of real work, and `TF = 2·M·N·K·freq/wall` credits the *full* M·N·K against that
tiny wall → a fake 70×. **The rung-7 "synchronized lockstep" headline, and its phase breakdown, were built
on a measurement artifact.** Rung 8 (per-tile desync, no movement) and rung 9 (no ingredient reproduces it,
all correct) are both consistent with this: there was never a real speedup to reproduce.

**Two hard conclusions:**
1. **The honest, correct FED throughput is 1.4 TF / 2.7 GB/s** — confirmed by every correct-output variant
   (baseline, rung-8 sweep, rung-9 PB1–4). The 96 TF is retired.
2. **In-kernel realtime phase timers (the `PROFILE`/`s_sendmsg_rtn` mechanism) cannot be trusted on this
   kernel — they don't just perturb it, they *break* it** (0/64). The whole timer path is shelved for good.

**Methodology guard (the trap rung 7 fell into):** any "it got faster" must be **correctness-gated**, or for
intentionally-garbage probes (FEEDONLY, dropped-wait neuters) the memory traffic must be *proven* to have
happened (a final dependency on all loaded data, or an achieved-bandwidth counter) — otherwise "faster" can
just mean "skipped the feed," and `TF` will lie.

**Next: back to the pre-rung-7 plan — top-down neuters on the CORRECT kernel.** Localize the 1.4-TF wall in
how the real per-K-tile loop *sequences* the (individually-fast) feed: (1) fixed B re-read address vs the
per-K `s20 += 2·NT·256` progression (loop-carried scalar addr dep + cold stride); (2) coarsen/over-wait
analysis on the DBUF==1 `s_wait_loadcnt 0x0`; (3) drop the per-K A-frag `ds_load`; (4) single-wave real-loop
clone. Each measured with the acc00 gate (or a bandwidth counter for garbage probes). *(STOP — GPT review.)*

## STACK LADDER (post rung-9): rebuild the fast feed from a known-good core, +1 obligation/rung

After the rung-7 PROFILE 70x was proven a measurement artifact (rung 9), GPT + user reset the campaign: stop
single-variable neuters; **stack from a minimal truthful core, adding exactly ONE real-GEMM obligation per
rung**, reporting the three figures at every rung -- **TF (or TF-eq) / GB/s / correctness-or-traffic proof**
(+ maxlive, claims). The first rung that collapses from "fast" toward the real FED's **2.7 GB/s / 1.4 TF**
names the wall. Planned: 1 load-only base -> 2 real K-loop address progression -> 3 4-wave WG -> 4 LDS A
publication -> 5 B global_load_tr -> 6 WMMA -> 7 store. Hard rule (the rung-7 trap): never credit a fast
number unless the third figure proves the work happened.

### Stack rung 1 -- load-only truthful base (`occ_kernel_stack.s` RUNG=1, `--stack`)

1-wave (32-thread) WGs stream FRAGS=8 b64 from **both** real A and B buffers per claimed slice (P=1 full
drain), consume every loaded dword into a per-wave sum, and at exit emit TWO proofs: a **slice count**
(occ[6], leader-lane) and a **data checksum** (occ[7], all lanes). The harness credits GB/s only if
**occ[6]==CLAIM_CEIL AND occ[7]==CPU-recompute AND checksum!=0**.

| variant | TF-eq | GB/s | proof | maxlive | claims |
|---|---|---|---|---|---|
| 1 load-only A+B (P=1) | 12.9 | **24.9** | PASS cnt=1048576 chk=CPU | 1024 | 1310720 |

The guardrail **works**: a first run had a userdata-index bug (`CLAIM_CEIL` landed at s14, kernel reads s13
-> CEIL=0 -> every wave exited immediately, streaming nothing) and it surfaced as **checksum=0 -> FAIL, GB/s
NOT credited** -- exactly the rung-7 trap, defused. (claims over-counts CEIL because each WG does one atomic
before the `s17>=CEIL` exit check; the count proof uses the separate consumed-slice counter occ[6].)

**Base calibration (24.9 vs feedpipe 122).** Same machine/maxlive/wave-shape, the old feedpipe P=1 = **122
GB/s on a 64 MiB stream** (cache-resident on R9700's Infinity Cache); rung 1 = 24.9 GB/s on a **128 MiB**
working set (A 64 + B 64 MiB -> exceeds cache -> HBM-bound). The 5x gap is **working-set/cache, not kernel
overhead**. 24.9 is the *working-set-matched* base (the real GEMM touches the full 128 MiB A+B footprint), so
the honest collapse window is **24.9 -> 2.7 GB/s (~9x)**. *(Open for GPT: accept 24.9 as the HBM-matched base
and proceed to rung 2, or first sweep rung-1 working set 32/64/128 MiB to pin cache-vs-HBM and fix the
canonical base.) STOP -- GPT review.*

### Stack rung-1 BANDWIDTH HUNT -- "where is the 640 GB/s?" (24.9 is NOT memory bandwidth)

24.9 GB/s = 3.9% of the R9700's ~640 GB/s. Systematic falsification (every row traffic-proofed: cnt==CEIL,
chk==CPU):

| lever varied | result | verdict |
|---|---|---|
| residency: nWG 512/1024/2048/4096/8192 | GB/s 25.8/24.4/23.4/21.1/17.4; maxlive caps at **2048** | NOT occupancy (flat, even falls) |
| stride: 8K (25% dense) vs 2K (contig) | 24.8 vs 24.0 | NOT access-pattern |
| working set: 16MiB (cache) vs 256MiB (HBM) | 24.x vs 23.x (identical) | **NOT memory BW** (cache would win) |
| consume: 32-deep serial sum vs 4 parallel chains | 24 vs 24 | NOT the checksum overhead |

The cache==HBM result is the decisive one: if memory were the bottleneck, the 16 MiB cache-resident window
would be far faster than the 256 MiB HBM window. It isn't. **So the ~24 GB/s is a kernel/issue ceiling, not
a memory-bandwidth measurement.**

**The smoking gun is feedpipe-vs-stack, both cache-resident:** feedpipe P=1 (8 b64 loads/slice, ONE buffer)
= **122 GB/s**; this kernel (16 loads/slice, A+B two buffers) = **24 GB/s**. Doubling the loads *quartered*
the bandwidth -- anti-proportional, so a per-slice load-issue/drain serialization that worsens super-linearly
with load count and/or the dual-stream (A at s[2:3], B at s[4:5]) feed. Arithmetic: at 24 GB/s the effective
per-slice time is ~350 us for 16 loads even from cache (~13x any plausible cache latency) -> all ~2048 waves
are queueing behind one shared ~94M-coalesced-loads/s resource, identical for cache and HBM (so it sits at or
above the L2/coalescer, not DRAM).

**Conclusion: the missing ~615 GB/s is NOT in the memory system** -- it's behind a per-wave load-issue/drain
serialization in the P=1 persistent-claim structure, amplified by the two-stream feed and capped residency
(maxlive=2048 = half of Navi48's ~4096 wave slots; cf the old 768-wave / atomic-claim walls #294/#295). What
I can't crack black-box (raw PM4 exposes no memory perf counters): the exact mechanism of the anti-proportional
load-count penalty.

**Next probes (for GPT/user):** (1) A-only vs A+B (isolate the dual-stream penalty -- does single-stream hit
~122 like feedpipe?); (2) b128 wide loads (fewer, wider transactions -> tests a coalescer/transaction-rate
cap); (3) software-pipelined depth at fixed high occupancy; (4) **a NON-persistent grid-stride dispatch** --
the persistent atomic-claim raw-PM4 structure itself (2048-wave cap) may be the BW limiter, independent of the
GEMM. *(STOP -- GPT review.)*

## ROOT CAUSE FOUND: operands were in SYSTEM/GTT memory (PCIe), not VRAM -- the whole campaign fed over PCIe

GPT/user set a strict gate: prove the raw-PM4 vehicle moves data NEAR SPEC (~640 GB/s) BEFORE any more GEMM.
Built a CLEAN streaming bandwidth probe (`occ_kernel_bw.s`, `--bw`): ONE atomic/wave for a dense worker id
(no TGID), then a pure streaming hot loop -- wide coalesced loads (b32/b64/b128), UNROLL-deep MLP, NO
atomics/LDS/barriers/per-element drain. read(checksum) / copy(load+store) / write(fill). Traffic-proofed
(steps==workers*STEPS AND checksum!=0).

**The clean probe (system memory) still capped ~10-24 GB/s, dead flat across occupancy:** WGSIZE sweep (b128
read, constant 4096 workers) = 8.9/9.4/9.4/9.7 GB/s while maxlive ranged 2048->256. Wider loads helped
(b32/b64/b128 = 8.7/13.2/18.4), copy 24, write 17 -- all ~2-4% of 640. Occupancy/pattern/width could not
break ~20 GB/s.

**~20-24 GB/s == PCIe bandwidth.** Inspecting `AllocGpu` (inherited verbatim from MAD-304 `pm4_dispatch.cpp`):

```c
// GPU buffer (system/GTT, host-visible).
f.ui32.HostAccess = 1;
f.ui32.NonPaged   = 0;   // <-- PAGED = system/GTT RAM, GPU reads it over PCIe (~25 GB/s), NOT 640 GB/s VRAM
```

**Every buffer (A, B, C, all stream/probe buffers) lives in system RAM. The GPU has been reading operands
over PCIe the entire campaign.** This is why nothing scaled with occupancy/pattern, why the FED was 2.7 GB/s,
and why even the clean probe capped ~20. (The feedpipe's 122 was just its 64 MiB window staying in on-die GPU
cache after first touch.)

**Fix (partial) -- device-local VRAM** (`NonPaged=1, CoarseGrain=1` for the data buffers; HostAccess kept, works
on this large-BAR card -- checksums still PASS):

| config | GTT (before) | VRAM (after) |
|---|---|---|
| read b32  | 8.7 | 11.4 |
| read b64  | 13.2 | 22.7 |
| read b128 | 18.4 | **45.2** |
| copy b128 | 24.0 | **84.8** |
| write b128| 16.6 | **45.0** |

**2-4x jump, confirming the PCIe diagnosis.** Still only 7-13% of 640 -> a SECOND limiter remains (MLP/latency
on VRAM: b128>b64>b32 linear in width; copy>read because more memory ops are in flight). That's a normal
bandwidth-tuning problem (deeper software-pipelined MLP), NOT the catastrophe.

**Implications:** (1) the entire MAD-305 GEMM "feed wall" (2.7 GB/s, 1.4 TF) was measured with PCIe-fed
operands -- it must be re-baselined with A/B in VRAM. (2) The rung-1..9 / stack / lockstep analyses were all
downstream of this. **Next:** (a) roll VRAM allocation into the GEMM path (`run_wggemm_perf` A/Bshuf/C) and
re-measure the real FED -- it should jump; (b) deepen MLP (sliding-window software pipeline, MLP>>8) to push
the clean probe 45 -> hundreds of GB/s; (c) confirm full VRAM residency. *(STOP -- GPT review.)*

### VRAM fix applied to the GEMM: 1.4 TF -> 78.3 TF (56x), output still correct

Rolled the device-local VRAM allocation into the real GEMM feed (`run_wggemm_perf`: Ad/Bd -> `deviceLocal=true`;
occ kept GTT for CPU polling). `--wggemm-perf`, STORE=0, no other change:

| shape | before (GTT/PCIe) | after (VRAM) | acc00 |
|---|---|---|---|
| 1024^2 x K2048  | ~ | 3.2 TF (1.0%) | 64/64 |
| 2048^2 x K4096  | ~ | 15.4 TF (5.0%) | 64/64 |
| **4096^2 x K16384 (TARGET)** | **1.4 TF** | **78.3 TF (25.5%, 4.05 WMMA/cyc)** | 64/64 |

**56x from one alloc flag.** The entire "FED wall" (rungs 1-9, the stack ladder, the lockstep hunt) was
overwhelmingly the operands being PCIe-fed from system RAM, not any kernel/compute/scheduling effect.
Correctness held (acc00 64/64). 78.3 TF is now half the HIP winner (161) and on the path to the 250-300
pitch bar. The remaining gap is the second (legitimate) limiter -- MLP/latency on VRAM (the clean probe is at
45 read / 85 copy GB/s, ~10% of 640) -- which is normal feed/pipeline tuning, now on honest footing.

**Next:** (a) deepen MLP in the real feed (the clean BW probe shows wide-load + more-in-flight is the lever);
(b) push the clean BW probe 45 -> hundreds (sliding-window software pipeline) to know the true feed ceiling;
(c) re-run the NOFEED compute ceiling now that feed is VRAM-resident. *(STOP -- GPT review.)*

### VRAM RE-BASELINE (GPT-ordered reset): guard + canonical four @ 4096^2 x K16384, all operands device-local

Per GPT: everything before the VRAM fix is contaminated for perf; re-establish baselines BEFORE tuning.
(1) Allocation guard: `GpuBuf.vram` records device-local; `run_wggemm_perf` aborts if Ad/Bd/C aren't VRAM
(the PCIe bug cannot recur silently). C moved to VRAM too -- FED acc00 still OK (CoarseGrain VRAM CPU-read
after the release fence is coherent on this card). (2) Added a matched FEEDONLY on the REAL DBUF==1 path
(`.if FEEDONLY` skips only the 32 WMMA; all loads/prefetch/barriers/waits kept). `--baselines`:

| baseline (VRAM) | TF | % of 307 | WMMA/cyc | note |
|---|---|---|---|---|
| PM4 FED (DBUF==1 real)   | **79.2** | 25.8% | 4.10 | acc00 OK |
| PM4 FEEDONLY (DBUF==1)   | 70.2 | 22.9% | 3.64 | output N/A |
| PM4 NOFEED (compute)     | **136.1** | 44.3% | 7.05 | output N/A |
| ref: HIP FED winner      | 161 | 52% | -- | hard pass 153 |
| ref: HIP NOFEED          | ~272 | 89% | -- | -- |

**Decomposition (now honest):**
- FED 79 < NOFEED 136 -> the feed drags the compute ceiling down ~1.7x; FED is feed-bound. FEEDONLY(70) is
  *slower* than FED(79) -> the WMMA actually HIDES feed latency (the kernel is reasonably overlapped), but
  the feed is still the binding constraint on the FED path.
- NOFEED is only 136 = 44% of 307 (HIP NOFEED ~272). The compute ceiling itself is capped, by the
  maxlive=192-WG residency wall (#294/#295), now on VRAM-honest footing.

**Two levers to the pitch bar, both real:** (a) raise the NOFEED ceiling 136 -> ~272 (occupancy/residency --
the 192-WG wall); (b) close FED -> NOFEED 79 -> 136 (feed overlap/MLP). FED 79 is already 49% of the HIP
winner (161) from zero tuning. *Next per GPT: scrutinize the clean BW probe (is 45/85 GB/s a real VRAM
ceiling or still MLP/work-queue-limited?), then attack the 192-WG NOFEED wall and the feed overlap.*

### Clean BW probe scrutiny (GPT step 4): verified-VRAM, NOT MLP-limited -> issue-rate cap, and NOT the GEMM's wall

Added a VRAM guard to the BW probe (run_bw aborts if src/dst aren't device-local) + an MLP-depth sweep.
Verified-VRAM results @ WG=32, 256MiB, 2048 workers:

| probe | GB/s | %640 | note |
|---|---|---|---|
| read  b32 / b64 / b128 | 11 / 23 / 45 | 1.8/3.6/7.1% | **linear in load width** |
| copy b128 / write b128 | 85 / 45 | 13/7% | copy = read+write ports |
| b64 read UNROLL 4/8/16 | 22.7 / 22.8 / 22.8 | flat | **NOT MLP-limited** |

**Diagnosis:** BW = bytes/load x (constant ~88M coalesced loads/s) across b32/b64/b128 -> the clean probe is
**load-ISSUE-RATE capped (~88M/s), not bandwidth- or MLP-limited.** Deeper UNROLL does nothing.

**But this ceiling is NOT the GEMM's binding constraint.** The GEMM FED (79 TF) and NOFEED (136 TF) both run
ABOVE the 45 GB/s streaming ceiling because the GEMM reuses A/B tiles from **LDS** -- its global-load rate is
far below a pure stream. So chasing the clean probe to ~600 GB/s would be wasted effort; the GEMM doesn't
stream every byte from global.

**Step-5 decision -- the real levers are NOT raw streaming BW:**
1. **NOFEED 136 = 44% of 307, capped by the maxlive=192-WG residency wall** (#294/#295), now VRAM-honest.
   This is the biggest lever (136 -> ~272 = HIP NOFEED). **Attack this first.**
2. Feed overlap: FED 79 -> NOFEED 136 (the feed cost not hidden by compute).

The ~88M-loads/s issue-rate cap is a real raw-PM4 property worth understanding, but it is downstream of the
occupancy wall and the GEMM's LDS reuse hides it. *Next: raise the 192-WG NOFEED ceiling. STOP -- GPT review.*

### NOFEED ceiling ATTRIBUTION (post-VRAM, GPT step 5): 136 is NOT residency -- PM4 NOFEED reaches 287 TF (93.5%)

GPT: don't assume "192-WG wall"; attribute *why* PM4 NOFEED is only ~136. Re-ran the four sweeps @4096^2 x K16384
(all VRAM-guarded; NOFEED barely touches global so it was never PCIe-contaminated):

- **NFOCC (residency):** field26/lds8196 = 132 TF; field24/lds4100 = 139; **512WG = 102; 1024WG = 78.**
  maxlive **HARD-CAPS at 192** regardless of nWG, and over-dispatch HURTS. Smaller VGPR/LDS didn't raise it.
  -> **NOT residency-bound** (more waves don't help; 192 cap isn't VGPR/LDS-limited).
- **NFUNROLL (WMMA issue density):** KUNROLL 1/2/4/8 = 139/135/130/**185 TF**. Longer WMMA runs (fewer loop
  backedges) lift the ceiling -> backedge/loop overhead is real.
- **BANDSWP (claim granularity):** NOFEED band 1/2/4/8/16/32 = 185/**211**/140/166/128/64. Per-tile atomics
  cost; band=2 best. (FED bands 76/78/58 -> FED is band-INSENSITIVE = feed-bound, not claim-bound. acc00 OK.)
- **NFBF (barrier-free + chunk + WG shape):** 1w c1 = 156; 1w c16 = 203; **4w c16 = 287.2 TF = 93.5% of 307**
  (maxlive 215). More waves with c1 HURT (atomic contention: 156->86); chunk=16 fixes it (203). The 4-wave
  barrier-free chunk-16 config **BEATS HIP NOFEED (272-289).**

**Attribution verdict:** PM4 NOFEED is NOT residency/hardware-capped -- it reaches **287 TF (93.5%)** barrier-free.
The 136 baseline is lost to, in order: **barriers/LDS (4w barrier+LDS 132 vs barrier-free 287 ~= 2x), per-tile
atomic claims (chunk16/band2 ~= 1.4x), loop backedge density (KUNROLL8 ~= 1.3x).** Residency is a non-issue;
HIP's 272 is NOT the ceiling -- 307 is (PM4 already hit 287).

**Implication for the real FED (79 TF):** the compute ceiling it rides on is provably ~287, not 136. But FED
needs the LDS A-share + barriers for correctness (4 waves cooperate on an A-tile), and FED is feed-bound
(band-insensitive), so its levers differ from NOFEED's: (1) reduce per-K-tile barrier/LDS-publish cost while
keeping the cooperative A-share correct; (2) band/chunk the tile claim; (3) raise WMMA-per-backedge; (4) the
feed overlap (79->ceiling). *Next: design a correctness-preserving FED that adopts the barrier-free/banded/
high-KUNROLL structure that gave NOFEED 287. STOP -- GPT review.*

### FED A-LDS-ring K-WINDOW (GPT structural lever): barrier amortization gives +38%, correctness held

GPT: amortize the 4-wave A-publish barrier over U K-tiles via an A LDS ring (publish U slices, ONE barrier,
consume U slices, barrier before reuse) -- reduce barrier frequency, keep cooperative A-share correct. Built
`KWIN` defsym in occ_kernel_wggemm2.s (`.elseif KWIN`, overrides DBUF; KWIN=0 byte-identical to perf bin).
`--kwin`, correctness-gated:

| @4096^2 x K16384 | TF | % of 307 | maxlive | acc00 |
|---|---|---|---|---|
| FED baseline (DBUF==1) | 80.4 | 26.2% | 192 | OK |
| KWIN=2 (8KB ring)  | 95.1 | 31.0% | 192 | OK |
| **KWIN=4 (16KB ring)** | **111.0** | **36.2%** | 192 | OK |
| KWIN=8 (32KB ring) | 85.5 | 27.8% | **64** | OK |

**Verdict (GPT decision table): U=2/4 moved 80 -> 111 (+38%), correctness preserved -> barrier frequency WAS
a real wall.** KWIN=4 (16KB, 2 barriers/4 K-tiles) is the sweet spot: still maxlive=192. KWIN=8's 32KB ring
drops occupancy to 64 WGs (LDS-pressure cliff between 16 and 32 KB) -> regresses. FED is now **111 TF = 69%
of the HIP winner (161)** from this one structural change, no feed/compute-density work yet.

**Remaining levers (the consume phase still exposes latency):** (1) **B-prefetch one slice ahead inside the
window** (GPT step 3) -- the consume waits `s_wait_loadcnt 0x0` per slice, exposing B latency; prefetching
slice u+1's B during slice u's WMMA hides it. (2) **Window-level A double-buffer** to hide the publish-phase
A-load (currently exposed). (3) KUNROLL-style WMMA density. (4) band/chunk claim (secondary, FED band-flat).
*Next: add B-prefetch-one-ahead to the KWIN consume (cheapest, directly attacks the exposed per-slice B wait).
STOP -- GPT review.*

### KWIN + B-prefetch-one-ahead: NO-GO (regresses; B latency already hidden by occupancy)

GPT next lever: hide the per-slice B wait via B-prefetch-one-ahead inside the window. Built `KWINBPF` defsym
(2 B slots Bcur=176/Bnext=192 ping-pong; prologue issues B[t+0]; each iter issues B[t+u+1] then `s_wait_loadcnt 8`
retires B[u] while B[u+1] stays in flight -- loadcnt is B-only since A is from LDS/dscnt; gfx12 loadcnt is
in-order). KWINBPF=0 byte-identical to the simple KWIN; correctness held throughout. `--kwin` (variance ~+/-8%):

| @4096^2 x K16384 | TF | acc00 |
|---|---|---|
| FED baseline | 86.0 | OK |
| KWIN=2 / +BPF | 100.3 / 95.7 | OK |
| KWIN=4 (simple) | **119.1** | OK |
| KWIN=4 + B-prefetch | **95.5** | OK |
| KWIN=8 (32KB ring) | 86.4 (maxlive 64) | OK |

**Verdict: B-prefetch REGRESSES (119 -> 95.5), well outside variance -> B latency is NOT the bottleneck.** With
192 WGs / 768 waves resident, the per-slice B wait is already hidden across waves by occupancy; manual
prefetch only adds VGPR pressure (2 B slots = full 208) + issue overhead. **KWIN=4 simple (119 TF, 38.8%)
stays the FED best.** Remaining gap to the NOFEED ceiling (287) is ~2.4x.

**Redirect (next lever, for GPT):** the consume's B wait is hidden, so the exposed cost is likely the
**PUBLISH phase** -- per window it does KWIN x (2 b128 A-load + wait + 2 ds_store + wait) SERIALLY (the U
A-loads' latency is exposed, VGPR-bound to v16-23 so they can't overlap). Options: (1) overlap 2 A-loads
using v16-31 (16 regs = 2 slices) to halve publish exposure; (2) window-level A double-buffer (hide publish
behind the prior window's compute) -- but needs 2*U LDS slots (hits the 32KB occupancy cliff); (3) issue
density / KUNROLL inside the consume; (4) accept 119 and move to band/feed-overlap on a different axis.
*Next: probe whether the publish A-load is the residual cost (e.g., FEEDONLY-style on KWIN, or overlap 2
A-loads). STOP -- GPT review.*

### KWIN 2-wide publish: also NO-GO -> latency-hiding is exhausted; the 119->287 gap is structural, not latency

Followed the discipline (confirm the publish A-load is the residual cost by building the cheapest fix).
`KWINPUB2` defsym: overlap 2 A-slices' global loads per wait (KWIN/2 waits instead of KWIN). First attempt
HUNG (claim=13958956 runaway) -- a register bug: v24 is the ti-broadcast LDS address used by the claim loop
every tile, and the 2nd-slice A-load clobbered v24-31. Fixed by using the B-frag regs v176-183 (free during
publish). Re-run (correctness held):

| @4096^2 x K16384 (run var ~+/-8%) | TF | acc00 |
|---|---|---|
| FED baseline | 88.3 | OK |
| KWIN=4 (simple) | **119.1** | OK |
| KWIN=4 + B-prefetch | 93.1 | OK |
| KWIN=4 + 2-wide publish | 111.3 | OK |

**Verdict: 2-wide publish does NOT help (119 -> 111, slight regress).** Both latency-hiding attempts now fail
the same way (B-prefetch 119->93; publish-overlap 119->111). **The per-slice A and B load latencies are
already hidden across the 192 resident WGs (768 waves) -- so the 119 -> NOFEED-287 gap is throughput /
issue-density / structural, NOT latency.** Manual software pipelining is the wrong axis here.

**Where 119 stands:** FED 80 -> 119 TF (+49%) from the K-window barrier amortization, on top of the PCIe->VRAM
fix (1.4 -> 80, 56x). **119 TF = 74% of the HIP winner (161), 38.8% of the 307 ceiling.** The FED can't reach
the NOFEED 287 because NOFEED SKIPS the cooperative A-share (the LDS global->ds_store->barrier->ds_load
round-trip + 2 barriers/window) that the real GEMM needs for correctness -- that A-share is a real, partly
irreducible structural cost.

**Next levers (latency-hiding exhausted), for GPT:** (1) reduce the A-share STRUCTURAL cost -- fewer barriers
(KWIN=4 has 2/window; can a single reuse barrier be safe?) or fewer ds_store/ds_load instructions; (2)
issue-density / KUNROLL inside the consume (NOFEED gained 139->185 from it); (3) a smaller/wider tile shape;
(4) consolidate at 119 as the milestone. *STOP -- GPT review.*

### KWIN single-reuse-barrier: REAL RACE (tail barrier required) -- caught by the strong oracle, not acc00

GPT: try dropping the tail/reuse barrier (rely on the next window's publish barrier) but gate hard -- this is
exactly the change that can produce rare LDS races. Built `KWINNOTAIL` defsym + a RACE GATE: the full 16-frag
oracle (`run_wggemm_compute`, STORE=1) x10 repeats per shape, comparing tail vs no-tail.

| config | 256x256x256 (x10) | 512x512x2048 (x10) | verdict |
|---|---|---|---|
| KWIN=4 TAIL (safe) | 0 bad frags | 0 bad frags | PASS |
| KWIN=4 NO-TAIL | **136 bad frags** | **552 bad frags** | **RACE/FAIL** |

**The tail barrier is correctness-required.** Dropping it is a genuine cross-wave LDS race (a fast wave
overwrites an A-slot a slow wave is still reading; the next window's publish ds_store is not separated from
the prior window's consume reads). The strong full-fragment oracle x10 caught it decisively; acc00's ~64
samples might have missed it -- this is why GPT insisted on the strong gate. **KWIN=4 (safe, 2 barriers,
119 TF) stays the FED best.**

**Barrier reduction is now structurally exhausted on single-buffer:** the only race-free way to a single
barrier/window is a DOUBLE-BUFFERED window ring (2*KWIN LDS slots, ping-pong so the next publish writes a
different slot set), but that costs 2x LDS -> the 32KB occupancy cliff (192->64 WGs). For KWIN=2 double-buffer
(16KB) the barrier frequency equals KWIN=4 single-buffer anyway -- no net win.

**Exhausted axes:** latency-hiding (B-prefetch, publish-overlap: no help -- hidden by occupancy); barrier
reduction (single-barrier: race). **Next (GPT branch "fails/flat -> issue density"):** consume issue density /
KUNROLL -- though the KWIN consume already emits 32*KWIN=128 WMMA/backedge with per-slice B-loads interspersed
(batching all B up front would need KWIN*8=64 B-regs, over the 208 budget). Or consolidate at 119 TF (1.4 ->
79 VRAM -> 119 K-window; 74% of HIP winner 161; correctness-gated). *STOP -- GPT review.*

## ============ CHECKPOINT: FED 1.4 -> 119 TF, correctness-gated (MAD-305) ============

**Campaign arc (post-contamination-reset):**
| stage | TF @4096^2 x K16384 | what |
|---|---|---|
| (contaminated) | 1.4 | operands in SYSTEM RAM, GPU read over PCIe (~25 GB/s) -- INVALID, see contamination marker |
| **VRAM fix** | **79** | AllocGpu NonPaged=1 device-local -- the 56x headline; *the* root cause |
| **KWIN=4 barrier amortization** | **119** | A-LDS-ring K-window: publish 4 A-slices, 1 publish barrier, consume 4 (32*4 WMMA), 1 reuse barrier |

**119 TF = 38.8% of the 307 silicon ceiling, 74% of the HIP winner (161, hard-pass 153). Fully
correctness-gated (acc00 64/64 + full 16-frag oracle).**

**Reference points (post-VRAM, this kernel):** PM4 FED 119 ; PM4 FEEDONLY ~70 ; PM4 NOFEED (compute ceiling)
136, BUT barrier-free 4-wave chunk-16 NOFEED = **287** (93.5%, beats HIP NOFEED 272-289). So the silicon can
do ~287 compute; the real FED rides below it because of the cooperative A-share (LDS publish + 2 barriers/
window) which NOFEED-287 skips -- proven irreducible (the single-reuse-barrier RACE GATE: dropping the tail
barrier gave 136/552 bad frags).

**Axes tried and their verdicts (all on the VRAM-correct kernel):**
- Barrier amortization (KWIN): **WIN, 79->119** (the live lever).
- B-prefetch-one-ahead: NO-GO (119->93; B latency hidden by 192-WG occupancy).
- 2-wide publish (overlap A-loads): NO-GO (119->111; A latency also hidden by occupancy).
- Single-reuse-barrier (drop tail): NO-GO, REAL RACE (full-frag oracle x10 caught 136/552 bad).
- Consume issue-density / longer WMMA clusters: **REGISTER-BLOCKED** at 4x4. acc=v32-159 (128); a 64-WMMA
  cluster needs 2 slices' A (v160-191) + 2 slices' B (32 more) = v160-223 > the 208-VGPR budget (field 26).
  field 28 (224 VGPR) would fit but cuts occupancy (kernel is occupancy-sensitive: field24 > field26 in NOFEED).

**KEY LESSONS (banked):** (1) audit where operands physically live before blaming the kernel -- a verbatim
alloc flag fed the whole campaign over PCIe; (2) under high occupancy, manual latency-hiding (prefetch/
overlap) is redundant or harmful -- attack throughput/structure; (3) for race-prone changes use the strongest
oracle + repeats, not acc00 (it would have missed the 136-bad-frag race).

**What remains (NOT cheap -- deliberate decisions, not impulse probes):** (a) a structural rewrite marrying the
barrier-free/banded NOFEED-287 structure to a correct feed; (b) field-28 consume-by-2 (issue density vs
occupancy tradeoff -- measure, expect ~net-flat); (c) 2x2 tile (frees regs but 2x2 hurt NOFEED density badly --
likely trades away the reuse that makes 4x4 competitive). *Branch checkpointed at 119 TF.*

## ============ BREAKTHROUGH: 8x2 reuse tile = 162 TF, FIRST to beat HIP 161 (MAD-305, 2026-06-18) ============

**Campaign arc since the 119 checkpoint:** 119 (KWIN barrier amort) -> ~145 (KWIN=4 pub-width-4 tuning) ->
149.7 (4x4 pub-w4 saturated, prior best) -> **162.0 (8x2 reuse tile)**. 162 = **52.8% of the 307 ceiling,
100.6% of the HIP winner (161)**, vs hipBLASLt 143. Fully oracle-verified (1024/1024 frags, STORE=1).

**The model correction that unlocked it.** The ~148-150 wall is FEED-bound, WMMA free (proven: 4x4 FED 149.0
== FEEDONLY 149.8, ratio 0.995). The per-wave reuse ratio (FM+FN)/(FM*FN) SPLITS into two independent terms:
  - **B-feed/MAC = 1/FM**  <- the BINDING side (B streamed per-wave from global via global_load_tr)
  - **A-feed/MAC = 1/FN**  <- NON-binding (A is LDS-staged + shared across waves)
The earlier-measured TWN=4 result (flat 148.7 vs 149.7) already proved the A-side does not bind. So the
post-compact "4x5" plan (improves only 1/FN) would have been FLAT. **Grow FM, not FN.** 8x2 (FM=8,FN=2)
halves B-tr/MAC 0.25->0.125 (each B-frag reused across 8 M-rows not 4) at the cost of DOUBLING cheap A-LDS
reads (A-rd/MAC 0.25->0.50). Total feed/MAC went UP (0.50->0.625) yet TF ROSE +8.7% -> proves the binding
cost is specifically B-global-tr, not total-issue and not A-reads.

| geom @65536^2 x K16384, TWN=4, vgprField=26 | NOFEED | FED | FEEDONLY | FED/FO | residWv |
|---|---:|---:|---:|---:|---:|
| 4x4 (prior) | 281 | 149.0 | 149.8 | 0.995 | 512 |
| **8x2 (new)** | (hang*) | **161.7** | **182.2** | **0.887** | 512 |

**WALL ATTRIBUTION (--wall82): the wall MOVED.** 8x2 raised the FEED CEILING 149.8 -> 182.2 TF (+22%) by
halving the B feed, BUT now FED < FEEDONLY (0.887) -> **WMMA/compute/register-scheduling EMERGED** as a
co-limiter. The WMMA (same 32/slice as 4x4, absolute ceiling ~281) is no longer fully hidden because the feed
got fast enough to stop covering it. The remaining 162->182 gap (~+12%) is **latency-hiding / feed-WMMA
overlap at 64-WG/512-wave residency**, NOT feed.

**NEXT STEP (morning): build 8x2 @ TWN=2.** Doubly motivated -- stronger B-reuse tile + 4-wave/192-WG regime
= 768 resident waves (vs TWN4's 512) to RE-HIDE the emerged WMMA. 8x2 can't add residency at TWN4 (B-frags
pin a ~200-VGPR floor). Needs NBANDS=FM/TWN=4 A-fill generalization (publish hardcodes 2 bands; v24=ti-bcast
/ v25=B-LDS taken -> sequential load-store per band or reg realloc). Complementary if short: KWINBPF deeper
B-prefetch / publish-width retune to overlap WMMA; ds_load_2addr_b64 A-read halving (raises FEEDONLY>182, only
helps FED once WMMA re-hidden).

**Impl (all gated, baseline byte-identical, UNCOMMITTED):** occ_kernel_wggemm2.s -- symbolized KWIN consume
frags v160->FA / v176->FB (FA=ACC+FM*FN*8, FB=FA+2*FM*2), FM==8 branch (AROW_SH=12,TROW_SH=8,WM_SH=11).
occ_dispatch.cpp -- run_wggemm_perf + run_wggemm_compute take FNt (compute also TWN), N-axis uses FN; modes
--reuse82 (perf+oracle) and --wall82 (FED/FEEDONLY/NOFEED + oracle). build.sh -- 82_tw4_kwin4_pw4 (perf),
82_tw4_kwin4_st1 (oracle), 82_tw4_{kwin4_pw4_feedonly,nofeed} (wall). *(hang) = 8x2 NOFEED probe variant
hangs (claim runaway, WGs miss TOTAL-exit); isolated to that diagnostic bin, winner unaffected, deferred.

KG: a8ea0196 (8x2 win), 4ef267cd (wall attribution), 6fea5cfc (session pickup). *Branch at 162 TF. STOP -- GPT review.*
