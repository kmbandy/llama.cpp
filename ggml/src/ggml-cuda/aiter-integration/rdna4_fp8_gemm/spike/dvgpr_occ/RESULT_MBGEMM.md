# RESULT (MAD-305): fed fat-tile micro-batch GEMM + the three-way feed-wall map

**Date:** 2026-06-16. **HW:** AMD R9700 / gfx1201 (RDNA4), KFD node 1, wave32, ROCm 7.2.3.
Follows the Phase-2 dyn-VGPR occupancy proof (`RESULT.md`) and Phase-3 (`RESULT_P3.md`).

## The question

We hit **276 TF (90% of the 307 fp8-WMMA ceiling)** with the persistent micro-batch dyn-VGPR
vehicle on a **pure-WMMA** (feed-free) load. Does that transfer to a **real fp8 GEMM** with a real
per-K memory feed — i.e. can the micro-batch architecture beat the conventional **155-TF** LDS
kernel and approach the 250–300 pitch bar?

## What was built

- **`occ_kernel_mbgemm.s`** — fed fat-tile micro-batch GEMM on the proven raw-PM4 vehicle.
  Parameterized `FM×FN` accumulator tile, `DYNVGPR`, `BATCH` (tiles/atomic-grab), via `-defsym`.
  Real feed: **A** direct-from-L2 `global_load_b64`, **B** `global_load_tr_b64` from a pre-shuffled
  tile-major buffer (the Phase-1 lever). Power-of-2 column-tile decode (shift/mask, no ISA divide).
  Double-buffered K-prefetch (unroll-by-2). Per-tile grow → ship → shrink. **Bit-exact** vs a
  chained-`wmma_ref` oracle. Two isolation knobs: `NOFEED=1` (operands loaded once, reused — removes
  the feed) and `PROFILE=1` (workgroup-0 accumulates the in-hardware 100 MHz REALTIME counter into
  6 per-phase buckets → `occ[24..44]`).
- **`occ_dispatch.cpp`** — `--mbgemm` (reuse/tile + K-depth + NO-FEED sweeps; static big tiles to
  256 VGPR with no umr; `--fat` dyn big tiles via the umr cap-lift), `--mbprof` (phase-timing
  table), `--microbatch` extended to KDEPTH 1 M.
- **`bench/gemm_fp8_levers.hip`** — added a K-depth sweep on the 155 winner + a `PROFILE_WINNER`
  isolated-dispatch path so **rocprofv3** can read real counters on it.

## Result — the feed wall, measured three independent ways

**1. Feed-free matrix ceiling = 305 TF = 99.5%** (the 276 was *under-fed*).
`--microbatch`, occ 8, 2048 tiles, KDEPTH up:

| KDEPTH | 65536 | 131072 | 262144 | 524288 | 1048576 |
|--------|-------|--------|--------|--------|---------|
| TF     | 274   | 288    | 296    | 302    | **305** |
| %307   | 89    | 94     | 96     | 98     | **99.5** |

`okTiles 0/2048` at KDEPTH ≥ 262144 is **f32 oracle drift** (acc sums the same product 262k+ times
past the 5e-3 gate), *not* a compute error — bit-exact held 2048/2048 at 65536 & 131072; the
throughput is valid (real WMMAs, validated 100 MHz timer, dyn ≈ static).

**2. Fed micro-batch GEMM = feed-bound** (`--mbprof`, in-kernel REALTIME, workgroup-0, 2×4 b32, 2048³):

| phase | % of total |
|-------|-----------|
| COMPUTE (K-loop incl. feed `s_wait`) | 94.8 |
| ATOMIC (grab) | 3.6 |
| SETUP (decode+zero) | 1.5 |
| STORE / GROW / SHRINK | ~0 |

Isolating with `NOFEED`: the K-loop is **18.3 ms fed vs 4.9 ms no-feed**, so the **per-K memory-feed
stall = 13.4 ms = 69 % of total time**; matrix compute 26 %; *all* per-tile overhead ~5 %. The
micro-batch has **no feed-hiding**, so a real GEMM tops out ~0.9 TF — the wall is the feed, not the
framework. (rocprof cannot see this raw-PM4 kernel — it runs under the HSA layer rocprof hooks.)

**3. Conventional 155/LDS kernel = issue-bound at 54%** (real `rocprofv3`, the one thing rocprof
*can* see). `gemm_fp8_lever<128,128,2,2,32,AFEED=0,DBUF=1>` (= **A staged in LDS, 4 cooperating
waves** = the LDS-cooperative method):

- `GRBM_GUI_ACTIVE` = 3.88 M cyc, `GRBM_COUNT` = 3.88 M → **GPU 100 % busy** (not idle/throttled).
- 33.55 M WMMA ÷ 3.88 M cyc = **8.6 WMMA/cycle = 54 % of the 15.9 ceiling**.
- `SQ_INSTS_VALU` reads 0 (known gfx1201 counter gap) — but GRBM gives the efficiency directly.
- K-sweep (M=N=4096): 4096→146, **8192→158 (peak)**, 16384→158, 32768→156, 65536→106 (L2 thrash).
  So 155–158 ≈ 54 % is its **true steady-state ceiling — not under-fed**.

## Conclusion

The matrix unit runs at ~**100 %** feed-free; the best **real** fed fp8 GEMM saturates at ~**54 %**
(≈155–158 TF) because **WMMA and the operand feed share the one VALU issue port** on RDNA — this is
silicon, and it's exactly where hipBLASLt (143) lands too. **250–300 TF on a real *square* fp8 GEMM
is not reachable on gfx1201.** The 305↔155 gap *is* the feed, confirmed from three sides.

## Next direction — merge, don't pick

The two kernels are complementary: the **micro-batch** has the scheduler (persistent atomic
work-queue = stream-K load-balance) + a matrix-saturation vehicle but no feed-hiding; the **155/LDS**
kernel has the feed-hider (shared-memory A tile) but a rigid grid scheduler. The merge —
**stream-K + shared-memory tiling** (the CUTLASS/CK pattern) — keeps both. On a uniform square GEMM
it lands ~155 (feed wall is silicon); its real win is **irregular work — the 35B-A3B MoE fp8 GEMM**,
where the stream-K queue's load-balancing + LDS feed-hiding *stack* and beat a tiled kernel that
strands waves on the ragged tail. First build: in HIP (this kernel is ~90 % of it — swap the
grid-scheduler for a persistent atomic queue), benchmarked on a ragged/MoE shape vs the tiled 155.
