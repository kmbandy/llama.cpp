# ml8 FP8 Phase 0 — Results + the perf reframe

**Date:** 2026-06-14. **Epic:** MAD-293. **Story:** MAD-294 (Phase 0).
**Hardware:** R9700 / gfx1201 (RDNA4), 64 CU, boost 2920 MHz, game 2350 MHz.

---

## 1. The denominator (official AMD spec, R9700)

Dense vs sparse is mechanical — every "Structured Sparsity" row is **exactly 2×** its dense row, and the precision ladder doubles each step:

| precision (Matrix) | dense | 2:4 sparse |
|---|---|---|
| FP16 | 191 TF | 383 TF |
| **FP8 (E4M3/E5M2)** | **383 TF** | 766 TF |
| INT8 | 383 TOP | 766 TOP |
| INT4 | 766 TOP | (—) |

We are **dense FP8** (no 2:4 pruning), so **the denominator is 383 TFLOPS**. Sparse (766) would require physically pruning weights to 2:4 + recalibration — a different, quality-costed model; deferred. Note RDNA4 has **no FP4 matrix type** (only INT4).

ml8 nuance: ml8 stores 4-bit LUT indices but **computes in fp8** (dequant index→fp8 centroid → fp8 WMMA), so its compute ceiling is **dense fp8 = 383**, not 766. The 4-bit is a memory/bandwidth win, not a compute-rate win.

## 2. Where we are (% of 383 dense FP8)

| path | best TFLOPS | **% of 383** |
|---|---|---|
| **ml8 LUT kernel** (`gemm_ml8.py` WEIGHT_FORMAT=1) | ~11 | **~2.9%** |
| `torch._scaled_mm` (hipBLASLt) | ~138 | ~36% |
| aiter a8w8-blockscale (Triton) | ~148 | ~38.6% |
| **target (aiter-on-CDNA parity)** | 306–345 | **80–90%** |

ml8 LUT kernel is **flat at ~11 TF across all shapes** (2048×{2560,9216} and 8192³ alike) — diagnostic of a per-element serial bottleneck, not a bad-tile config (a bad config still scales with size).

Microbench: `scripts/calibration/microbench_a8w8_fp8.py` (+ synthetic-layer LUT probe).
Pre/post-bump aiter-a8w8 delta was modest (+5–13% at M=2048); the bump did **not** close the gap → not a Triton-version issue.

## 3. Root cause — the LUT kernel starves the WMMA units

`gemm_ml8.py` WEIGHT_FORMAT=1 inner K-loop pays **two gathers per iteration** before `tl.dot`:

```python
b_byte = tl.gather(b_packed, byte_row_2d, axis=0)            # gather #1: [K/2,N] nibbles -> [K,N] bytes
b_idx  = ((b_byte >> shift) & 0x0F).to(int32)
b_fp8  = tl.load(centroid_lut_ptr + k*stride_lut_k + b_idx)  # gather #2: per-element indexed LUT load [K,N]
accumulator += tl.dot(a, b_fp8) * a_scale * b_scale
```

The `tl.dot` (WMMA) is fine; it's starved by two uncoalesced, serializing memory ops that materialize the weight tile each K-step. ~11 TF = the gather throughput, not the matrix throughput. (The kernel's own comment labels itself the "explicit-dequant baseline.")

## 4. The optimization (the ~30× lever, zero quality cost)

1. **Kill gather #1:** unpack nibbles arithmetically (`lo=b&0xF; hi=(b>>4)&0xF`; interleave along K) instead of `tl.gather`.
2. **Kill gather #2:** stage the 16-entry fp8 LUT into **LDS/registers** once per block (16 bytes/group) so index→value is a banked select, not a global indexed load. **The big one.**
3. **Tile/warp tune** for gfx1201 (BLOCK_M/N, warps, kpack; `num_stages=1` UAF constraint) once dequant stops being the wall.
4. **rocprof** between steps to confirm memory-bound → WMMA-bound.

**Method:** TDD — correctness oracle = dequant-in-torch vs kernel output (bit-exact within fp8); a benchmark gate (% of 383) per change. Iterate.

## 5. Where the win lands (honest)

- **Inference:** the LUT GEMM *is* the prefill compute hot-path → 11→~306 TF is the headline AMD number ("ml8 at ~80% of RDNA4 dense FP8").
- **Training:** stacks but is not the *current* trainer wall — the 4.3s micro-step is **host-bound** (89% GPU idle, ~80K dispatches; the MAD-293 graphs/fusion lever). Fast kernel raises the GPU-bound ceiling; killing dispatch is the orthogonal lever. Both needed for the "workstation card pulls real training throughput" story.

## 6. Phase 0 verdict / scope update

- **Done:** Triton bumped to `007ef1530` (#10458), **correctness-neutral** (e4m3 n_mismatch=0 pre & post); reproducible RAM-capped build script.
- **Reframe:** the substrate's binding constraint is **not** currency or config tuning — it's the **LUT GEMM at 2.9% of dense fp8**. The fp8-GEMM tuning (D) is a secondary ~2× lever (37%→80%); the **LUT kernel optimization is the primary ~30× lever** and the headline.
- **New story (file under MAD-293):** optimize the ml8 LUT GEMM to ~80% of dense fp8 (383) on gfx1201.
- D (aiter a8w8 config tuning) stays a tracked secondary lever, not the headline.
