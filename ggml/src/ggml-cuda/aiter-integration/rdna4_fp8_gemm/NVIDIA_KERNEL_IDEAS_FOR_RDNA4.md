# NVIDIA Kernel Ideas To Steal For RDNA4 FP8 GEMM

Date: 2026-06-19

This note is not about porting CUDA instructions to RDNA4. It is about stealing kernel-design patterns from NVIDIA's better-developed GEMM/attention ecosystem and translating them into RDNA4/gfx1201 experiments.

## 1. Persistent Tile Scheduler, Not Just Bigger Tiles

NVIDIA pattern:

- CUTLASS Hopper/Blackwell GEMMs have persistent kernels and explicit tile scheduler parameters.
- Blackwell adds dynamic persistence scheduling and stream-K schedulers.
- CUTLASS examples tune raster order and CTA swizzle because cross-CTA locality matters.

RDNA4 translation:

- The current PM4 kernel already has an atomic tile-claim mechanism. Treat that as a persistent scheduler, not just a workaround for missing workgroup IDs.
- Add real scheduler knobs:
  - M-major, N-major, and grouped/raster-swizzled tile order.
  - Per-CU tile strips to improve B/L2 locality.
  - Stream-K variants for skinny or imbalanced training shapes.

Experiment:

- Keep the 8x2 kernel unchanged and only change tile order.
- Measure whether B global feed improves from L2 reuse.
- If yes, make scheduler selection shape-dependent.

## 2. Warp Specialization Becomes Wave Specialization

NVIDIA pattern:

- FlashAttention-3 and Hopper/Blackwell CUTLASS split producer and consumer roles to overlap Tensor Core work with TMA/data movement.
- ThunderKittens names this as a Load-Store-Compute-Finish template.

RDNA4 translation:

- RDNA4 lacks Hopper TMA/WGMMA semantics, but a workgroup can still dedicate waves to roles:
  - B-loader wave(s): pull next B fragments or B LDS ring.
  - A-publisher wave(s): publish A to LDS.
  - Compute wave(s): issue dense WMMA.
- The key is not role purity forever; it is temporal separation so compute waves are not interleaving address/feed ops every few instructions.

Experiment:

- Prototype a 4-wave workgroup with 1 loader wave and 3 compute waves.
- First use a tiny K window and LDS handoff, even if it is not faster.
- Use FEEDONLY and FED separately to tell whether specialization improves overlap or just adds barriers.

## 3. Layouts Are A First-Class API

NVIDIA pattern:

- CuTe makes thread/value layouts explicit.
- CUTLASS 3.x explicitly moved away from implicit iterator math toward layout algebra as the source of truth.
- CUTLASS mixed-dtype guidance says narrow types should be reordered so each thread reads contiguous data in global/shared memory.
- CUTLASS exposes helper concepts like memory reordering atoms instead of treating layout as a hidden iterator detail.

RDNA4 translation:

- Stop treating B/A repack as a private benchmark prepass. Define a fragment-major contract:
  - lane-major B for plain `global_load_b128`,
  - wide-K K=32 lane packs,
  - scale layout adjacent to the fragment that consumes it.
- Add layout names and oracles so experiments do not blur together.

Experiment:

- Create explicit layout IDs: `B_TR`, `B_LANE64`, `B_LANE128_WIDEK`, `A_LDS`, `A_LANE128_WIDEK`.
- Make the host repacker and correctness oracle print the layout ID.
- Only compare kernels with known layout contracts.

Local references:

- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201_tiled.cu`
- `/home/kmbandy/GitHub/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`

## 4. Blockscale In The Mainloop, Not Afterthought Scaling

NVIDIA pattern:

- CUTLASS Hopper example 67 has FP8 GEMM with blockwise scale factors integrated into the mainloop and epilogue.
- Blackwell examples extend blockscaled and groupwise GEMMs heavily.

RDNA4 translation:

- Training wants scaling to be a core data path, not a scalar side channel.
- If scales are loaded late or through a conflicting path, they can become the next bottleneck after B feed.

Experiment:

- Add a scale-feed microbenchmark before full blockscale GEMM:
  - scale in SGPR,
  - scale in VGPR loaded alongside B,
  - scale staged in LDS with B.
- Track whether scale placement changes WMMA issue density even when math is mocked.

Layout rule:

- Scale layout should follow the MMA K step, not just the logical tensor layout.
- For blockscale FP8, scale lookup must be cheap inside the K loop and naturally aligned with K grouping.

Local references:

- `/home/kmbandy/GitHub/pytorch/third_party/cutlass/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu`
- `/home/kmbandy/GitHub/pytorch/third_party/cutlass/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu`
- `/home/kmbandy/GitHub/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`

## 5. Overlapping Accumulators / Epilogue Latency Hiding

NVIDIA pattern:

- CUTLASS changelog calls out an overlapping accumulator optimization for block tile N=256 to hide epilogue latency.
- CK's WMMA pipeline has a similar idea: leak the last WMMA block into epilogue to cover LDS/shuffle latency.

RDNA4 translation:

- The current kernel focuses on mainloop TFLOPS, but training needs epilogue work: store, scale, maybe cast, maybe bias/activation for fused paths.
- Do not wait until the end of all accumulators to start epilogue movement if some accumulator groups are done.

Experiment:

- Split accumulators into two groups.
- While group 1 stores/converts, group 2 finishes the last K steps.
- Use this only after mainloop feed is less binding; otherwise it will mask the wrong issue.

## 6. Tile Swizzle / L2 Locality Search

NVIDIA pattern:

- CUTLASS exposes rasterization direction and max swizzle size as benchmark knobs.
- Grouped GEMM applies swizzle per group based on shape.

RDNA4 translation:

- B is the expensive operand. Tile scheduling should maximize reuse of a B panel across adjacent M tiles.
- Current atomic claim order may be leaving L2 reuse on the floor.

Experiment:

- Add scheduler modes:
  - `N_STATIONARY`: hold N/B panel, sweep M.
  - `M_STATIONARY`: current or row-major order.
  - `BLOCK_SWIZZLE_2/4/8`: small Morton-like grouping of output tiles.
- Run FEEDONLY first. If only FED changes, it is issue/residency rather than cache.

## 7. Split-K / Stream-K As Load Balancing, Not Just Reduction

NVIDIA pattern:

- CUTLASS has Stream-K and persistent schedulers to handle shapes where ordinary CTA tiling underfills or imbalances the device.
- Blackwell makes stream-K a composable scheduler feature.

RDNA4 translation:

- For large training GEMMs, split-K may be less about parallelism and more about allowing a smaller, more efficient inner tile that avoids barriers or register cliffs.

Experiment:

- Compare:
  - cooperative 8x2 full-K,
  - smaller 1- or 2-wave split-K with cleaner B feed,
  - reduction overhead included.
- Do not reject split-K because it duplicates C writes; reject it only if end-to-end step time loses.

Local references:

- `/home/kmbandy/GitHub/pytorch/third_party/cutlass/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu`
- `/home/kmbandy/GitHub/pytorch/third_party/composable_kernel/include/ck_tile/ops/common/streamk_common.hpp`
- `/home/kmbandy/GitHub/pytorch/third_party/composable_kernel/include/ck_tile/core/utility/persistent_async_input_scheduler.hpp`
- `/home/kmbandy/GitHub/pytorch/third_party/composable_kernel/example/ck_tile/03_gemm/gemm_splitk_two_stage.cpp`
- `/home/kmbandy/GitHub/pytorch/third_party/composable_kernel/test/ck_tile/gemm_streamk/extended_tests/test_gemm_streamk_fp8_persistent.cpp`

## 8. Asymmetric Scaling Mindset

NVIDIA pattern:

- FlashAttention-4 explicitly changes pipelines because Blackwell tensor throughput scaled faster than shared memory and non-matmul units.

RDNA4 translation:

- Your R9700 measurement already says the WMMA silicon can do much more than the current kernel. The bottleneck is not "FP8 compute"; it is feed, issue overlap, LDS/barriers, and scheduling.
- Any design that increases arithmetic intensity by reusing B more is likely more valuable than a symmetric A/B optimization.

Experiment:

- Rank every proposed kernel by `B global bytes per WMMA` and `B issue instructions per WMMA` before coding it.
- Prefer asymmetric tiles that make B stationary even if A traffic increases moderately.

## 9. Compiler-Control Lessons

NVIDIA pattern:

- CUTLASS notes advanced compiler control files for specific kernels/toolkits.
- Triton exposes layout selection, warp specialization, and persistent scheduling as explicit compiler IR concepts.

RDNA4 translation:

- Hand assembly gives control, but the same discipline applies:
  - freeze waitcnt placement,
  - freeze register layout,
  - freeze scheduling priority,
  - archive disassembly and perf next to each variant.

Experiment:

- Add a variant manifest that records:
  - VGPR bases/pads,
  - waitcnt distances,
  - `s_setprio` polarity,
  - B layout ID,
  - tile scheduler mode.
- This turns tuning into a reproducible search rather than a pile of ad hoc assembly diffs.

## 10. Ragged/Grouped Shapes Without Blanket Padding

NVIDIA pattern:

- Hopper grouped FP8 work uses descriptor/setup machinery to avoid padding every group to a fixed large alignment.
- CUTLASS grouped and MoE kernels have separate schedulers for ragged or grouped work.

RDNA4 translation:

- We do not have TMA descriptor pools in the same sense, but we can still avoid over-padding by having a small set of precomputed residual load/store variants.
- This matters for training/MoE shapes where padding to 128 can hide a nominally fast kernel behind wasted math.

Experiment:

- Implement residual handlers only for common tails: M/N/K mod 16 and K mod 32.
- Compare against padding in end-to-end grouped GEMM or MoE-like batches.
- Keep the first version out of the peak square-GEMM path.

## Highest-Priority RDNA4 Experiments From NVIDIA Cross-Pollination

1. Persistent scheduler tile-order sweep for B/L2 locality.
2. Wide-K lane-major `global_load_b128` for both A and B.
3. Wave-specialized B-loader prototype.
4. Fragment layout IDs plus correctness oracles.
5. Scale-feed microbench before full blockscale training GEMM.
6. Split-K smaller-tile experiment with end-to-end reduction included.
7. Residual/tail variants for grouped FP8 without blanket padding.

## Sources

- CUTLASS local examples: `/home/kmbandy/GitHub/pytorch/third_party/cutlass/examples/54_hopper_fp8_warp_specialized_gemm`, `67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling`, `74_blackwell_gemm_streamk`, `87_blackwell_geforce_gemm_blockwise`, `92_blackwell_moe_gemm`, `94_ada_fp8_blockwise`.
- CUTLASS README/changelog: https://github.com/NVIDIA/cutlass
- CUTLASS 3.x design: https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_3x_design.html
- CUTLASS GEMM API: https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html
- CUTLASS grouped scheduler docs: https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html
- CuTe layout algebra paper: https://arxiv.org/abs/2603.02298
- Stream-K paper: https://arxiv.org/abs/2301.03598
- FlashAttention-3 paper: https://arxiv.org/abs/2407.08608
- FlashAttention-4 paper: https://arxiv.org/abs/2603.05451
- FlashAttention repo: https://github.com/Dao-AILab/flash-attention
- ThunderKittens repo: https://github.com/HazyResearch/ThunderKittens
