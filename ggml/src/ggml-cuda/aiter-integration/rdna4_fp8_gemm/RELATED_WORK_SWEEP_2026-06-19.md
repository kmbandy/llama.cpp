# RDNA4 FP8 GEMM Related Work Sweep

Date: 2026-06-19

This note captures the repo and web sweep for adjacent RDNA4/gfx12 FP8 GEMM work. It is focused on ideas that can change the current hand-rolled PM4 kernel, not general FP8 background.

## Highest-Signal Leads

### AMD RDNA4 WMMA wide-K guide

Source: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/

AMD's June 2026 RDNA4 guide says the native low-precision WMMA fragment gives each lane 8 contiguous FP8/INT8 elements, which is only 64 bits. Their proposed fix is to fuse two 16x16x16 WMMA operations into a logical 16x16x32 operation so the main loop can use 128-bit vector loads.

Why it matters here:

- This is a direct public confirmation of the current feed bottleneck model.
- It gives a concrete experiment beyond `global_load_tr_b64`: pack two K-adjacent fragments per lane, issue one `global_load_b128` or equivalent, then feed two WMMA ops from the two halves.
- This overlaps with, but is not identical to, the lane-major B idea in `NEW_IDEAS.md`: wide-K should be tested for both A and B, while lane-major B is specifically about escaping the transpose-load path.

First local experiment:

- Make a single-wave 16x16x32 oracle that uses plain 128-bit loads for A and B from a prepacked K-major lane layout.
- Split the 16 bytes into low/high 8-byte fragments and issue two `v_wmma_f32_16x16x16_fp8_fp8` instructions.
- Only after that passes, plug it into the 8x2 wave-group kernel.

### AMD RDNA4 register-resident fusion and transpose guides

Sources:

- https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/
- https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/

Part 1 keeps the output of one GEMM in registers and reuses it as the input to the next GEMM. Part 3 uses RDNA4's WMMA layout plus an identity matrix to transpose in registers with no memory traffic.

Why it matters here:

- For training, the highest-value optimization may be producer/consumer fusion rather than only improving standalone GEMM.
- The identity-WMMA transpose suggests a possible alternative to writing/repacking intermediate tiles when the next op wants the opposite fragment orientation.
- This strengthens the `Producer-Side Fragment-Major Layout` idea: if a previous op can produce a register or lane-major fragment, avoid a global repack completely.

### AITER local gfx1201/gfx1250 work

Local paths:

- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201.cu`
- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201_w64.cu`
- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201_tiled.cu`
- `/home/kmbandy/GitHub/aiter/csrc/include/opus/opus.hpp`
- `/home/kmbandy/GitHub/aiter/aiter/ops/flydsl/kernels/gemm_fp8fp4_gfx1250.py`
- `/home/kmbandy/GitHub/aiter/csrc/ck_gemm_a8w8_bpreshuffle/`
- `/home/kmbandy/GitHub/aiter/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE.json`
- `/home/kmbandy/GitHub/aiter/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=4096-K=4096.json`

Useful observations:

- `test_wmma_gfx1201.cu` documents the gfx12 fragment map used by the direct builtins: A is row-distributed, while B/C are column-distributed. That matches the current hand-written layout assumptions and is a good oracle source.
- `test_wmma_gfx1201_w64.cu` shows a wave64 mapping trap: C/D row groups are interleaved with a small LUT. Keep this around as a guard against accidental wave64 compiler or descriptor drift.
- `test_wmma_gfx1201_tiled.cu` is the likely repair target for a gfx12-specific tiled adaptor. If this high-level adaptor can be made correct on gfx1201, its partition layouts become another oracle for the hand-written assembly.
- `opus.hpp` exposes gfx1201/gfx1200 wave32 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. It does not expose a native gfx1201 16x16x64 or 16x16x128 FP8 op; wide-K has to be explicit stacking.
- The gfx1250 FlyDSL FP8 kernel has schedule names worth mining: `fp8_quadrant`, `fp8_deep_pipeline`, and `b_streaming`. Even though this is not the same ISA target, the scheduling vocabulary maps to current experiments: B-stationary order, A-fragment prefetch, and B feed isolation.
- AITER has production `a8w8_bpreshuffle` and blockscale/preshuffle code paths. That supports treating B layout as an operator contract, not a private kernel detail.
- AITER's gfx1201 Triton A8W8 blockscale configs use `BK=128`, `kpack=2`, and vary `waves_per_eu` by shape. Preshuffled configs often use lower `waves_per_eu` and `num_stages=1`. Treat these as search priors, not proof, because they come from a different compiler/runtime path.

### Triton AMD backend clues

Local paths:

- `/home/kmbandy/GitHub/triton/third_party/amd/python/triton_amd.cc`
- `/home/kmbandy/GitHub/triton/third_party/amd/lib/TritonAMDGPUTransforms/MoveUpPrologueLoads.cpp`
- `/home/kmbandy/GitHub/triton/third_party/amd/lib/TritonAMDGPUTransforms/OptimizeBufferOpPtr.cpp`
- `/home/kmbandy/GitHub/triton/third_party/amd/lib/TritonAMDGPUTransforms/WmmaGroup.cpp`

Useful observations:

- The AMD pass list includes load hoisting, pointer canonicalization, buffer pointer optimization, block ping-pong, wait-count update, and in-thread transpose. Those names map cleanly to hand-asm knobs: hoist loads, cut loop address VALU, control waitcnt placement, and test register transpose alternatives.
- `MoveUpPrologueLoads.cpp` explicitly treats early load issue as a register-pressure tradeoff. That is the compiler version of the current `FED < FEEDONLY` problem: extra live fragments may help latency hiding until VGPR pressure costs more than it saves.
- `OptimizeBufferOpPtr.cpp` targets loop address-add pressure. If the assembly still burns VALU on per-fragment address math, a pointer-increment rewrite could improve WMMA/feed overlap without changing the tile.
- `WmmaGroup.cpp` maps gfx12 FP8/BF8 WMMA to 16x16x16, while later WMMA variants prefer wider K when legal. This reinforces testing explicit wide-K stacking on gfx1201.

### CK transpose-load and preshuffle paths

Local paths:

- `/home/kmbandy/GitHub/rocblas-pkg/src/rocm-libraries/projects/composablekernel/include/ck/utility/amd_transpose_load.hpp`
- `/home/kmbandy/GitHub/rocblas-pkg/src/rocm-libraries/projects/composablekernel/include/ck/utility/amd_wmma.hpp`
- `/home/kmbandy/GitHub/rocblas-pkg/src/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`
- `/home/kmbandy/GitHub/rocblas-pkg/src/rocm-libraries/projects/composablekernel/test/gemm_universal/test_gemm_universal_wmma_fp8.cpp`

Useful observations:

- CK's `amd_global_load_transpose_to_vgpr` uses `__builtin_amdgcn_global_load_tr_b64_v2i32` for 1-byte types and `__builtin_amdgcn_global_load_tr_b128_v8f16` for 2-byte types.
- I did not find a public CK path using transpose-load b128 for FP8. That keeps the plain 128-bit lane-major FP8 load experiment alive.
- CK has gfx12 WMMA wrappers for `f8/f8`, `f8/bf8`, `bf8/f8`, and `bf8/bf8`, plus WMMA FP8 tests. It is useful for validating builtin signatures and C++ oracle behavior, less useful for PM4/dyn-VGPR mechanics.
- Installed CK headers under `/opt/rocm/include/ck` include a direct-VMEM "Skip LDS" WMMA mode and a WMMA pipeline that intentionally lets the final WMMA block overlap epilogue latency. Both are worth translating into assembly experiments if the current barrier/LDS handoff remains visible.

### Dynamic VGPR and launch metadata

Local paths:

- `/opt/rocm/lib/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h`
- `/opt/rocm/include/hsa/amd_hsa_kernel_code.h`
- `/home/kmbandy/GitHub/aiter/csrc/include/aiter_hip_common.h`

Useful observations:

- LLVM's descriptor header is a better source of truth than the public HSA C header for newer gfx12 dynamic-VGPR metadata. The local header shows gfx120 dynamic VGPR in RSRC2 bit 6 and gfx125 moving related control to RSRC3 bit 17.
- AITER's hand-asm launch path uses `hipModuleLaunchKernel` with extra-buffer args and validates HSACO metadata such as `group_segment_fixed_size` before launch. This is a practical reference for moving static hand-asm kernels toward normal HSA/HIP launch while preserving metadata checks.

### HipKittens

Sources:

- https://github.com/HazyResearch/HipKittens
- https://arxiv.org/abs/2511.08083

HipKittens targets CDNA3/CDNA4 rather than RDNA4, but it is relevant for kernel organization. The README describes tensor-core-sized tile primitives, coalesced/bank-conflict-free tile memory ops, direct buffer loads to shared memory, and two overlap schedules: 8-wave ping-pong and 4-wave interleave.

Why it matters here:

- It supports trying explicit wave specialization and interleave patterns even if the exact instructions differ.
- The main transferable idea is not their code; it is their scheduling discipline: separate tile movement from MMA consumption and measure the overlap schedule independently.

## Negative Findings

- I did not find a recent public gfx1201 FP8 GEMM that is both hand-written PM4 and built around `S_ALLOC_VGPR`.
- I did not find a public implementation using `global_load_tr_b128` for FP8. The visible CK wrapper uses transpose b64 for 1-byte element types.
- The public AITER/CK/Triton paths are useful for layouts and builtins, but they do not appear to be solving the same PM4 wave-group/dynamic-VGPR problem.

## New Experiments Added By The Sweep

1. Add a `WIDEK_B128` oracle: K=32 logical WMMA, one 128-bit load per operand fragment, two K=16 WMMA ops.
2. Compare `WIDEK_B128` against `BPLAIN_B128`: wide-K tests 128-bit feed for both operands; B-plain tests avoiding the transpose-load path specifically.
3. Extract the AITER gfx1201 direct-WMMA test into this folder as a correctness oracle for fragment layout, or at least copy its lane map into `frag_layout.h` tests.
4. Mine AITER FlyDSL's `fp8_deep_pipeline` and `b_streaming` schedules for issue order, not syntax.
5. Use CK's `amd_transpose_load.hpp` as evidence to deprioritize waiting for an FP8 transpose b128 miracle; treat plain b128 prepack as the more plausible path.
6. Add a descriptor audit test that checks the PM4/HSA metadata bits for wave32, dynamic VGPR enablement, LDS size, and workgroup size before performance runs.
7. Try a pointer-increment address schedule that removes repeated VALU address adds in the hot K loop, mirroring Triton's buffer pointer optimization pass.
