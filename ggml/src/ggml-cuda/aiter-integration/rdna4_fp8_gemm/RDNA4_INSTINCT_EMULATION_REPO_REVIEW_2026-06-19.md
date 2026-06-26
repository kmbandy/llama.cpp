# Review: Maxritz/RDNA4_Instinct_Emulation_Library

Date: 2026-06-19

Repo: https://github.com/Maxritz/RDNA4_Instinct_Emulation_Library

## Verdict

Low value for the current PM4 FP8 GEMM performance push.

This repo is mostly a rocWMMA/HIP wrapper and demo scaffold. It does not contain hand-written gfx1201 assembly, PM4 dispatch work, fragment-major layouts, `global_load_tr` alternatives, dynamic VGPR handling, wide-K explicit stacking, or a serious scheduler/feed model.

## Potentially Useful Pieces

- The source is a compact negative baseline for what "naive rocWMMA RDNA4 GEMM" looks like.
- The Windows DLL export path may be useful only as a simple C ABI wrapper example.
- The GTT allocator reinforces the existing lesson: managed/GTT fallback can make correctness work but is not a performance path.
- The repo's mistaken native K=32 FP8 assumption is a useful warning. It resembles the AMD wide-K idea superficially, but the right implementation is two explicit K=16 WMMA ops fed from a 128-bit lane-major load, not a presumed native `16x16x32` gfx1201 FP8 instruction.

## Red Flags

- `src/rdna4_wmma_kernels.hip.cpp` says the FP8 kernel is a placeholder and stores `sum = 0.0f` to one output element per tile.
- `src/rdna4_multiprecision_gemm.hpp` describes native `V_WMMA_F32_16X16X32_FP8` and `V_WMMA_F32_16X16X64_FP4`; that does not match the useful gfx1201 model from AMD's RDNA4 WMMA guidance.
- `src/rdna4_wmma_gemm.hpp` uses rocWMMA fragments with naive one-fragment-per-loop structure and no B reuse, LDS strategy, or layout control.
- Several grid/block mappings look inconsistent with wave32 WMMA expectations and may duplicate work or overlaunch tiles.
- Tests include `../shared/rdna4_integration.hpp`, but the repo layout uses `src/`, so the packaged tests appear stale or broken.
- The "RT sparse" exported path is plain CSR iteration. The hipRT BVH path builds a trivial root containing all primitives, so traversal is still effectively all-nnz scanning.
- Build files and docs are strongly Windows-local and include paths from the author's machine.

## Takeaway For Our Kernel

Do not mine this for performance mechanics. The only useful action is to keep it as a checklist of what not to rely on:

1. Do not assume rocWMMA K=32 FP8 maps to the right gfx1201 hardware path.
2. Do not call managed/GTT spillover a performance feature.
3. Do not trust README performance claims unless the kernel body and test prove the math path.
4. Keep pursuing explicit wide-K load pairing, lane-major B layouts, B-stationary scheduling, and PM4/HSA metadata control.

