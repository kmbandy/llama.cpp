# aiter-integration

**Status:** DRAFT scaffolding. Not yet wired into the parent ggml-hip build.

This directory contains the build-system glue + kernel sources for integrating
AMD's [aiter](https://github.com/ROCm/aiter) attention kernels and
[vllm-project/vllm](https://github.com/vllm-project/vllm)'s TurboQuant
backend into our llama.cpp fork via Triton's ahead-of-time (AOT) compilation
toolchain.

Strategic design: see [`docs/aiter-integration/ARCHITECTURE.md`](../../../../docs/aiter-integration/ARCHITECTURE.md).

## Directory layout

```
aiter-integration/
├── CMakeLists.txt          — build orchestration, references TritonAOT.cmake
├── README.md               — you're reading it
├── cmake/
│   └── TritonAOT.cmake     — reusable add_triton_aot_kernel() function
├── kernels/                — vendored / authored Triton .py source files
│   └── vector_add.py       — POC kernel (MAD-187 milestone 1)
│   # future:
│   # ├── unified_attention.py             — vendored from ROCm/aiter
│   # ├── triton_turboquant_store.py       — vendored from vllm-project/vllm
│   # └── triton_turboquant_decode.py      — vendored from vllm-project/vllm
└── wrappers/               — thin C++ launchers consumed by ggml-hip
    └── vector_add_wrapper.cpp
```

## How the build works

1. **CMake configure time:** `TritonAOT.cmake` probes Python for a working
   `triton.tools.compile`. If absent, this whole subtree is silently
   disabled (the parent build keeps working with our hand-rolled kernels).

2. **CMake build time:** For each `add_triton_aot_kernel(...)` invocation
   in `CMakeLists.txt`, Python is invoked once per `(kernel, specialization,
   arch)` tuple. Each invocation runs:

   ```
   python -m triton.tools.compile <kernel.py> \
       --kernel-name <name> \
       --target hip:<arch>:32 \         # gfx1201/RDNA4 wave32
       --signature "<typed signature with alignment hints>" \
       --grid '<launch grid expression>' \
       --num-warps 4 \
       --num-stages 1 \
       --out-name <name> \
       --out-path <out-dir>/<name>
   ```

   This emits two files in the per-arch out-dir:

   - `<name>.<spec-hash>.c` — kernel binary embedded as a static `uint8_t[]`,
     plus a `hipError_t kernel_<spec>(hipStream_t, gX, gY, gZ, ...typed_args)`
     launcher that calls `hipModuleLoad` on first use and caches the module.
   - `<name>.<spec-hash>.h` — header declaring the launcher.

3. **C++ wrapper layer:** `wrappers/*.cpp` files include the generated headers
   and expose stable ggml-friendly entry points to the rest of ggml-hip.

4. **Linkage:** The generated `.c` files + wrappers compile into a static
   library `aiter_triton_aot`. The parent ggml-hip target links it in.

## Enabling the integration

Default state: **off**. Once MAD-187 milestone 1 (vector-add POC) demonstrates
the toolchain works on gfx1201, we add to `ggml/src/ggml-cuda/CMakeLists.txt`:

```cmake
option(LLAMA_AITER_TRITON "Enable AITER + vLLM Triton AOT kernel path" OFF)
if(LLAMA_AITER_TRITON)
    add_subdirectory(aiter-integration)
    target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)
endif()
```

Then a fork-build that includes the integration:

```bash
cmake -B build-aiter -DGGML_HIP=ON -DLLAMA_AITER_TRITON=ON \
      -DGPU_TARGETS="gfx1030;gfx1201" \
      -DAITER_AOT_ARCHES="gfx1030;gfx1201" \
      .
cmake --build build-aiter -j
```

## Validation order (matches MAD-187)

1. **m1 — vector_add POC**: `cmake -DLLAMA_AITER_TRITON=ON` configures; build
   produces `aiter_triton_aot.a` containing the AOT-compiled vector_add for
   gfx1201; a small C++ test launches it on the R9700 and verifies output.
2. **m2 — Multi-arch**: same vector_add compiled for gfx1030 too (R9700 +
   6900XT). Test the gfx1030 path on the 6900XT.
3. **m3 — unified_attention**: vendor AITER's `unified_attention.py` and
   `kernel_unified_attention_{2d,3d}` from
   `aiter/ops/triton/_triton_kernels/attention/`. Wire through the same
   AOT path with `(HEAD_SIZE, KV_TYPE)` specialization matrix.
4. **m4 — turboquant**: same pattern for vLLM's `triton_turboquant_*` kernels.

## Open questions to resolve during MAD-187 m1

These can only be answered by actually running the AOT compile and inspecting
the output. Captured here so we don't lose them:

1. **Specialization-hash suffix discovery.** Triton's tools.compile emits
   files like `vector_add.add_kernel_0d1d2d3de.c`. The suffix is a
   deterministic hash of the specialization but we can't compute it
   offline. Two approaches:
   - Glob the output dir at build time (current `TritonAOT.cmake` strategy).
   - Use `triton.tools.link` to produce a stable-named wrapper from the
     specialized files. Cleaner long-term but requires another tool.
2. **HIP vs CUDA in generated .c file.** Triton's tools.compile docstring
   says the launcher returns `CUresult`; on the HIP backend it presumably
   returns `hipError_t`. Need to confirm and update the wrapper accordingly.
3. **Implicit dependencies of the generated .c file.** Does it need
   `<hip/hip_runtime.h>` or just `hip_runtime_api.h`? Does it call
   `hipModuleLaunchKernel` or the lower-level dispatch APIs?
4. **Build determinism.** Will the same kernel source produce identical
   binary output across runs / machines? Affects whether we can cache
   build artifacts in CI.

## License + attribution

The TritonAOT.cmake module and the wrapper scaffolding here are MIT
(matching the parent llama.cpp project).

Triton itself is MIT-licensed. Generated `.c` files from `triton.tools.compile`
inherit the MIT license. The AITER and vLLM kernel sources that will land
in `kernels/` later are Apache-2.0; their license headers MUST be preserved
in the vendored copies.

See [`docs/aiter-integration/ARCHITECTURE.md`](../../../../docs/aiter-integration/ARCHITECTURE.md)
section 11 for the full running attribution list.
