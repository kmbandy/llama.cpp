// vector_add_wrapper.cpp
//
// Thin C++ launcher for the AOT-compiled `vector_add` Triton kernel.
// Demonstrates the integration pattern: include the generated header(s),
// extract device pointers from our backend tensor type, dispatch to the
// generated C launcher.
//
// Status: DRAFT scaffolding. Verified end-to-end on 2026-05-17 via an
// out-of-tree POC (see /tmp/triton-aot-test/ and POC-test_vector_add.cpp
// in this directory). The actual generated function name carries a
// deterministic specialization-hash suffix that varies by
// (kernel-source, signature). For vector_add with our current
// signature the suffix is `b87d3d74_0d1d2d` — but that's only stable
// against the source frozen in this commit; any kernel edit changes it.
//
// At build time, the actual .h path will be globbed via the CMake module
// (see ../cmake/TritonAOT.cmake `${vector_add_AOT_C_FILES}`). The wrapper
// here uses a `extern "C"` include because the generated .h does NOT
// have `__cplusplus` guards (Triton upstream nit — file later).

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Verified pattern from POC: wrap the generated header in extern "C".
// At build time CMake's per-arch include dir will resolve this; the
// glob in TritonAOT.cmake picks up vector_add.<spec-hash>.h.
//
// PLACEHOLDER include name — production code will use the linker.py
// merged stable symbol OR a small dispatcher header we generate.
//
// extern "C" {
// #include "vector_add.PLACEHOLDER.h"
// }

namespace mt::aiter {

// Public C++ entry point — what the rest of ggml-hip calls.
// Signature mirrors the Triton kernel parameters minus the BLOCK_SIZE
// constexpr (folded in at AOT compile time).
//
// Returns hipSuccess on success, else propagates the HIP error from the
// kernel launcher.
hipError_t vector_add(
    hipStream_t stream,
    const float * x,
    const float * y,
    float       * out,
    int32_t       n_elements) {

    // Triton's generated launcher takes the grid dims explicitly. For
    // vector_add we use 1024-element blocks (matching the AOT signature
    // constexpr `1024` for BLOCK_SIZE).
    const unsigned BLOCK_SIZE = 1024;
    const unsigned gX = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const unsigned gY = 1;
    const unsigned gZ = 1;

    // PLACEHOLDER — actual call resolved during MAD-187 m1:
    //
    // return add_kernel_<spec-hash>(stream, gX, gY, gZ,
    //                                const_cast<float *>(x),
    //                                const_cast<float *>(y),
    //                                out,
    //                                n_elements);

    (void) stream; (void) x; (void) y; (void) out; (void) gX;
    fprintf(stderr, "mt::aiter::vector_add: wrapper is a draft placeholder — "
                    "real launcher resolved during MAD-187 milestone 1\n");
    return hipErrorNotInitialized;
}

}  // namespace mt::aiter
