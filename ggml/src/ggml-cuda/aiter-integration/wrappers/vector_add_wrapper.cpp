// vector_add_wrapper.cpp
//
// Thin C++ launcher for the AOT-compiled `vector_add` Triton kernel.
// Demonstrates the integration pattern: include the generated header(s),
// extract device pointers from our backend tensor type, dispatch to the
// generated C launcher.
//
// Status: DRAFT — wrapper structure is illustrative. The actual function
// names from Triton's tools.compile output have a deterministic
// specialization-hash suffix (e.g. `add_kernel_0d1d2d3de`) that we don't
// know offline. MAD-187 milestone 1 will:
//   1. Run the AOT compile and inspect the produced .h file
//   2. Update this wrapper with the exact symbol name OR use the
//      `triton.tools.link` aggregator to produce a stable wrapper symbol

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Generated header — produced by `triton.tools.compile` into
// ${CMAKE_BINARY_DIR}/aiter-integration/triton-out/vector_add/<arch>/
// vector_add.<spec-hash>.h
//
// PLACEHOLDER include — actual name will be discovered at MAD-187 m1.
// #include "vector_add.PLACEHOLDER.h"

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
