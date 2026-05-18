// Tiny end-to-end test for the AOT-compiled vector_add Triton kernel.
// Allocates two input vectors on device, fills them with constants, calls
// the generated launcher, copies result back, verifies.
//
// MAD-187 milestone 1: prove the AOT → hipModuleLoad → launch pipeline
// works on gfx1201 (R9700).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Generated header from Triton's AOT compiler doesn't wrap symbols in
// extern "C". Force it from our side. (Worth filing a Triton upstream
// nit — adding extern "C" guards would let C++ consumers include cleanly.)
extern "C" {
#include "vector_add.b87d3d74_0d1d2d.h"
}

#define CHECK(call)                                                  \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            fprintf(stderr, "HIP error at %s:%d  %s: %s\n",          \
                    __FILE__, __LINE__, #call, hipGetErrorString(err)); \
            return 1;                                                \
        }                                                            \
    } while (0)

int main() {
    constexpr int N = 4096;
    constexpr size_t BYTES = N * sizeof(float);

    // Host buffers
    float * x_h = (float *) std::malloc(BYTES);
    float * y_h = (float *) std::malloc(BYTES);
    float * out_h = (float *) std::malloc(BYTES);

    for (int i = 0; i < N; ++i) {
        x_h[i] = 1.0f;
        y_h[i] = 2.0f;
        out_h[i] = -1.0f;  // sentinel: must overwrite
    }

    // Device buffers
    void *x_d = nullptr, *y_d = nullptr, *out_d = nullptr;
    CHECK(hipMalloc(&x_d, BYTES));
    CHECK(hipMalloc(&y_d, BYTES));
    CHECK(hipMalloc(&out_d, BYTES));

    CHECK(hipMemcpy(x_d, x_h, BYTES, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(y_d, y_h, BYTES, hipMemcpyHostToDevice));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // The moment of truth.
    hipError_t launch_err = vector_add_b87d3d74_0d1d2d(
        stream,
        (hipDeviceptr_t) x_d,
        (hipDeviceptr_t) y_d,
        (hipDeviceptr_t) out_d,
        N);
    if (launch_err != hipSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", hipGetErrorString(launch_err));
        return 1;
    }

    CHECK(hipStreamSynchronize(stream));

    CHECK(hipMemcpy(out_h, out_d, BYTES, hipMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (out_h[i] != 3.0f) {
            if (errors < 5) {
                fprintf(stderr, "mismatch at i=%d  got %f, expected 3.0\n", i, out_h[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("✓ vector_add AOT test PASSED — N=%d, all elements = 3.0\n", N);
    } else {
        printf("✗ vector_add AOT test FAILED — %d/%d mismatches\n", errors, N);
    }

    // Cleanup
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(x_d));
    CHECK(hipFree(y_d));
    CHECK(hipFree(out_d));
    std::free(x_h);
    std::free(y_h);
    std::free(out_h);

    // Optional: unload the module (proves we have that API surface too)
    unload_vector_add_b87d3d74_0d1d2d();

    return errors == 0 ? 0 : 2;
}
