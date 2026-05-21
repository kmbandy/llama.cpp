// MAD-214 Phase 1E test: validate the GPU FWHT kernel matches the CPU
// reference implementation across all supported head_dim values.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -I ggml/src/ggml-cuda -o /tmp/test_turbo_fp8_hadamard \
//       tests/test_turbo_fp8_hadamard.cu

#include "turbo_fp8_hadamard.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

static bool test_one_head_dim(int head_dim, int n_rows, std::mt19937 & rng) {
    printf("[test_fwht head_dim=%4d, n_rows=%d] ", head_dim, n_rows);
    fflush(stdout);

    const size_t n_floats = (size_t) n_rows * head_dim;
    std::vector<float> input(n_floats);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : input) v = dist(rng);

    // CPU reference: apply FWHT in-place to a copy
    std::vector<float> ref(input);
    mt_turbo_fp8_fwht_reference_cpu(ref.data(), n_rows, head_dim, head_dim);

    // GPU: copy input, apply FWHT, copy back
    float * d_data = nullptr;
    hipError_t err;
    err = hipMalloc(&d_data, n_floats * sizeof(float));
    if (err != hipSuccess) { printf("hipMalloc FAIL\n"); return false; }
    err = hipMemcpy(d_data, input.data(), n_floats * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { hipFree(d_data); printf("hipMemcpy H2D FAIL\n"); return false; }

    err = mt_turbo_fp8_fwht(0 /*stream*/, d_data, n_rows, head_dim, head_dim);
    if (err != hipSuccess) {
        hipFree(d_data);
        printf("kernel launch FAIL: %s\n", hipGetErrorString(err));
        return false;
    }
    hipDeviceSynchronize();

    std::vector<float> gpu_out(n_floats);
    err = hipMemcpy(gpu_out.data(), d_data, n_floats * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_data);
    if (err != hipSuccess) { printf("hipMemcpy D2H FAIL\n"); return false; }

    // Compare element-wise
    float max_abs_err = 0.0f, sum_sq_err = 0.0f;
    int n_mismatches = 0;
    for (size_t i = 0; i < n_floats; ++i) {
        float d = std::fabs(gpu_out[i] - ref[i]);
        max_abs_err = std::max(max_abs_err, d);
        sum_sq_err += d * d;
        if (d > 1e-3f) ++n_mismatches;
    }
    float rms_err = std::sqrt(sum_sq_err / (float) n_floats);
    bool ok = max_abs_err < 1e-3f;
    printf("%s max_err=%.4g  rms=%.4g  mismatches=%d\n",
           ok ? "OK" : "FAIL", max_abs_err, rms_err, n_mismatches);

    // Round-trip sanity (Hadamard is self-inverse up to a 1/d factor — applying
    // FWHT twice and scaling by sqrt(d) twice should return the input).
    if (ok && n_rows == 1) {
        std::vector<float> rt(ref);  // apply FWHT again to the GPU output
        mt_turbo_fp8_fwht_reference_cpu(rt.data(), n_rows, head_dim, head_dim);
        // After 2 FWHT applications with 1/sqrt(d) normalization each, the
        // total scaling is 1/d (FWHT not normalized) * (sqrt(d))^2 / d = 1/d * d = 1.
        // Should match input exactly (up to fp32 precision).
        float roundtrip_max = 0.0f;
        for (size_t i = 0; i < n_floats; ++i)
            roundtrip_max = std::max(roundtrip_max, std::fabs(rt[i] - input[i]));
        printf("                                 round-trip max_err=%.4g  %s\n",
               roundtrip_max, roundtrip_max < 1e-4f ? "OK" : "WARN");
    }

    return ok;
}

int main() {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    std::mt19937 rng(2026);
    bool all_ok = true;
    for (int d : {16, 32, 64, 128, 256, 512, 1024}) {
        all_ok &= test_one_head_dim(d, 8, rng);
    }
    printf("\n=== %s ===\n", all_ok ? "ALL FWHT TESTS PASSED" : "SOME FWHT TESTS FAILED");
    return all_ok ? 0 : 1;
}
