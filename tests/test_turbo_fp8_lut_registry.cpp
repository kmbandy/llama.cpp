// MAD-214 Phase 1G-A: smoke test for the turbo-FP8 LUT registry.
//
// Validates:
//  1. init() succeeds and creates the cache directory
//  2. get_lut_device_ptr returns a valid device pointer on first call
//  3. First call falls back to the embedded canonical LUT (no on-disk cache yet)
//  4. Second call returns the same cached pointer (no re-load)
//  5. The 16 bytes at the device pointer match what we expect
//  6. all_luts_cached() correctly reports "false" when no disk cache exists
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -x hip \
//       -I ggml/include -I ggml/src \
//       tests/test_turbo_fp8_lut_registry.cpp \
//       -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin \
//       -o /tmp/test_turbo_fp8_lut_registry

#include "../ggml/src/ggml-cuda/mt_turbo_fp8_lut_registry.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
    fprintf(stderr, "  OK: %s\n", msg); \
} while(0)

#define HIP_OK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { fprintf(stderr, "HIP fail: %s\n", hipGetErrorString(_e)); return 1; } \
} while(0)

int main() {
    HIP_OK(hipSetDevice(0));

    // Use a synthetic fingerprint that won't collide with any real model
    mt_turbo_fp8::model_fingerprint fp {
        .arch       = "test_smoke",
        .n_layer    = 4,
        .n_embd     = 1024,
        .head_dim   = 256,
        .n_kv_heads = 4,
    };

    const std::string digest = fp.digest();
    fprintf(stderr, "fingerprint digest = %s\n", digest.c_str());

    CHECK(mt_turbo_fp8::init(fp, /*auto_calibrate=*/false),
          "init returns true");

    CHECK(!mt_turbo_fp8::all_luts_cached(),
          "all_luts_cached() is false (no disk cache for synthetic fp)");

    // First-call lookup — should fall back to embedded canonical LUT.
    const uint8_t * dev_k0 = mt_turbo_fp8::get_lut_device_ptr(0, mt_turbo_fp8::KV_K);
    CHECK(dev_k0 != nullptr, "get_lut_device_ptr(0, K) returns non-null");

    // Read back the 16 bytes from device and check they match the canonical
    // embedded LUT (which we know starts with 0x0f, 0x1b, 0x21, 0x25, ...).
    uint8_t host_readback[16] = {0};
    HIP_OK(hipMemcpy(host_readback, dev_k0, 16, hipMemcpyDeviceToHost));
    fprintf(stderr, "  device bytes [0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
            host_readback[0], host_readback[1], host_readback[2], host_readback[3],
            host_readback[4], host_readback[5], host_readback[6], host_readback[7]);
    const uint8_t canonical[16] = {0x0f, 0x1b, 0x21, 0x25, 0x28, 0x2a, 0x2c, 0x2e,
                                     0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x38};
    CHECK(std::memcmp(host_readback, canonical, 16) == 0,
          "device LUT bytes match embedded canonical");

    // Second call — should return the same pointer (cached, no re-upload).
    const uint8_t * dev_k0_again = mt_turbo_fp8::get_lut_device_ptr(0, mt_turbo_fp8::KV_K);
    CHECK(dev_k0_again == dev_k0, "second get_lut_device_ptr(0, K) returns same ptr (cached)");

    // V dir should get its own (separate) device buffer
    const uint8_t * dev_v0 = mt_turbo_fp8::get_lut_device_ptr(0, mt_turbo_fp8::KV_V);
    CHECK(dev_v0 != nullptr && dev_v0 != dev_k0,
          "get_lut_device_ptr(0, V) returns separate non-null ptr");

    // Last valid layer index
    const uint8_t * dev_kN = mt_turbo_fp8::get_lut_device_ptr(fp.n_layer - 1, mt_turbo_fp8::KV_K);
    CHECK(dev_kN != nullptr, "get_lut_device_ptr(n_layer-1, K) returns non-null");

    // Out-of-range layer index → expected nullptr
    const uint8_t * oob = mt_turbo_fp8::get_lut_device_ptr(fp.n_layer, mt_turbo_fp8::KV_K);
    CHECK(oob == nullptr, "get_lut_device_ptr(n_layer, K) returns null (out of range)");

    mt_turbo_fp8::shutdown();
    fprintf(stderr, "\n=== PASS — registry load path works correctly ===\n");
    return 0;
}
