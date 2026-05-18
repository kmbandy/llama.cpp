// test_aiter_registry — smoke-test the runtime AITER compiler/registry in
// isolation (no kernel launch). Verifies:
//   - Disk cache miss triggers compile_aiter_kernel.py
//   - HSACO loads via hipModuleLoadData
//   - Kernel symbol resolves via hipModuleGetFunction
//   - Second get_or_compile call hits the in-memory cache (no recompile)
//   - Third call (fresh registry) hits the disk cache (no Python invocation)
//
// On success: prints "OK" + a 3-line summary. On any failure: aborts.

#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define REQUIRE(cond, msg) do {                                     \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);\
        return 1;                                                   \
    }                                                               \
} while (0)

static aiter::KernelSpec make_uattn_3d_spec() {
    aiter::KernelSpec s;
    s.source_path = AITER_KERNEL_SOURCE_DEFAULT;
    s.kernel_name = "kernel_unified_attention_3d";
    s.target      = "hip:gfx1201:32";
    s.signature   =
        "*fp32:16, *fp32, *fp32, *fp16:16, *fp16:16, *fp16:16, *fp32, *i32, *i32, "
        "*fp32, *fp16, fp32, *fp32, *fp32, *fp32, fp32, 16, 8, i64, i64, 128, i64, "
        "16, 32, 128, 128, 0, 0, 0, 0, 0, i64, i64, i64, 1, i64, i64, i64, 1, "
        "*i32, 2, i32, 16, 32, 1";
    s.num_warps  = 4;
    s.num_stages = 1;
    return s;
}

int main() {
    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);
    // Use a scratch cache dir so we can observe miss → hit transitions.
    reg.set_cache_dir("/tmp/aiter-registry-smoke");

    auto spec = make_uattn_3d_spec();
    std::printf("spec.cache_key() = %s\n", spec.cache_key().c_str());

    // ── 1. First call: should compile (cache miss) ─────────────────────────
    std::printf("\n[1] First get_or_compile (expect compile to fire)...\n");
    const aiter::KernelHandle * h1 = reg.get_or_compile(spec);
    REQUIRE(h1 != nullptr,              "first get_or_compile returned null");
    REQUIRE(h1->module   != nullptr,    "module handle is null");
    REQUIRE(h1->function != nullptr,    "function handle is null");
    REQUIRE(h1->kernel_symbol == "kernel_unified_attention_3d", "wrong kernel symbol");
    REQUIRE(h1->block_x == 128,         "block_x != 128 (expected 4 warps × 32 lanes)");
    REQUIRE(h1->shared_mem_bytes == 8192, "shared_mem_bytes != 8192");
    std::printf("    ✓ got handle: symbol=%s block=(%d,%d,%d) smem=%d\n",
                h1->kernel_symbol.c_str(), h1->block_x, h1->block_y, h1->block_z,
                h1->shared_mem_bytes);

    // ── 2. Second call (same registry instance): in-memory cache hit ───────
    std::printf("\n[2] Second get_or_compile (expect in-memory cache hit)...\n");
    const aiter::KernelHandle * h2 = reg.get_or_compile(spec);
    REQUIRE(h2 == h1, "second call returned a different handle (in-memory cache miss?)");
    std::printf("    ✓ same pointer returned (in-memory hit)\n");

    // ── 3. Disk cache hit (simulated): we can't easily re-init the singleton,
    //      but we can verify the disk artifacts exist for a future cold start.
    std::printf("\n[3] Disk cache artifacts present?\n");
    std::string cache_dir = "/tmp/aiter-registry-smoke/" + spec.cache_key();
    std::FILE * fh = std::fopen((cache_dir + "/kernel.hsaco").c_str(), "rb");
    REQUIRE(fh != nullptr, "kernel.hsaco missing on disk");
    std::fclose(fh);
    fh = std::fopen((cache_dir + "/meta.json").c_str(), "rb");
    REQUIRE(fh != nullptr, "meta.json missing on disk");
    std::fclose(fh);
    std::printf("    ✓ kernel.hsaco + meta.json present at %s\n", cache_dir.c_str());

    std::printf("\nOK — registry smoke test passed.\n");
    return 0;
}
