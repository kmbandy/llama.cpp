// Regression test for the router's VRAM-estimate cache key
// (server_models::estimate_need_bytes_key(), tools/server/server-models.cpp).
//
// Bug: apply_to_params() applied LLAMA_ARG_KV_TIERED, LLAMA_ARG_CTX_CHECKPOINTS
// and LLAMA_ARG_CACHE_RAM to the params used for measurement, but the cache
// key string did not include any of them. Two presets that differed only in
// one of those options collided on the same cache key/file, so the router
// silently reused a stale VRAM estimate across incompatible presets (the
// tiel-35b-par3 incident: physical_free dropped to 2.17 GiB on a 32 GB card).
//
// This test builds presets via the same common_preset_context::load_from_args()
// path the router uses, and checks that presets differing only in kv-tiered,
// or only in cache-ram, produce different keys, and that the key changes
// when the "v2|" version prefix changes (i.e. it's actually present).

#include "preset.h"
#include "server-models.h"

#include <cstring>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static common_preset make_preset(const std::vector<std::string> & extra_args) {
    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);

    std::vector<std::string> args = { "llama-server", "-m", "test-model.gguf" };
    args.insert(args.end(), extra_args.begin(), extra_args.end());

    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }

    return ctx.load_from_args((int) argv.size(), argv.data());
}

static server_model_meta make_meta(const std::vector<std::string> & extra_args) {
    server_model_meta meta;
    meta.name = "test";
    meta.preset = make_preset(extra_args);
    return meta;
}

int main() {
    // Baseline preset.
    const server_model_meta base = make_meta({});
    const std::string base_key = server_models::estimate_need_bytes_key(base);

    // Differs only in --kv-tiered: must produce a different key.
    const server_model_meta kv_tiered = make_meta({ "--kv-tiered", "25,25,50" });
    const std::string kv_tiered_key = server_models::estimate_need_bytes_key(kv_tiered);
    assert(kv_tiered_key != base_key && "kv-tiered must affect the estimate cache key");

    // Differs only in --cache-ram: must produce a different key.
    const server_model_meta cache_ram = make_meta({ "--cache-ram", "512" });
    const std::string cache_ram_key = server_models::estimate_need_bytes_key(cache_ram);
    assert(cache_ram_key != base_key && "cache-ram must affect the estimate cache key");

    // Differs only in --ctx-checkpoints: must produce a different key.
    const server_model_meta ctx_cp = make_meta({ "--ctx-checkpoints", "4" });
    const std::string ctx_cp_key = server_models::estimate_need_bytes_key(ctx_cp);
    assert(ctx_cp_key != base_key && "ctx-checkpoints must affect the estimate cache key");

    // All three keys must be pairwise distinct too.
    assert(kv_tiered_key != cache_ram_key);
    assert(kv_tiered_key != ctx_cp_key);
    assert(cache_ram_key != ctx_cp_key);

    // The version prefix must actually be present in the key, so bumping it
    // in the source invalidates every previously-cached estimate on disk.
    assert(base_key.rfind("v2|", 0) == 0 && "estimate key must start with the version prefix");

    // Two presets built identically must produce identical keys (sanity: the
    // key must be deterministic, not e.g. pointer- or time-of-day-derived).
    const server_model_meta base2 = make_meta({});
    assert(server_models::estimate_need_bytes_key(base2) == base_key);

    return 0;
}
