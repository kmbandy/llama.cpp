// MAD-214 Phase 1G-A: turbo-FP8 LUT registry implementation.
//
// Load path only — calibration warmup (Phase 1G-E) hooks into a separate
// API once the capture pipeline (1G-C) and Lloyd-Max fitter (1G-D) land.
// Until then, missing LUTs fall back to the embedded qwen3.5-4B header.

#include "mt_turbo_fp8_lut_registry.h"
#include "aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace mt_turbo_fp8 {

namespace {

constexpr size_t LUT_BYTES = 16;  // 16 E4M3 centroids per LUT

// ── FNV-1a 64 + SHA-style hex truncation. Fast, no extra deps. The
//    digest only has to be globally unique enough to avoid cache collisions
//    between users' models; not cryptographically meaningful.
uint64_t fnv1a64(const std::string & s) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

std::string env_or(const char * name, const std::string & fallback) {
    const char * v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

std::string cache_root() {
    std::string xdg = env_or("XDG_CACHE_HOME", "");
    if (xdg.empty()) xdg = env_or("HOME", "/tmp") + "/.cache";
    return xdg + "/llama.cpp/turbo-fp8";
}

void mkdir_p(const std::string & path) {
    // Walk + mkdir each component. Tolerate EEXIST.
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur.push_back(path[i]);
        if (path[i] == '/' || i + 1 == path.size()) {
            if (!cur.empty() && cur != "/") {
                ::mkdir(cur.c_str(), 0755);
            }
        }
    }
}

bool file_exists(const std::string & p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Pull `n` bytes from a file into `out`. Returns false on short read or
// I/O failure. Caller guarantees `out` has space for `n` bytes.
bool read_file_bytes(const std::string & p, uint8_t * out, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char *>(out), (std::streamsize) n);
    return f.gcount() == (std::streamsize) n;
}

// ── Registry singleton state. Wrapped in an accessor so static-init order
//    doesn't bite.
struct registry_state {
    std::mutex                              mu;
    bool                                    initialized = false;
    bool                                    auto_calibrate = false;
    model_fingerprint                       fp {};
    std::string                             cache_dir;
    int                                     n_attn_layers = 0;
    // Per-(layer, k|v) device pointer. Filled lazily by get_lut_device_ptr.
    std::vector<std::pair<uint8_t *, uint8_t *>> dev_luts;   // (k_ptr, v_ptr)
    bool                                    warned_fallback = false;
};

registry_state & state() {
    static registry_state s;
    return s;
}

// ── Embedded-header fallback. The Qwen3.5-4B header has L3, L7, ..., L31
//    (the 8 attention layers in its 32-layer hybrid arch; DeltaNet for the
//    rest). The fitted centroids are nearly identical across layers (14 of
//    16 are byte-for-byte the same), so for stop-gap fallback purposes we
//    return ONE canonical LUT regardless of layer/direction. Phase 1G-E
//    auto-calibration replaces this with real per-(kv, layer) values.
const uint8_t * embedded_fallback_canonical() {
    return mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L15;  // any well-formed LUT works
}

// Resolve the 16-byte LUT for (layer, dir): prefer on-disk cache, fall
// back to embedded header. Writes into `out` (16 bytes). Returns false
// only if BOTH cache and fallback are unavailable.
bool resolve_lut_bytes(int layer, kv_dir dir, uint8_t * out) {
    auto & s = state();
    // 1. On-disk per-model cache
    std::ostringstream path;
    path << s.cache_dir << "/l" << layer << "_" << (dir == KV_K ? "k" : "v") << ".bin";
    if (file_exists(path.str()) && read_file_bytes(path.str(), out, LUT_BYTES)) {
        return true;
    }
    // 2. Embedded canonical fallback — log a one-time warning per process.
    //    Same LUT for any (layer, dir) until Phase 1G-E auto-calibration
    //    lands. This is correct enough to validate the integration pipeline
    //    (registry → scatter → dispatch → attention) end-to-end, but quality
    //    numbers will not reflect what real per-(kv, layer) calibration
    //    delivers.
    (void) layer; (void) dir;
    std::memcpy(out, embedded_fallback_canonical(), LUT_BYTES);
    if (!s.warned_fallback) {
        std::fprintf(stderr,
            "mt_turbo_fp8: WARNING — no calibrated LUTs found for fingerprint %s, "
            "falling back to a single embedded canonical LUT for all (layer, dir). "
            "Quality numbers will not be representative until Phase 1G-E "
            "auto-calibration lands.\n",
            s.fp.digest().c_str());
        s.warned_fallback = true;
    }
    return true;
}

}  // namespace

std::string model_fingerprint::digest() const {
    std::ostringstream key;
    key << arch << "|nL=" << n_layer << "|nE=" << n_embd
        << "|hD=" << head_dim << "|nKV=" << n_kv_heads;
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016lx", (unsigned long) fnv1a64(key.str()));
    return std::string(buf);
}

bool init(const model_fingerprint & fp, bool auto_calibrate_if_missing) {
    auto & s = state();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.initialized) {
        if (s.fp.digest() != fp.digest()) {
            std::fprintf(stderr,
                "mt_turbo_fp8: ERROR — registry already initialized for %s, can't "
                "switch to %s mid-process\n", s.fp.digest().c_str(), fp.digest().c_str());
            return false;
        }
        return true;
    }
    s.fp             = fp;
    s.auto_calibrate = auto_calibrate_if_missing;
    s.cache_dir      = cache_root() + "/" + fp.digest();
    s.n_attn_layers  = fp.n_layer;
    s.dev_luts.assign(s.n_attn_layers, { nullptr, nullptr });
    mkdir_p(s.cache_dir);
    // Write a manifest for human inspection (not consumed by this code).
    {
        std::ofstream m(s.cache_dir + "/manifest.json");
        if (m) {
            m << "{\n"
              << "  \"arch\": \""        << fp.arch       << "\",\n"
              << "  \"n_layer\": "       << fp.n_layer    << ",\n"
              << "  \"n_embd\": "        << fp.n_embd     << ",\n"
              << "  \"head_dim\": "      << fp.head_dim   << ",\n"
              << "  \"n_kv_heads\": "    << fp.n_kv_heads << ",\n"
              << "  \"fingerprint\": \"" << fp.digest()   << "\"\n"
              << "}\n";
        }
    }
    s.initialized = true;
    std::fprintf(stderr,
        "mt_turbo_fp8: registry init — arch=%s n_layer=%d head_dim=%d n_kv_heads=%d "
        "cache=%s auto_calibrate=%d\n",
        fp.arch.c_str(), fp.n_layer, fp.head_dim, fp.n_kv_heads,
        s.cache_dir.c_str(), (int) auto_calibrate_if_missing);
    return true;
}

const uint8_t * get_lut_device_ptr(int layer, kv_dir dir) {
    auto & s = state();
    std::lock_guard<std::mutex> g(s.mu);
    if (!s.initialized) {
        std::fprintf(stderr, "mt_turbo_fp8: get_lut_device_ptr called before init\n");
        return nullptr;
    }
    if (layer < 0 || layer >= (int) s.dev_luts.size()) {
        std::fprintf(stderr, "mt_turbo_fp8: layer %d out of range [0, %zu)\n",
                     layer, s.dev_luts.size());
        return nullptr;
    }
    uint8_t *& slot = (dir == KV_K) ? s.dev_luts[layer].first : s.dev_luts[layer].second;
    if (slot != nullptr) return slot;

    uint8_t host_lut[LUT_BYTES];
    if (!resolve_lut_bytes(layer, dir, host_lut)) return nullptr;

    uint8_t * dev = nullptr;
    if (hipMalloc(&dev, LUT_BYTES) != hipSuccess) {
        std::fprintf(stderr, "mt_turbo_fp8: hipMalloc(%zu) failed\n", LUT_BYTES);
        return nullptr;
    }
    if (hipMemcpy(dev, host_lut, LUT_BYTES, hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "mt_turbo_fp8: hipMemcpy upload failed\n");
        hipFree(dev);
        return nullptr;
    }
    slot = dev;
    return dev;
}

bool all_luts_cached() {
    auto & s = state();
    std::lock_guard<std::mutex> g(s.mu);
    if (!s.initialized) return false;
    for (int l = 0; l < s.n_attn_layers; ++l) {
        for (auto d : { KV_K, KV_V }) {
            std::ostringstream p;
            p << s.cache_dir << "/l" << l << "_" << (d == KV_K ? "k" : "v") << ".bin";
            if (!file_exists(p.str())) return false;
        }
    }
    return true;
}

void shutdown() {
    auto & s = state();
    std::lock_guard<std::mutex> g(s.mu);
    for (auto & p : s.dev_luts) {
        if (p.first)  hipFree(p.first);
        if (p.second) hipFree(p.second);
    }
    s.dev_luts.clear();
    s.initialized = false;
}

}  // namespace mt_turbo_fp8
