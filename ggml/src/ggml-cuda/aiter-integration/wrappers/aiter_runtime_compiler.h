// aiter_runtime_compiler.h
//
// Runtime AITER kernel compiler + registry. Replaces the build-time AOT
// scaffolding for production use: given a kernel spec (signature, target,
// constexpr-baked values), returns a launchable hipFunction_t — compiling
// on cache miss via the compile_aiter_kernel.py helper script and caching
// the resulting HSACO blob on disk under
// `${AITER_CACHE_DIR:-${HOME}/.cache/llama.cpp/aiter}/<key>/`.
//
// Why this exists: vLLM-style build-time AOT (one .c file per shape compiled
// into the static library) explodes the build matrix and ties the binary to
// the model shape. Triton AOT itself is cheap (~2.5s on R9700). Doing it at
// server startup once per shape, with a disk cache for warm restarts, is
// strictly less friction with no runtime cost in the steady state.
//
// Threading: handles are stable for the lifetime of the registry. Concurrent
// get_or_compile() calls for distinct specs may run sequentially under the
// registry lock — compilation itself is rare (a handful of times per server
// lifetime once the cache is warm), so coarse locking is fine.
//
// MAD-188.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aiter {

// Identifies one Triton kernel specialization. The cache key is derived
// from all fields; changing any field forces a recompile.
struct KernelSpec {
    std::string source_path;   // absolute path to the .py source (mtime is part of key)
    std::string kernel_name;   // @triton.jit function name in source
    std::string target;        // e.g. "hip:gfx1201:32"
    std::string signature;     // Triton signature string
    int         num_warps  = 4;
    int         num_stages = 1;

    // Deterministic, human-readable-ish cache key. Includes a hash of the
    // signature so collisions are negligible without bloating directory names.
    std::string cache_key() const;
};

// Loaded, launch-ready kernel. Returned by AiterRegistry; the registry owns
// the lifetime — never delete one of these.
struct KernelHandle {
    hipModule_t    module           = nullptr;
    hipFunction_t  function         = nullptr;
    int            block_x          = 0;
    int            block_y          = 1;
    int            block_z          = 1;
    int            shared_mem_bytes = 0;
    std::string    kernel_symbol;          // symbol used in hipModuleGetFunction (for logging)
    std::string    cache_key;              // for diagnostics

    // Launch helper — caller computes the 3D grid from current shape and
    // packs args[] in the order the Triton signature expects (pointers as
    // hipDeviceptr_t*, fp32 scalars as float*, i64 as int64_t*, etc.).
    //
    // Important: pack fp32 scalars as `float`, NOT `double`. The build-time
    // AOT path had a bug where Triton emitted `double` in the C launcher; we
    // bypass that by packing args ourselves.
    hipError_t launch(hipStream_t stream,
                      unsigned int gX, unsigned int gY, unsigned int gZ,
                      void **kernel_args) const;
};

// Process-global registry. Configure via the env vars below before the first
// get_or_compile call; defaults are usable but only when the install layout
// is known.
//
//   AITER_PYTHON          — python interpreter to use (default: "python3")
//   AITER_COMPILE_SCRIPT  — path to compile_aiter_kernel.py (no default;
//                           must be set or registered explicitly)
//   AITER_CACHE_DIR       — on-disk artifact cache root (default:
//                           ${XDG_CACHE_HOME:-${HOME}/.cache}/llama.cpp/aiter)
class Registry {
public:
    static Registry & instance();

    // Explicit configuration — overrides env-var defaults. Safe to call
    // multiple times; later calls win. Settings are read once per
    // get_or_compile call, so re-configuration between calls takes effect.
    void set_python(std::string py)              { std::lock_guard<std::mutex> g(mu_); python_         = std::move(py);   }
    void set_compile_script(std::string path)    { std::lock_guard<std::mutex> g(mu_); compile_script_ = std::move(path); }
    void set_cache_dir(std::string path)         { std::lock_guard<std::mutex> g(mu_); cache_dir_      = std::move(path); }

    // Get a launchable kernel for `spec`, compiling on cache miss. Returns
    // nullptr on failure (errors printed to stderr). The returned handle is
    // owned by the registry; do not delete it.
    const KernelHandle * get_or_compile(const KernelSpec & spec);

private:
    Registry();
    Registry(const Registry &)             = delete;
    Registry & operator=(const Registry &) = delete;

    // Ensure the disk cache for `cache_key` contains kernel.hsaco + meta.json,
    // invoking the Python compile helper if missing. Returns true on success.
    bool ensure_on_disk(const std::string & cache_key, const KernelSpec & spec);

    // Read meta.json + kernel.hsaco from disk and produce a loaded handle.
    // Returns nullptr on failure.
    std::unique_ptr<KernelHandle> load_into_memory(const std::string & cache_key);

    std::string                                                 python_;
    std::string                                                 compile_script_;
    std::string                                                 cache_dir_;
    std::mutex                                                  mu_;
    std::unordered_map<std::string, std::unique_ptr<KernelHandle>> handles_;
};

}  // namespace aiter
