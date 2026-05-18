// aiter_runtime_compiler.cpp — see header for design notes.

#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <sys/stat.h>

namespace aiter {

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

static uint64_t fnv1a64(const std::string & s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string to_hex16(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016lx", (unsigned long)h);
    return buf;
}

// Replace characters that would break filesystem paths (`:`, `/`, ` `, etc.)
// with `-`. Keeps the slug short and human-readable.
static std::string slug(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('-');
        }
    }
    return out;
}

static std::string env_or(const char * name, const std::string & fallback) {
    const char * v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

// Returns mtime in seconds since epoch, or 0 if stat() fails.
static int64_t file_mtime(const std::string & path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return 0;
    return (int64_t) st.st_mtime;
}

// Read whole file into a string. Returns empty on failure (caller checks).
static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<unsigned char> read_file_bytes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> out(n > 0 ? (size_t) n : 0);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

// Pull one int field from JSON text. Tolerates whitespace; returns `dflt` if
// the key is missing.
static long long json_get_int(const std::string & text, const std::string & key, long long dflt) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return std::stoll(m[1].str());
    }
    return dflt;
}

static std::string json_get_str(const std::string & text, const std::string & key, const std::string & dflt) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        return m[1].str();
    }
    return dflt;
}

// ─────────────────────────────────────────────────────────────────────────
// KernelSpec
// ─────────────────────────────────────────────────────────────────────────

std::string KernelSpec::cache_key() const {
    // Combine all spec-determining inputs + the source file's mtime into one
    // hash. mtime ensures the cache invalidates when the .py source changes.
    std::string composite = source_path + "|" + kernel_name + "|" + target + "|" + signature
                          + "|W" + std::to_string(num_warps)
                          + "|S" + std::to_string(num_stages)
                          + "|M" + std::to_string(file_mtime(source_path));
    return slug(kernel_name) + "__" + slug(target)
         + "__W" + std::to_string(num_warps) + "S" + std::to_string(num_stages)
         + "__" + to_hex16(fnv1a64(composite));
}

// ─────────────────────────────────────────────────────────────────────────
// KernelHandle::launch
// ─────────────────────────────────────────────────────────────────────────

hipError_t KernelHandle::launch(hipStream_t stream,
                                 unsigned int gX, unsigned int gY, unsigned int gZ,
                                 void ** kernel_args) const {
    return hipModuleLaunchKernel(function,
                                  gX, gY, gZ,
                                  (unsigned int) block_x,
                                  (unsigned int) block_y,
                                  (unsigned int) block_z,
                                  (unsigned int) shared_mem_bytes,
                                  stream,
                                  kernel_args,
                                  /*extra=*/nullptr);
}

// ─────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────

Registry & Registry::instance() {
    static Registry r;
    return r;
}

Registry::Registry() {
    python_         = env_or("AITER_PYTHON", "python3");
    compile_script_ = env_or("AITER_COMPILE_SCRIPT", "");
    cache_dir_      = env_or("AITER_CACHE_DIR", "");
    if (cache_dir_.empty()) {
        std::string xdg = env_or("XDG_CACHE_HOME", "");
        if (xdg.empty()) {
            std::string home = env_or("HOME", "/tmp");
            xdg = home + "/.cache";
        }
        cache_dir_ = xdg + "/llama.cpp/aiter";
    }
}

const KernelHandle * Registry::get_or_compile(const KernelSpec & spec) {
    std::lock_guard<std::mutex> guard(mu_);

    const std::string key = spec.cache_key();

    // 1. In-memory hit?
    auto it = handles_.find(key);
    if (it != handles_.end()) {
        return it->second.get();
    }

    // 2. Ensure disk artifacts exist (compile if missing).
    if (!ensure_on_disk(key, spec)) {
        std::fprintf(stderr, "aiter::Registry: compile failed for spec %s (kernel=%s, target=%s)\n",
                     key.c_str(), spec.kernel_name.c_str(), spec.target.c_str());
        return nullptr;
    }

    // 3. Load HSACO into a hipModule + resolve the kernel symbol.
    auto handle = load_into_memory(key);
    if (!handle) {
        std::fprintf(stderr, "aiter::Registry: load failed for spec %s\n", key.c_str());
        return nullptr;
    }

    KernelHandle * raw = handle.get();
    handles_.emplace(key, std::move(handle));
    return raw;
}

bool Registry::ensure_on_disk(const std::string & cache_key, const KernelSpec & spec) {
    const fs::path artifact_dir = fs::path(cache_dir_) / cache_key;
    const fs::path hsaco_path   = artifact_dir / "kernel.hsaco";
    const fs::path meta_path    = artifact_dir / "meta.json";

    if (fs::exists(hsaco_path) && fs::exists(meta_path)) {
        return true;
    }

    if (compile_script_.empty()) {
        std::fprintf(stderr, "aiter::Registry: AITER_COMPILE_SCRIPT not set and no default — "
                              "cannot compile spec %s on miss\n", cache_key.c_str());
        return false;
    }

    std::error_code ec;
    fs::create_directories(artifact_dir, ec);
    if (ec) {
        std::fprintf(stderr, "aiter::Registry: mkdir failed for %s: %s\n",
                     artifact_dir.c_str(), ec.message().c_str());
        return false;
    }

    // Build the command. Quote every argument that may contain spaces; the
    // signature in particular has commas and spaces.
    auto shellq = [](const std::string & s) {
        std::string out = "'";
        for (char c : s) {
            if (c == '\'') out += "'\\''";
            else           out += c;
        }
        out += "'";
        return out;
    };

    std::ostringstream cmd;
    cmd << shellq(python_)
        << " "  << shellq(compile_script_)
        << " --source "      << shellq(spec.source_path)
        << " --kernel-name " << shellq(spec.kernel_name)
        << " --target "      << shellq(spec.target)
        << " --signature "   << shellq(spec.signature)
        << " --num-warps "   << spec.num_warps
        << " --num-stages "  << spec.num_stages
        << " --out-dir "     << shellq(artifact_dir.string())
        << " >&2";  // route stdout to stderr (the helper logs progress on stderr)

    const std::string cmd_str = cmd.str();
    std::fprintf(stderr, "aiter::Registry: compiling %s (%s)\n",
                 spec.kernel_name.c_str(), cache_key.c_str());
    int rc = std::system(cmd_str.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "aiter::Registry: compile script returned %d\n", rc);
        return false;
    }

    if (!fs::exists(hsaco_path) || !fs::exists(meta_path)) {
        std::fprintf(stderr, "aiter::Registry: compile reported success but artifacts missing in %s\n",
                     artifact_dir.c_str());
        return false;
    }
    return true;
}

std::unique_ptr<KernelHandle> Registry::load_into_memory(const std::string & cache_key) {
    const fs::path artifact_dir = fs::path(cache_dir_) / cache_key;
    const fs::path hsaco_path   = artifact_dir / "kernel.hsaco";
    const fs::path meta_path    = artifact_dir / "meta.json";

    const std::string meta_text = read_file(meta_path.string());
    if (meta_text.empty()) {
        std::fprintf(stderr, "aiter::Registry: meta.json empty/unreadable at %s\n", meta_path.c_str());
        return nullptr;
    }

    auto handle = std::make_unique<KernelHandle>();
    handle->kernel_symbol    = json_get_str(meta_text, "kernel_symbol", "");
    handle->block_x          = (int) json_get_int(meta_text, "block_x",          0);
    handle->block_y          = (int) json_get_int(meta_text, "block_y",          1);
    handle->block_z          = (int) json_get_int(meta_text, "block_z",          1);
    handle->shared_mem_bytes = (int) json_get_int(meta_text, "shared_mem_bytes", 0);
    handle->cache_key        = cache_key;

    if (handle->kernel_symbol.empty() || handle->block_x == 0) {
        std::fprintf(stderr, "aiter::Registry: meta.json missing required fields at %s\n", meta_path.c_str());
        return nullptr;
    }

    auto blob = read_file_bytes(hsaco_path.string());
    if (blob.empty()) {
        std::fprintf(stderr, "aiter::Registry: HSACO empty/unreadable at %s\n", hsaco_path.c_str());
        return nullptr;
    }

    hipError_t err = hipModuleLoadData(&handle->module, blob.data());
    if (err != hipSuccess) {
        std::fprintf(stderr, "aiter::Registry: hipModuleLoadData failed: %s (cache_key=%s)\n",
                     hipGetErrorString(err), cache_key.c_str());
        return nullptr;
    }

    err = hipModuleGetFunction(&handle->function, handle->module, handle->kernel_symbol.c_str());
    if (err != hipSuccess) {
        std::fprintf(stderr, "aiter::Registry: hipModuleGetFunction(%s) failed: %s\n",
                     handle->kernel_symbol.c_str(), hipGetErrorString(err));
        hipModuleUnload(handle->module);
        return nullptr;
    }

    return handle;
}

}  // namespace aiter
