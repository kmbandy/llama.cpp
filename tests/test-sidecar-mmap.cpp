// Unit check for the per-file mmap split added to llama_model_loader
// (LLAMA_MMAP_SIDECAR_ONLY): a "main" GGUF and a "sidecar" GGUF loaded
// together should, by default, both be mmapped (unchanged upstream
// behaviour); with LLAMA_MMAP_SIDECAR_ONLY=1 set before construction, the
// main file must load with no OS mapping at all (mappings[idx] == nullptr)
// while the sidecar keeps its mapping. CPU-only, no model weights needed
// beyond a couple of scalars -- this only exercises the loader's file
// bookkeeping, not graph building or inference.
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "../src/llama-model-loader.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__); \
            return 1; \
        } \
    } while (0)

// Write a minimal single-tensor GGUF file at `path`.
static void write_tiny_gguf(const std::string & path, const char * tensor_name) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() + 256,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, tensor_name);
    float * data = (float *) t->data;
    for (int i = 0; i < 4; ++i) {
        data[i] = (float) i;
    }

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_add_tensor(gguf_ctx, t);
    if (!gguf_write_to_file(gguf_ctx, path.c_str(), /*only_meta =*/ false)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        std::exit(1);
    }

    gguf_free(gguf_ctx);
    ggml_free(ctx);
}

// Construct a loader over `main_path` + sidecar `side_path` with the given
// LLAMA_MMAP_SIDECAR_ONLY env setting (nullptr = unset) and mmap requested.
static llama_model_loader make_loader(const std::string & main_path, const std::string & side_path, const char * env_val) {
    if (env_val) {
        setenv("LLAMA_MMAP_SIDECAR_ONLY", env_val, 1);
    } else {
        unsetenv("LLAMA_MMAP_SIDECAR_ONLY");
    }

    std::vector<std::string> splits;
    std::vector<std::string> sidecars = { side_path };

    return llama_model_loader(
        /*metadata =*/ nullptr,
        /*set_tensor_data =*/ nullptr,
        /*set_tensor_data_ud =*/ nullptr,
        /*fname =*/ main_path,
        /*splits =*/ splits,
        /*file =*/ nullptr,
        /*load_mode =*/ LLAMA_LOAD_MODE_MMAP,
        /*check_tensors =*/ false,
        /*no_alloc =*/ true,
        /*load_mtp =*/ false,
        /*param_overrides_p =*/ nullptr,
        /*param_tensor_buft_overrides_p =*/ nullptr,
        sidecars);
}

int main() {
    ggml_backend_load_all();

    const std::string main_file = "/tmp/test-sidecar-mmap-main-" + std::to_string(getpid()) + ".gguf";
    const std::string side_file = "/tmp/test-sidecar-mmap-side-" + std::to_string(getpid()) + ".gguf";

    write_tiny_gguf(main_file, "a");
    write_tiny_gguf(side_file, "b");

    // Default (LLAMA_MMAP_SIDECAR_ONLY unset): both files mapped, exactly
    // upstream behaviour -- this is the no-op / backward-compat guarantee.
    {
        llama_model_loader ml = make_loader(main_file, side_file, nullptr);
        TEST_ASSERT(ml.n_main_files == 1);
        TEST_ASSERT(ml.mmap_enabled_for_file(0) == true);
        TEST_ASSERT(ml.mmap_enabled_for_file(1) == true);

        ml.init_mappings(/*prefetch =*/ false, /*mlock_mmaps =*/ nullptr);
        TEST_ASSERT(ml.mappings.size() == 2);
        TEST_ASSERT(ml.mappings[0] != nullptr);
        TEST_ASSERT(ml.mappings[1] != nullptr);
    }

    // LLAMA_MMAP_SIDECAR_ONLY=1: main file gets no OS mapping, sidecar keeps
    // its mapping.
    {
        llama_model_loader ml = make_loader(main_file, side_file, "1");
        TEST_ASSERT(ml.n_main_files == 1);
        TEST_ASSERT(ml.mmap_enabled_for_file(0) == false);
        TEST_ASSERT(ml.mmap_enabled_for_file(1) == true);

        ml.init_mappings(/*prefetch =*/ false, /*mlock_mmaps =*/ nullptr);
        TEST_ASSERT(ml.mappings.size() == 2);
        TEST_ASSERT(ml.mappings[0] == nullptr);
        TEST_ASSERT(ml.mappings[1] != nullptr);

        // unmap_weight()/load_data_range() must treat the main file's weight
        // as not-mmapped (no dereference of a null mapping).
        const auto & w_main = ml.require_weight("a");
        TEST_ASSERT(w_main.idx == 0);
        ml.unmap_weight(w_main); // no-op; must not crash

        std::vector<uint8_t> buf(w_main.tensor ? ggml_nbytes(w_main.tensor) : 16);
        const void * data = ml.load_data_range(w_main, 0, buf.size(), buf.data());
        TEST_ASSERT(data == buf.data()); // read via file I/O, not mmap
    }

    // LLAMA_MMAP_SIDECAR_ONLY=0 behaves like unset (explicit off).
    {
        llama_model_loader ml = make_loader(main_file, side_file, "0");
        TEST_ASSERT(ml.mmap_enabled_for_file(0) == true);
        TEST_ASSERT(ml.mmap_enabled_for_file(1) == true);
    }

    unsetenv("LLAMA_MMAP_SIDECAR_ONLY");
    std::remove(main_file.c_str());
    std::remove(side_file.c_str());

    printf("test-sidecar-mmap: ok\n");
    return 0;
}
