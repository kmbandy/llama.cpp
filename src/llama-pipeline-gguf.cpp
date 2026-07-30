// GGUF-aware part of the pipeline band helpers.
//
// Kept out of llama-pipeline.cpp on purpose: that file is pure arithmetic over
// band bounds and tensor names with no ggml dependency, which is what lets
// test-pipeline-band compile it standalone. Only this one function needs to
// open a file, so only this one function pays for gguf.

#include "llama-pipeline.h"

#include "gguf.h"

bool llama_pipeline_peek_band_from_file(const char * path, int32_t * first, int32_t * last) {
    if (path == nullptr || first == nullptr || last == nullptr) {
        return false;
    }

    // Metadata only -- no tensor data is read or allocated.
    gguf_init_params gp = {};
    gp.no_alloc = true;
    gp.ctx      = nullptr;

    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (gguf == nullptr) {
        // Not readable as a GGUF. Not this function's job to complain: the
        // model loader will produce a far better message shortly.
        return false;
    }

    const int64_t k_first = gguf_find_key(gguf, "pipeline.layer_first");
    const int64_t k_last  = gguf_find_key(gguf, "pipeline.layer_last");

    bool ok = false;
    // Both or nothing. A file carrying exactly one is malformed; report "no
    // band" here and let the loader reject it with its specific diagnostic
    // rather than half-adopting a band from it.
    if (k_first >= 0 && k_last >= 0) {
        *first = gguf_get_val_i32(gguf, k_first);
        *last  = gguf_get_val_i32(gguf, k_last);
        ok     = true;
    }

    gguf_free(gguf);
    return ok;
}
