#pragma once

// Reader for wp-repack's expert-major blob set.
//
// wp-repack rewrites a MoE model's routed experts so that one expert's
// gate/up/down weights sit CONTIGUOUSLY in a blob, turning three scattered
// reads per routed expert into one sequential read. It emits, per shard, a
// `.wpb` blob plus a `.wpi.json` sidecar, and one `-manifest.json` describing
// the whole set.
//
// Those descriptors are JSON, and libllama deliberately carries no JSON
// dependency, so parsing happens here (common/ already vendors nlohmann, and
// it is the same library wp-repack wrote them with). The result is handed to
// libllama as flat arrays through llama_model_params; see llama_wp_blob_entry.
//
// The returned object OWNS the storage the llama_wp_blob_entry::name pointers
// reference. It must outlive the model load that consumes it.

#include "llama.h"

#include <string>
#include <vector>

struct common_wp_blob_index {
    // Blob paths, indexed by llama_wp_blob_entry::blob_idx.
    std::vector<std::string>  blob_files;
    // C-string views of blob_files, in the same order (what llama.h wants).
    std::vector<const char *> blob_file_ptrs;

    // Backing storage for the entry names. Sized exactly once so the
    // const char * in `entries` stay valid.
    std::vector<std::string>          entry_names;
    std::vector<llama_wp_blob_entry>  entries;

    // `entries[i].name` and `blob_file_ptrs[i]` point into the string vectors
    // above. Moving is safe -- a vector move relocates neither its elements
    // nor their heap buffers -- but copying would leave every pointer aimed at
    // the original, so copying is forbidden rather than left as a trap.
    common_wp_blob_index()                                         = default;
    common_wp_blob_index(const common_wp_blob_index &)             = delete;
    common_wp_blob_index & operator=(const common_wp_blob_index &) = delete;
    common_wp_blob_index(common_wp_blob_index &&)                  = default;
    common_wp_blob_index & operator=(common_wp_blob_index &&)      = default;

    bool empty() const { return entries.empty(); }
};

// Load the blob set described by `manifest_path`.
//
// `model_path` is the model this set must belong to; the manifest records the
// model it was built from and a mismatch is rejected rather than silently
// reading another model's weights. Blob file sizes are checked against the
// manifest as a cheap corruption/truncation guard (the recorded sha256 is not
// verified -- that would mean hashing hundreds of GB at every load; use
// `wp-repack --verify` for that).
//
// Throws std::runtime_error with a diagnostic message on any problem.
common_wp_blob_index common_wp_blob_index_load(const std::string & manifest_path,
                                               const std::string & model_path);
