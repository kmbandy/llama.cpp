// Metadata-only load of the real DeepSeek-V4.1 spine GGUF plus the Engram
// sidecar GGUF. No weight bytes are read (no_alloc), so this is a shape/KV
// check against the converted files, not inference. Skips (exit 0) unless
// DSV41_SPINE points at the spine GGUF; DSV41_ENGRAM may add the sidecar.
#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"

#include "../src/models/models.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main() {
    const char * spine  = std::getenv("DSV41_SPINE");
    const char * engram = std::getenv("DSV41_ENGRAM");
    if (spine == nullptr || spine[0] == '\0') {
        printf("DSV41_SPINE not set, skipping real spine load\n");
        return 0;
    }
    ggml_backend_load_all();

    std::vector<const char *> sidecars;
    if (engram != nullptr && engram[0] != '\0') {
        sidecars.push_back(engram);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.no_alloc     = true;
    mparams.load_mode    = LLAMA_LOAD_MODE_NONE;
    mparams.load_mtp     = true;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, nullptr };
    mparams.devices = devices;
    mparams.sidecar_files   = sidecars.empty() ? nullptr : sidecars.data();
    mparams.n_sidecar_files = sidecars.size();

    llama_model_ptr model(llama_model_load_from_file(spine, mparams));
    if (!model) {
        fprintf(stderr, "failed to load %s\n", spine);
        return 1;
    }
    if (model->arch != LLM_ARCH_DEEPSEEK41) {
        fprintf(stderr, "arch mismatch: %s\n", llm_arch_name(model->arch));
        return 1;
    }
    if (!model->routed_experts_external) {
        fprintf(stderr, "spine must carry weight_pager.routed_experts_external\n");
        return 1;
    }
    const auto & hp = model->hparams;
    if (hp.n_layer() != 40 || hp.n_layer_all != 43 || hp.n_expert != 384) {
        fprintf(stderr, "hparams: n_layer=%u n_layer_all=%u n_expert=%u\n", hp.n_layer(), hp.n_layer_all, hp.n_expert);
        return 1;
    }
    for (uint32_t il = hp.n_layer(); il < hp.n_layer_all; ++il) {
        const auto & layer = model->layers[il];
        if (layer.ffn_gate_inp == nullptr || layer.ffn_gate_inp->ne[1] != 128) {
            fprintf(stderr, "MTP layer %u router should have 128 experts\n", il);
            return 1;
        }
    }
    if (!sidecars.empty()) {
        for (uint32_t i = 0; i < hp.dsv41_n_engram_layers; ++i) {
            const auto & layer = model->layers[hp.dsv41_engram_layer_ids[i]];
            if (layer.engram_embd == nullptr || layer.engram_wkv == nullptr) {
                fprintf(stderr, "Engram layer %u tensors missing from sidecar\n", hp.dsv41_engram_layer_ids[i]);
                return 1;
            }
        }
    }
    if (model->fc == nullptr) {
        fprintf(stderr, "DSpark head (fc.weight) not loaded\n");
        return 1;
    }

    // Engram hasher vs DeepSeek inference/engram.py. DSV41_ENGRAM_REF is the
    // file written by scratchpad gen_engram_ref.py: header "n_tok n_layers
    // n_cols pad_id", the token ids, then one line of n_cols ids per
    // (position, layer). Bit-exact match required.
    if (const char * ref_path = std::getenv("DSV41_ENGRAM_REF")) {
        const auto & hasher = static_cast<const llama_model_deepseek41 *>(model.get())->engram;
        if (!hasher.ready()) {
            fprintf(stderr, "engram hasher not ready (token_map/multipliers missing from GGUF)\n");
            return 1;
        }
        std::ifstream in(ref_path);
        int n_tok = 0, n_layers = 0, n_cols = 0, pad_id = 0;
        in >> n_tok >> n_layers >> n_cols >> pad_id;
        if (!in || n_layers != (int) hasher.n_layers || n_cols != (int) hasher.n_cols() || pad_id != hasher.pad_id) {
            fprintf(stderr, "engram ref header mismatch: layers %d/%u cols %d/%u pad %d/%d\n",
                    n_layers, hasher.n_layers, n_cols, hasher.n_cols(), pad_id, hasher.pad_id);
            return 1;
        }
        std::vector<int32_t> hist(n_tok);
        for (int i = 0; i < n_tok; ++i) {
            int32_t tok = 0;
            in >> tok;
            hist[i] = hasher.compress(tok);
        }
        std::vector<int32_t> row(n_cols);
        std::vector<int32_t> ctx_ids(hasher.max_ngram);
        size_t mismatches = 0;
        for (int pos = 0; pos < n_tok; ++pos) {
            // hash_position now takes the already-resolved n-gram context (index 0 = current
            // token, index s = s positions back, pad substituted for out-of-range lookback)
            // instead of a hist+pos pair; build it the same way the old signature did.
            for (uint32_t s = 0; s < hasher.max_ngram; ++s) {
                const int32_t src_pos = pos - (int32_t) s;
                ctx_ids[s] = (src_pos >= 0 && src_pos < n_tok) ? hist[src_pos] : hasher.pad_id;
            }
            for (int layer = 0; layer < n_layers; ++layer) {
                hasher.hash_position(ctx_ids.data(), (uint32_t) layer, row.data());
                for (int c = 0; c < n_cols; ++c) {
                    long long want = 0;
                    in >> want;
                    if ((long long) row[c] != want) {
                        if (mismatches < 8) {
                            fprintf(stderr, "engram hash mismatch pos=%d layer=%d col=%d got=%d want=%lld\n",
                                    pos, layer, c, row[c], want);
                        }
                        ++mismatches;
                    }
                }
            }
        }
        if (!in || mismatches != 0) {
            fprintf(stderr, "engram hasher does not reproduce the reference (%zu mismatches)\n", mismatches);
            return 1;
        }
        printf("Engram hasher matches inference/engram.py: %d positions x %d layers x %d cols\n", n_tok, n_layers, n_cols);
    }
    printf("DeepSeek-V4.1 spine metadata load ok: files=%zu n_layer_all=%u engram_layers=%u\n",
           1 + sidecars.size(), hp.n_layer_all, hp.dsv41_n_engram_layers);
    return 0;
}
