#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cstdlib>

#include "ggml-ml8.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");
    hparams.n_layer_kv_from_start = hparams.n_layer_all - hparams.n_layer_nextn;

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed (tied LM head)
    const bool output_tied = (output == NULL);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // Any weight type that can carry the ml8 calibration sidecars
    // (centroids/awq_scale/rotation_h_a/rotation_meta): ML8_4, ML8_FP8, and
    // (2026-09-17 FP8_B128 phase 2) FP8_B128. FP8_B128 reuses the exact same
    // sidecar names/meta layout as ML8_FP8 (see the design doc referenced
    // below) — the converter and the split-granularity regex in
    // llama-model.cpp already treat them identically.
    auto is_ml8_sidecar_weight = [](const struct ggml_tensor * w) {
        return w && (w->type == GGML_TYPE_ML8_4 || w->type == GGML_TYPE_ML8_FP8 || w->type == GGML_TYPE_FP8_B128);
    };

    // ml8-4 / ml8-fp8 / fp8_b128 sidecar loader (MAD-223; MAD-266 extended to
    // ML8_FP8; 2026-09-17 extended to FP8_B128). Reads the GGUF metadata for
    // the rotation factor h_a so we can declare its (a, a) shape without
    // baking the python factor_for_dim heuristic into C++. All sidecars are
    // TENSOR_NOT_REQUIRED: an ml8 weight without rotation/awq still loads
    // cleanly.
    //
    // centroids/awq_scale stay ML8_4-only (the ml8-4 LUT dequant path and its
    // AWQ pre-scale have no FP8 counterpart — see design note in
    // llama-ml8-registry.h). rotation_h_a/rotation_meta are created for any
    // of the three types: ML8_FP8 and FP8_B128 both need them for the
    // tensor-parallel K-split rotation (kronecker if rotation_h_a is
    // present, block_hadamard if only rotation_meta is present — h_a is
    // optional for block_hadamard since Q = I_a ⊗ H_b has no H_a leg, so
    // rotation_meta must be checked independently of rotation_h_a here).
    auto load_ml8_sidecars = [&](
            struct ggml_tensor * weight,
            llm_tensor tensor_id,
            int il_,
            int64_t k_dim,
            struct ggml_tensor ** out_centroids,
            struct ggml_tensor ** out_rotation_h_a,
            struct ggml_tensor ** out_rotation_meta,
            struct ggml_tensor ** out_awq_scale) {
        if (!is_ml8_sidecar_weight(weight)) {
            return;
        }
        if (weight->type == GGML_TYPE_ML8_4) {
            *out_centroids = create_tensor(tn(tensor_id, "centroids", il_),
                                           { 16, k_dim / 64 }, TENSOR_NOT_REQUIRED);
            *out_awq_scale = create_tensor(tn(tensor_id, "awq_scale", il_),
                                           { k_dim }, TENSOR_NOT_REQUIRED);
        }
        const auto * h_a_meta = ml.get_tensor_meta(tn(tensor_id, "rotation_h_a", il_).str().c_str());
        if (h_a_meta != nullptr) {
            const int64_t a = h_a_meta->ne[0];
            *out_rotation_h_a  = create_tensor(tn(tensor_id, "rotation_h_a",  il_),
                                               { a, a }, TENSOR_NOT_REQUIRED);
        }
        // block_hadamard weights carry rotation_meta with NO rotation_h_a, so
        // this is checked independently rather than nested under h_a_meta.
        const auto * meta_meta = ml.get_tensor_meta(tn(tensor_id, "rotation_meta", il_).str().c_str());
        if (meta_meta != nullptr) {
            *out_rotation_meta = create_tensor(tn(tensor_id, "rotation_meta", il_),
                                               { 4 }, TENSOR_NOT_REQUIRED);
        }
    };

    // Read the rotation_meta I32[4] = [a_dim, b_dim, in_features, kind_id]
    // sidecar's raw bytes directly from the GGUF file. This runs during
    // load_arch_tensors, which is called BEFORE llama_model_loader::
    // init_mappings() and load_all_data() — the sidecar tensor's backend
    // buffer isn't allocated or populated yet, so ggml_backend_tensor_get /
    // ml.load_data_range() aren't usable here. `ml.files` (opened for the
    // GGUF header/metadata scan) IS available this early, and every read
    // seeks first (matches llama_model_loader::load_all_data's own non-mmap
    // read path), so this is safe to call from anywhere in load_arch_tensors.
    // Returns false when the sidecar tensor isn't present in the GGUF.
    auto read_rotation_meta = [&](llm_tensor tensor_id, int il_, int32_t (&out)[4]) -> bool {
        const std::string name = tn(tensor_id, "rotation_meta", il_).str();
        const auto * w = ml.get_weight(name.c_str());
        if (w == nullptr) {
            return false;
        }
        GGML_ASSERT(w->idx < ml.files.size());
        ml.files.at(w->idx)->seek(w->offs, SEEK_SET);
        ml.files.at(w->idx)->read_raw(out, sizeof(out));
        return true;
    };

    // Fill in an ml8_sidecars' rotation_b_dim / rotation_block_hadamard from
    // rotation_meta, when present. `rotation_meta` is the tensor created by
    // load_ml8_sidecars above (non-null iff the GGUF has the sidecar).
    auto fill_rotation_meta = [&](ml8_sidecars & sc, struct ggml_tensor * rotation_meta,
                                  llm_tensor tensor_id, int il_) {
        if (rotation_meta == nullptr) {
            return;
        }
        int32_t meta[4];
        if (!read_rotation_meta(tensor_id, il_, meta)) {
            return;
        }
        const int32_t kind_id = meta[3];
        GGML_ASSERT((kind_id == GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER ||
                     kind_id == GGML_ML8_ROTATION_KIND_BLOCK_HADAMARD) &&
                    "rotation_meta: unrecognized kind_id");
        GGML_ASSERT((kind_id == GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER) == (sc.rotation_h_a != nullptr) &&
                    "rotation_meta kind_id / rotation_h_a presence mismatch");
        sc.rotation_b_dim          = meta[1];
        sc.rotation_block_hadamard = (kind_id == GGML_ML8_ROTATION_KIND_BLOCK_HADAMARD);
    };

    // ml8-4 / ml8-fp8 / fp8_b128 registry registration (MAD-223 T13; MAD-266
    // extended to ML8_FP8; 2026-09-17 extended to FP8_B128). For a target
    // weight of any of the three types, create its sidecars via
    // load_ml8_sidecars and register the (weight → sidecars) mapping so
    // build_lora_mm routes the base matmul through the ml8 helper. Sidecar
    // tensors are owned by the model's context (create_tensor tracks them
    // for loading); the registry only holds their pointers. Guarded on
    // ML8_4/ML8_FP8/FP8_B128: any other type registers nothing for these
    // roles → registry miss → plain mul_mat, unchanged from before.
    //
    // k_dim is the weight's input feature count (ne[0] / K) — the same value
    // the weight's create_tensor used for its leading dim.
    auto register_ml8_weight = [&](struct ggml_tensor * weight,
                                   llm_tensor tensor_id, int il_, int64_t k_dim) {
        if (!is_ml8_sidecar_weight(weight)) {
            return;
        }
        struct ggml_tensor * centroids    = nullptr;
        struct ggml_tensor * rotation_h_a = nullptr;
        struct ggml_tensor * rotation_meta = nullptr;
        struct ggml_tensor * awq_scale    = nullptr;
        load_ml8_sidecars(weight, tensor_id, il_, k_dim,
                          &centroids, &rotation_h_a, &rotation_meta, &awq_scale);
        ml8_sidecars sc{ centroids, rotation_h_a, awq_scale };
        fill_rotation_meta(sc, rotation_meta, tensor_id, il_);
        ml8_reg.register_weight(weight, sc);
    };

    // FFN gate/up/down registry registration for every ML8-family type
    // (ML8_4, ML8_FP8, FP8_B128). build_layer_ffn routes all of them through
    // build_ffn()/build_lora_mm()/build_ml8_or_mul_mat, whose
    // apply_ml8_input_xform applies awq + both rotation kinds from the
    // registered sidecars (2026-09-18: the former inline ML8_4 FFN path only
    // knew the kronecker kind and skipped ffn_down's block_hadamard rotation).
    // ML8_4 needs its centroids (LUT) and awq_scale registered too; the fp8
    // family carries neither. Sidecar tensors are already created by
    // load_ml8_sidecars at every call site; this only registers the
    // (weight -> sidecars) mapping.
    auto register_ffn = [&](struct ggml_tensor * weight,
                            struct ggml_tensor * centroids,
                            struct ggml_tensor * rotation_h_a,
                            struct ggml_tensor * rotation_meta,
                            struct ggml_tensor * awq_scale,
                            llm_tensor tensor_id, int il_) {
        if (!weight || (weight->type != GGML_TYPE_ML8_4 && weight->type != GGML_TYPE_ML8_FP8 && weight->type != GGML_TYPE_FP8_B128)) {
            return;
        }
        if (weight->type == GGML_TYPE_ML8_4) {
            GGML_ASSERT(centroids && "ML8_4 FFN weight missing centroids sidecar");
        }
        ml8_sidecars sc{ weight->type == GGML_TYPE_ML8_4 ? centroids : nullptr, rotation_h_a,
                         weight->type == GGML_TYPE_ML8_4 ? awq_scale : nullptr };
        fill_rotation_meta(sc, rotation_meta, tensor_id, il_);
        ml8_reg.register_weight(weight, sc);
    };

    // token_embd ml8-4 sidecars (MAD-256). A native-4-bit token_embd needs its
    // centroid LUT for BOTH the input embedding gather (ggml_ml8_get_rows, via
    // build_inp_embd) and, when the LM head is tied, the output logits
    // projection. Register it under the tok_embd pointer so build_inp_embd
    // finds it; capture the centroids so a tied output head can share the LUT.
    struct ggml_tensor * tok_embd_centroids = nullptr;
    {
        struct ggml_tensor * te_rot = nullptr, * te_rmeta = nullptr, * te_awq = nullptr;
        load_ml8_sidecars(tok_embd, LLM_TENSOR_TOKEN_EMBD, -1, n_embd,
                          &tok_embd_centroids, &te_rot, &te_rmeta, &te_awq);
        if (tok_embd && tok_embd->type == GGML_TYPE_ML8_4) {
            ml8_reg.register_weight(tok_embd, { tok_embd_centroids, te_rot, te_awq });
        }
    }

    // lm_head (output) sidecars are model-scope (no il). When the head is tied
    // (duplicated from token_embd) it shares token_embd's weights AND centroid
    // LUT — there is no separate output.centroids in the GGUF, so reuse the
    // token_embd LUT. When untied, load the output's own sidecars.
    if (output_tied) {
        if (output && output->type == GGML_TYPE_ML8_4) {
            ml8_reg.register_weight(output, { tok_embd_centroids, nullptr, nullptr });
        }
    } else {
        register_ml8_weight(output, LLM_TENSOR_OUTPUT, -1, n_embd);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            // ml8-4 registry (MAD-223 T13). create_tensor_qkv populates either
            // a fused wqkv or split wq/wk/wv; register whichever exist. All have
            // input dim n_embd; wo input is n_embd_head_k * n_head.
            register_ml8_weight(layer.wqkv, LLM_TENSOR_ATTN_QKV, il, n_embd);
            register_ml8_weight(layer.wq,   LLM_TENSOR_ATTN_Q,   il, n_embd);
            register_ml8_weight(layer.wk,   LLM_TENSOR_ATTN_K,   il, n_embd);
            register_ml8_weight(layer.wv,   LLM_TENSOR_ATTN_V,   il, n_embd);
            register_ml8_weight(layer.wo,   LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);

            // ml8-4 registry (MAD-223 T13). wqkv/wqkv_gate input dim is n_embd;
            // ssm_out input dim is value_dim.
            register_ml8_weight(layer.wqkv,      LLM_TENSOR_ATTN_QKV,  il, n_embd);
            register_ml8_weight(layer.wqkv_gate, LLM_TENSOR_ATTN_GATE, il, n_embd);
            register_ml8_weight(layer.ssm_out,   LLM_TENSOR_SSM_OUT,   il, value_dim);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);

        // ml8-4 / ml8-fp8 sidecars (MAD-223; MAD-266). No-op when the FFN
        // weights are not ml8-typed.
        load_ml8_sidecars(layer.ffn_gate, LLM_TENSOR_FFN_GATE, il, n_embd,
                          &layer.ffn_gate_centroids, &layer.ffn_gate_rotation_h_a,
                          &layer.ffn_gate_rotation_meta, &layer.ffn_gate_awq_scale);
        load_ml8_sidecars(layer.ffn_up,   LLM_TENSOR_FFN_UP,   il, n_embd,
                          &layer.ffn_up_centroids,   &layer.ffn_up_rotation_h_a,
                          &layer.ffn_up_rotation_meta,   &layer.ffn_up_awq_scale);
        load_ml8_sidecars(layer.ffn_down, LLM_TENSOR_FFN_DOWN, il, n_ff,
                          &layer.ffn_down_centroids, &layer.ffn_down_rotation_h_a,
                          &layer.ffn_down_rotation_meta, &layer.ffn_down_awq_scale);
        register_ffn(layer.ffn_gate, layer.ffn_gate_centroids, layer.ffn_gate_rotation_h_a,
                     layer.ffn_gate_rotation_meta, layer.ffn_gate_awq_scale, LLM_TENSOR_FFN_GATE, il);
        register_ffn(layer.ffn_up,   layer.ffn_up_centroids,   layer.ffn_up_rotation_h_a,
                     layer.ffn_up_rotation_meta,   layer.ffn_up_awq_scale,   LLM_TENSOR_FFN_UP,   il);
        register_ffn(layer.ffn_down, layer.ffn_down_centroids, layer.ffn_down_rotation_h_a,
                     layer.ffn_down_rotation_meta, layer.ffn_down_awq_scale, LLM_TENSOR_FFN_DOWN, il);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // ml8-4 registry (MAD-223 T13). MTP attention mirrors the trunk dense
        // attention: fused wqkv or split wq/wk/wv (input n_embd), wo input
        // n_embd_head_k * n_head.
        register_ml8_weight(layer.wqkv, LLM_TENSOR_ATTN_QKV, il, n_embd);
        register_ml8_weight(layer.wq,   LLM_TENSOR_ATTN_Q,   il, n_embd);
        register_ml8_weight(layer.wk,   LLM_TENSOR_ATTN_K,   il, n_embd);
        register_ml8_weight(layer.wv,   LLM_TENSOR_ATTN_V,   il, n_embd);
        register_ml8_weight(layer.wo,   LLM_TENSOR_ATTN_OUT, il, n_embd_head_k * n_head);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // ml8-4 / ml8-fp8 sidecars (MAD-223; MAD-266) — same wiring as the
        // trunk block. No-op if not ml8-typed.
        load_ml8_sidecars(layer.ffn_gate, LLM_TENSOR_FFN_GATE, il, n_embd,
                          &layer.ffn_gate_centroids, &layer.ffn_gate_rotation_h_a,
                          &layer.ffn_gate_rotation_meta, &layer.ffn_gate_awq_scale);
        load_ml8_sidecars(layer.ffn_up,   LLM_TENSOR_FFN_UP,   il, n_embd,
                          &layer.ffn_up_centroids,   &layer.ffn_up_rotation_h_a,
                          &layer.ffn_up_rotation_meta,   &layer.ffn_up_awq_scale);
        load_ml8_sidecars(layer.ffn_down, LLM_TENSOR_FFN_DOWN, il, n_ff,
                          &layer.ffn_down_centroids, &layer.ffn_down_rotation_h_a,
                          &layer.ffn_down_rotation_meta, &layer.ffn_down_awq_scale);
        register_ffn(layer.ffn_gate, layer.ffn_gate_centroids, layer.ffn_gate_rotation_h_a,
                     layer.ffn_gate_rotation_meta, layer.ffn_gate_awq_scale, LLM_TENSOR_FFN_GATE, il);
        register_ffn(layer.ffn_up,   layer.ffn_up_centroids,   layer.ffn_up_rotation_h_a,
                     layer.ffn_up_rotation_meta,   layer.ffn_up_awq_scale,   LLM_TENSOR_FFN_UP,   il);
        register_ffn(layer.ffn_down, layer.ffn_down_centroids, layer.ffn_down_rotation_h_a,
                     layer.ffn_down_rotation_meta, layer.ffn_down_awq_scale, LLM_TENSOR_FFN_DOWN, il);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        // ml8-4 registry (MAD-223 T13). eh_proj input dim is 2 * n_embd.
        register_ml8_weight(layer.nextn.eh_proj, LLM_TENSOR_NEXTN_EH_PROJ, il, 2 * n_embd);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = model.pipeline_layer_first() == 0
        ? build_inp_embd(model.tok_embd)
        : build_inp_hidden();

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = model.pipeline_layer_first(); il <= model.pipeline_layer_last(); ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (model.pipeline_layer_last() != n_layer - 1) {
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // MAD-LAB logits-on-head: the dense-segment TAIL worker stops here. res->t_embd
    // is exactly the tensor the LM head would consume, so the head can finish the
    // projection from the n_embd-wide wire payload and get the SAME matmul input
    // the tail would have fed it. Gating AFTER t_embd is deliberate -- the cut is
    // "post output_norm, pre projection", which keeps the RMS norm on the device
    // that runs it today and moves exactly one op across the wire boundary.
    if (cparams.no_output_head) {
        ggml_build_forward_expand(gf, cur);
        return;
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;
    // Pin z's GEMM into the graph HERE. ggml's DFS emits mul_gate's src[0]
    // subgraph (conv -> GDN -> RMS_NORM -> MUL) before its src[1] subgraph
    // (z GEMM -> SILU), so without this z is materialized AFTER the gated
    // norm's RMS_NORM -- the ml8-radiance pattern-C fusion (ggml-cuda.cu),
    // which runs at that RMS_NORM, read an unwritten z (chain 254: all zeros).
    ggml_build_forward_expand(gf, z);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    // MAD-406 (R4D GDN conv_prep, task 4). Expose the layer's raw-A_log sidecar
    // (llama_model_build_ssm_a_log_sidecars, llama-model.cpp; task 1) as an EXTRA src on this
    // SSM_CONV node so ggml-cuda's graph-level conv_prep fusion detector
    // (ggml_cuda_try_gdn_conv_prep_fusion, mt_gdn_r4d.cu) can find it without any dispatcher or
    // graph-shape change: every existing SSM_CONV consumer (CPU, plain CUDA/HIP compute paths,
    // ggml-cuda.cu's own SSM_CONV+SiLU fusion) reads only src[0]/src[1] and ignores src[2..], and
    // ssm_a_log is already a fully-resident, pre-allocated leaf (not something this graph needs to
    // compute or schedule) -- so this costs nothing beyond one more visited-leaf hash entry on the
    // R4D build. Left null (as it already is) for any layer/arch the sidecar wasn't derived for;
    // the detector treats a null src[2] as "not available" and declines.
    conv_output_proper->src[2] = model.layers[il].ssm_a_log;

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    const auto & layer = model.layers[il];

    // LLAMA_ACT_BF16 (2026-09-18 phase 2, read once): when set AND this is a
    // prefill ubatch (n_tokens > 32 -- the same threshold
    // ggml_cuda_ml8_4_mul_mat_supports_bf16_out in ml8.cu gates the
    // RDNA4_TRFEED bf16-dst GEMM path on, since the M_pad==32 decode kernel
    // has no bf16 epilogue), run the ffn_up/ffn_gate GEMM outputs and the
    // SwiGLU combine in bf16 instead of f32: halves the GEMM-output write
    // traffic and the GLU's read+write traffic (both already have CUDA-backend
    // bf16 support from Phase 1's BF16 activation coverage).
    //
    // SCOPE NOTE: this is deliberately narrower than the full
    // residual-stream-in-bf16 mode the phase-2 task describes. `cur` going
    // in and ffn_down's output are both left f32, so the residual stream and
    // every other op (attention, rope, GATED_DELTA_NET, norms, the LM head)
    // are completely unchanged by this switch -- only the three tensors
    // between ffn_up/ffn_gate and ffn_down run in bf16. Extending bf16
    // through the rest of the graph (build_norm's RMS_NORM+MUL, Qcur/Kcur/
    // Vcur, the residual adds) was not completed in this pass; see the report.
    static const bool act_bf16_env = [] {
        const char * e = std::getenv("LLAMA_ACT_BF16");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    const bool use_bf16_ffn = act_bf16_env && n_tokens > 32 &&
        layer.ffn_up->type   == GGML_TYPE_ML8_4 &&
        layer.ffn_gate->type == GGML_TYPE_ML8_4 &&
        layer.ffn_down->type == GGML_TYPE_ML8_4 &&
        !layer.ffn_up_s && !layer.ffn_gate_s && !layer.ffn_down_s;

    if (use_bf16_ffn) {
        ggml_tensor * up_b = build_lora_mm(layer.ffn_up, cur, nullptr, GGML_TYPE_BF16);
        cb(up_b, "ffn_up", il);
        ggml_tensor * gate_b = build_lora_mm(layer.ffn_gate, cur, nullptr, GGML_TYPE_BF16);
        cb(gate_b, "ffn_gate", il);
        // Same operand order as build_ffn's LLM_FFN_SILU/LLM_FFN_PAR case:
        // gate first (silu applied to it), up second.
        ggml_tensor * act = ggml_swiglu_split(ctx0, gate_b, up_b);
        cb(act, "ffn_swiglu", il);
        cur = build_lora_mm(layer.ffn_down, act);
        cb(cur, "ffn_out", il);
        return cur;
    }

    // ML8_4 / FP8_B128 / ML8_FP8 FFN weights all dispatch through build_ffn ->
    // build_lora_mm -> build_ml8_or_mul_mat (llama-ml8-registry.cpp), whose
    // apply_ml8_input_xform reads the registered sidecars and applies BOTH
    // rotation kinds (kronecker via rotation_h_a, block_hadamard via
    // rotation_meta) plus awq. An earlier inline ML8_4-only branch here had
    // its own copy of that transform that only knew the kronecker kind, so a
    // block_hadamard-rotated ffn_down (the converter's choice for every
    // K-split weight: rotation_meta kind 2, h_a absent) was multiplied by an
    // UNROTATED activation in every layer -- token-salad output from an
    // otherwise numerically-correct file (2026-09-18). One transform
    // implementation, in the registry, is the rule.
    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    static const bool wp_kv_only = [] {
        const char * env = std::getenv("WP_MTP_PREFILL_KV_ONLY");
        return env != nullptr && env[0] == '1';
    }();
    if (wp_kv_only && n_outputs == 0 && n_tokens > 32 && !cparams.embeddings &&
            cparams.embeddings_nextn_masked &&
            std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool enabled) { return enabled; }) &&
            !inp_attn->is_paged && !inp_attn->self_k_rot && !inp_attn->self_v_rot &&
            inp_attn->mctx->get_k(ctx0, il)->type == GGML_TYPE_F16 &&
            inp_attn->mctx->get_v(ctx0, il)->type == GGML_TYPE_F16) {
        // Catch-up needs K/V; its next hidden input comes from the target.
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        return;
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
