#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

class llama_memory_hybrid_idx_context;
class llama_dsv4_comp_state;

// The compressor state of a DeepSeek-V4 style compressed stream, as the graph sees it.
struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

// Shared by the DeepSeek-V4 and V4.1 graphs, which run the same compressed stream machinery.
float dsv4_rope_attn_factor(float freq_scale, float ext_factor);

ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0);

dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il);

dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il);

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

//
// models
//

struct llama_model_llama : public llama_model_base {
    llama_model_llama(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llama4 : public llama_model_base {
    llama_model_llama4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llama_embed : public llama_model_llama {
    llama_model_llama_embed(const struct llama_model_params & params) : llama_model_llama(params) {}
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool embed>
    using graph = llama_model_llama::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_maincoder : public llama_model_base {
    llama_model_maincoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_talkie : public llama_model_base {
    llama_model_talkie(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deci : public llama_model_base {
    llama_model_deci(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_baichuan : public llama_model_base {
    llama_model_baichuan(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_falcon : public llama_model_base {
    llama_model_falcon(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_grok : public llama_model_base {
    llama_model_grok(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_starcoder : public llama_model_base {
    llama_model_starcoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_refact : public llama_model_base {
    llama_model_refact(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bert : public llama_model_base {
    llama_model_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jina_bert_v2 : public llama_model_base {
    llama_model_jina_bert_v2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jina_bert_v3 : public llama_model_base {
    llama_model_jina_bert_v3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nomic_bert : public llama_model_base {
    llama_model_nomic_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nomic_bert_moe : public llama_model_base {
    llama_model_nomic_bert_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_modern_bert : public llama_model_base {
    llama_model_modern_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_neo_bert : public llama_model_base {
    llama_model_neo_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_eurobert : public llama_model_base {
    llama_model_eurobert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bloom : public llama_model_base {
    llama_model_bloom(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llama_model_base interface
struct llama_model_clip : public llama_model_base {
    llama_model_clip(const struct llama_model_params & params) : llama_model_base(params) {}

    [[noreturn]]
    void load_arch_hparams(llama_model_loader & ml) override;

    [[noreturn]]
    void load_arch_tensors(llama_model_loader & ml) override;

    [[noreturn]]
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mpt : public llama_model_base {
    llama_model_mpt(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_stablelm : public llama_model_base {
    llama_model_stablelm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_mellum : public llama_model_base {
    llama_model_mellum(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_nanbeige : public llama_model_base {
    llama_model_nanbeige(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    int  n_loops = 1;
    int  n_layer_phys = 0;
    bool skip_loop_final_norm = false;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_qwen : public llama_model_base {
    llama_model_qwen(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2 : public llama_model_base {
    llama_model_qwen2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dream : public llama_model_base {
    llama_model_dream(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llada : public llama_model_base {
    llama_model_llada(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llada_moe : public llama_model_base {
    llama_model_llada_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rnd1 : public llama_model_base {
    llama_model_rnd1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2vl : public llama_model_base {
    llama_model_qwen2vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2moe : public llama_model_base {
    llama_model_qwen2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3 : public llama_model_base {
    llama_model_qwen3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3moe : public llama_model_base {
    llama_model_qwen3moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3vl : public llama_model_base {
    llama_model_qwen3vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3vlmoe : public llama_model_base {
    llama_model_qwen3vlmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3tts : public llama_model_qwen3vl {
    llama_model_qwen3tts(const struct llama_model_params & params) : llama_model_qwen3vl(params) {}
};


struct llama_model_phi2 : public llama_model_base {
    llama_model_phi2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_phi3 : public llama_model_base {
    llama_model_phi3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_phimoe : public llama_model_base {
    llama_model_phimoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo : public llama_model_base {
    llama_model_plamo(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo2 : public llama_model_base {
    llama_model_plamo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        private:
            ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
            ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                    const llama_model & model, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo3 : public llama_model_base {
    llama_model_plamo3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gpt2 : public llama_model_base {
    llama_model_gpt2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_pockettts : public llama_model_base {
    llama_model_pockettts(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_codeshell : public llama_model_base {
    llama_model_codeshell(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_orion : public llama_model_base {
    llama_model_orion(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_internlm2 : public llama_model_base {
    llama_model_internlm2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minicpm3 : public llama_model_base {
    llama_model_minicpm3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma : public llama_model_base {
    llama_model_gemma(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma2 : public llama_model_base {
    llama_model_gemma2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma3 : public llama_model_base {
    llama_model_gemma3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma3n : public llama_model_base {
    llama_model_gemma3n(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const llama_model & model;

        const int64_t n_embd_head;
        const int64_t n_embd_altup;
        const int64_t n_altup;
        const int     i_altup_act;
        const int     n_layer_sparsity = 10; // number of layers using activation sparsity
        const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * calc_magnitude(ggml_tensor * x);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);

        ggml_tensor * gaussian_topk(ggml_tensor * x);
        ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
        ggml_tensor * altup_predict(ggml_tensor * cur, int il);
        ggml_tensor * laurel(ggml_tensor * cur, int il);
        ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma4 : public llama_model_base {
    llama_model_gemma4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const llama_model & model;

        const int64_t n_embd_per_layer;

        graph(const llama_model & model, const llm_graph_params & params);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma4_assistant : public llama_model_base {
    llama_model_gemma4_assistant(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma_embedding : public llama_model_base {
    llama_model_gemma_embedding(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_starcoder2 : public llama_model_base {
    llama_model_starcoder2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mamba : public llama_model_base {
    llama_model_mamba(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mamba2 : public llama_model_base {
    llama_model_mamba2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_mamba::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_maple : public llama_model_base {
    llama_model_maple(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jamba : public llama_model_base {
    llama_model_jamba(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_xverse : public llama_model_base {
    llama_model_xverse(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_command_r : public llama_model_base {
    llama_model_command_r(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_cohere2 : public llama_model_base {
    llama_model_cohere2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_cohere2moe : public llama_model_base {
    llama_model_cohere2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dbrx : public llama_model_base {
    llama_model_dbrx(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmo : public llama_model_base {
    llama_model_olmo(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmo2 : public llama_model_base {
    llama_model_olmo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmoe : public llama_model_base {
    llama_model_olmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_muse_glimmer : public llama_model_base {
    llama_model_muse_glimmer(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_openelm : public llama_model_base {
    llama_model_openelm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gptneox : public llama_model_base {
    llama_model_gptneox(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arctic : public llama_model_base {
    llama_model_arctic(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek : public llama_model_base {
    llama_model_deepseek(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek2 : public llama_model_base {
    llama_model_deepseek2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek32 : public llama_model_base {
    llama_model_deepseek32(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dots3note : public llama_model_base {
    llama_model_dots3note(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_deepseek4 : public llama_model_base {
    llama_model_deepseek4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params) : llm_graph_context(params) {}
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** post,
                ggml_tensor ** comb,
                int il) const;

        // V4.1 Single-Pass mHC: compute pre/post/comb without collapsing.
        void build_hc_mixes(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** pre,
                ggml_tensor ** post,
                ggml_tensor ** comb,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention(
                const llama_model & model,
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention_impl(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_hca_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t ratio,
                int64_t n_embd_head,
                const char * name,
                int il,
                ggml_tensor ** pre_rope = nullptr) const;

        ggml_tensor * build_overlap_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t ratio,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_lid_top_k(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_top_k_mask(
                ggml_tensor * kq_mask,
                ggml_tensor * top_k,
                const char * name,
                int il) const;

        ggml_tensor * build_csa_lid_attention(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hca_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_raw_attention(
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;

        // WP_DS4_CONST_SHAPE: pins graph topology (indexer/CSA top-k padding,
        // n_stream canonicalization) so a compiled graph can be captured/reused
        // across ubatches with different content. Exposed here (the underlying
        // check is a file-static in deepseek4.cpp) so deepseek41's CED prefill
        // trim -- a *conditional* topology change keyed on per-ubatch content --
        // can refuse to combine with it; the two are incompatible by construction
        // (see llama_model_deepseek41::graph::graph()).
        static bool ds4_const_shape_enabled();

    protected:
        struct constant_cache_entry {
            ggml_type type;
            int64_t ne[GGML_MAX_DIMS];
            uint32_t value_bits;
            ggml_tensor * tensor;
        };

        struct weight_view_cache_entry {
            ggml_tensor * source;
            int64_t ne0;
            int64_t i0;
            ggml_tensor * view;
        };

        ggml_tensor * get_constant(
                ggml_type type,
                const int64_t ne[GGML_MAX_DIMS],
                float value) const;

        ggml_tensor * get_weight_view_1d(
                ggml_tensor * tensor,
                int64_t ne0,
                int64_t i0) const;

        mutable std::vector<constant_cache_entry> constant_cache;
        mutable std::vector<weight_view_cache_entry> weight_view_cache;
    };

    struct graph_mtp : public graph {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_deepseek41 : public llama_model_deepseek4 {
    llama_model_deepseek41(const struct llama_model_params & params) : llama_model_deepseek4(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // Engram n-gram hasher (DeepSeek inference/engram.py NgramHashState).
    // token_map + multipliers come from the GGUF (convert emits them; the
    // tokenizers normalizer chain and numpy's rng are not reproducible here);
    // primes and bucket offsets are derived from engram.vocab_size at load and
    // checked against each table's row count. This bit-exact host-side hash
    // math is unchanged from the from-scratch implementation; only its data
    // source changed (see llm_graph_input_engram in deepseek41.cpp).
    struct engram_hasher {
        uint32_t max_ngram = 4;
        uint32_t n_heads   = 8;
        uint32_t n_layers  = 0;   // engram layers, ordinal order of engram.layer_ids
        int32_t  pad_id    = 0;   // COMPRESSED pad id (token_map[engram_pad_token_id])
        std::vector<int32_t> token_map;   // [n_vocab]; empty = identity (not converted)
        std::vector<int64_t> multipliers; // [n_layers][max_ngram]
        std::vector<int64_t> primes;      // [n_layers][max_ngram-1][n_heads]
        std::vector<int64_t> offsets;     // same layout, exclusive prefix sum per layer

        uint32_t n_cols() const { return (max_ngram > 1 ? max_ngram - 1 : 1) * n_heads; }
        bool ready() const { return n_layers > 0 && !multipliers.empty() && !primes.empty(); }
        int32_t compress(int32_t tok) const {
            return (tok >= 0 && (size_t) tok < token_map.size()) ? token_map[(size_t) tok] : tok;
        }
        // NgramHashState.forward for one position of one engram layer:
        // ctx_ids[0..max_ngram) holds COMPRESSED ids, index 0 = current token,
        // index s = s positions back (pad already substituted for missing
        // lookback by the caller); writes n_cols() ids to row.
        void hash_position(const int32_t * ctx_ids, uint32_t ordinal, int32_t * row) const;
    } engram;

    // position of layer il in the engram constants, or -1 if that layer has no engram
    int engram_index(int il) const;

    struct graph : public llama_model_deepseek4::graph {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * identity_pre_mix() const;

        // gather the n-gram rows of layer il: [engram_key_length * n_hash_cols, n_tokens]
        ggml_tensor * build_inp_engram(
                const llama_model & model,
                int il);

        ggml_tensor * build_engram(
                const llama_model & model,
                ggml_tensor * x,
                ggml_tensor * emb,
                int il) const;

        struct dsv41_rope_cfg rope_cfg(int il) const;

        ggml_tensor * build_attention_tail(
                const llama_model & model,
                ggml_tensor * out,
                ggml_tensor * inp_pos,
                int64_t nt,
                int il) const;

        // WP_DSV41_TOPK_REUSE: the most recent index source's top-k picks, which
        // the compressed layers after it reuse (reference model.py:
        // shared_attn.topk_idxs). Set while building the source layer, read by
        // the layers whose dsv41_topk_source names it.
        mutable ggml_tensor * dsv41_shared_top_k     = nullptr;
        mutable int           dsv41_shared_top_k_src = -1;

        // which compressed positions this layer's queries attend to
        ggml_tensor * build_indexer_top_k(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                const llm_graph_input_dsv4::comp_input & inp_comp,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        // cur_state (CED prefill trim, WP_DSV41_CED_PREFILL): defaults to `cur`.
        // Pass a WIDER, full-token `cur_state` only for the seam layer under the
        // trim, where `cur` itself has been narrowed to the trailing W-token
        // query window but the kv-source/indexer-publish block (the compressed
        // global KV every decoder layer reads, and the shared index keys) still
        // needs every token -- see the encoder/decoder split comment at the top
        // of this file and the trim gate in graph::graph().
        // ced_log_shapes (CED prefill trim observability, WP_DSV41_CED_PREFILL):
        // when true, logs q/k_all/kq_mask/n_kv_max shapes for this call via
        // dsv41_ced_log_if_changed's channel 2. The caller (graph::graph(),
        // which knows ced_seam) decides which layers this applies to; false
        // (default) costs nothing beyond the boolean check.
        // publish_only (CED prefill trim, WP_DSV41_CED_SKIP_NONFINAL): the
        // caller only needs the kv-source/indexer publish below (a non-final
        // ubatch's seam layer, see graph::graph()'s ced_skip_layer) -- skip
        // this layer's own query/RoPE/raw-window-write/attention entirely and
        // return nullptr. Requires cur_state != nullptr and a compressed
        // (ratio != 0) layer; both already guaranteed by the caller.
        ggml_tensor * build_attention_v41(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il,
                ggml_tensor * cur_state = nullptr,
                bool ced_log_shapes = false,
                bool publish_only = false) const;

    private:
        void build_dspark_encoder(const llama_model & model);
        void build_dspark_stages(const llama_model & model);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek2ocr : public llama_model_base {
    llama_model_deepseek2ocr(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm_dsa : public llama_model_base {
    llama_model_glm_dsa(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_eagle3 : public llama_model_base {
    llama_model_eagle3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dflash : public llama_model_base {
    llama_model_dflash(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    struct graph_dsv4 : public llama_model_deepseek4::graph {
        // MAD-LAB: allow the DSV4 graph to target in-model stage blocks.
        graph_dsv4(const llama_model & model, const llm_graph_params & params,
                   int stage_base = 0, int n_stages = 0);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mistral4 : public llama_model_deepseek2 {
    llama_model_mistral4(const struct llama_model_params & params) : llama_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_deepseek2

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_chatglm : public llama_model_base {
    llama_model_chatglm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm4 : public llama_model_base {
    llama_model_glm4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm4_moe : public llama_model_base {
    llama_model_glm4_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bitnet : public llama_model_base {
    llama_model_bitnet(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_t5 : public llama_model_base {
    llama_model_t5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_t5encoder : public llama_model_base {
    llama_model_t5encoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_t5::graph<true>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jais : public llama_model_base {
    llama_model_jais(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jais2 : public llama_model_base {
    llama_model_jais2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron : public llama_model_base {
    llama_model_nemotron(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron_h : public llama_model_base {
    llama_model_nemotron_h(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
            const llama_model & model, int64_t n_embd_head, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron_h_moe : public llama_model_nemotron_h {
    llama_model_nemotron_h_moe(const struct llama_model_params & params) : llama_model_nemotron_h(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_nemotron_h

    using graph = llama_model_nemotron_h::graph;

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone : public llama_model_base {
    llama_model_exaone(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone4 : public llama_model_base {
    llama_model_exaone4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone_moe : public llama_model_base {
    llama_model_exaone_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv6 : public llama_model_base {
    llama_model_rwkv6(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv6qwen2 : public llama_model_base {
    llama_model_rwkv6qwen2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv7 : public llama_model_base {
    llama_model_rwkv7(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arwkv7 : public llama_model_base {
    llama_model_arwkv7(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite : public llama_model_base {
    llama_model_granite(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  llm_graph_input_attn_kv * inp_attn,
            const llama_model             & model,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
            const llama_model       & model,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_moe : public llama_model_base {
    llama_model_granite_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_switch : public llama_model_base {
    llama_model_granite_switch(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    uint32_t n_adapters    = 0;
    uint32_t max_lora_rank = 0;
    float    router_gain   = 15.0f;

    std::unordered_map<llama_token, int32_t>     adapter_token_to_slot;
    std::unordered_map<llama_token, llama_token> adapter_token_to_substitute;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_switched_lora_delta(
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_switched_lora_mm(
                  ggml_tensor * w,
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  ggml_tensor             * adapter_ids,
                  llm_graph_input_attn_kv * inp_attn,
            const llama_model             & model,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
                  ggml_tensor       * adapter_ids,
            const llama_model       & model,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minicpm : public llama_model_base {
    llama_model_minicpm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_hybrid : public llama_model_base {
    llama_model_granite_hybrid(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const llama_model & model, const int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
            const llama_model & model,const int64_t n_embd_head, const int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_swa : public llama_model_base {
    llama_model_granite_swa(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor                  * cur,
                  ggml_tensor                  * inp_pos,
                  llm_graph_input_attn_kv_iswa * inp_attn,
            const llama_model                  & model,
            const int64_t                        n_embd_head,
            const int                            il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor * cur,
                  ggml_tensor * inpSA,
            const llama_model & model,
            const int           il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_chameleon : public llama_model_base {
    llama_model_chameleon(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_wavtokenizer_dec : public llama_model_base {
    llama_model_wavtokenizer_dec(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plm : public llama_model_base {
    llama_model_plm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hrm_text : public llama_model_base {
    llama_model_hrm_text(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        const llama_model & model;

        ggml_tensor * build_stack(
                llm_graph_input_attn_kv * inp_attn,
                        ggml_tensor * inp_pos,
                        ggml_tensor * cur,
                                int   slot_base) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bailingmoe : public llama_model_base {
    llama_model_bailingmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bailingmoe2 : public llama_model_base {
    llama_model_bailingmoe2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bailingmoe3 : public llama_model_base {
    llama_model_bailingmoe3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_seed_oss : public llama_model_base {
    llama_model_seed_oss(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dots1 : public llama_model_base {
    llama_model_dots1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arcee : public llama_model_base {
    llama_model_arcee(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_afmoe : public llama_model_base {
    llama_model_afmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_laguna : public llama_model_base {
    llama_model_laguna(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_ernie4_5 : public llama_model_base {
    llama_model_ernie4_5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_ernie4_5_moe : public llama_model_ernie4_5 {
    llama_model_ernie4_5_moe(const struct llama_model_params & params) : llama_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_paddleocr : public llama_model_ernie4_5 {
    llama_model_paddleocr(const struct llama_model_params & params) : llama_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_moe : public llama_model_base {
    llama_model_hunyuan_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_hy_v3 : public llama_model_base {
    llama_model_hy_v3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hy_v4 : public llama_model_base {
    llama_model_hy_v4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        // iHC (independent Hyper-Connections): pre reduces the hc streams to one and returns the
        // per-stream post gates, post writes the sublayer output back into the streams, head
        // collapses the streams before the final norm.
        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** post,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                const llama_model & model,
                llm_graph_input_attn_k * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                float kq_scale,
                int il) const;

        // DSA lightning indexer: top-k KV positions for this layer. Only "full" layers compute
        // it, "shared" layers reuse the last preceding full layer result through last_top_k.
        ggml_tensor * build_indexer_top_k(
                const llama_model & model,
                llm_graph_input_attn_k_dsa * inp_attn_dsa,
                ggml_tensor * cur,
                ggml_tensor * qr,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention_dsa(
                const llama_model & model,
                llm_graph_input_attn_k_dsa * inp_attn_dsa,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor ** last_top_k,
                float kq_scale,
                int il) const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_vl : public llama_model_base {
    llama_model_hunyuan_vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_dense : public llama_model_hunyuan_vl {
    llama_model_hunyuan_dense(const struct llama_model_params & params) : llama_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_hunyuan_vl

    using graph = llama_model_hunyuan_vl::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_smollm3 : public llama_model_base {
    llama_model_smollm3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_openai_moe : public llama_model_base {
    llama_model_openai_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_falcon_h1 : public llama_model_base {
    llama_model_falcon_h1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_lfm2 : public llama_model_base {
    llama_model_lfm2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_lfm2moe : public llama_model_base {
    llama_model_lfm2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_lfm2::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_smallthinker : public llama_model_base {
    llama_model_smallthinker(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_grovemoe : public llama_model_base {
    llama_model_grovemoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_apertus : public llama_model_base {
    llama_model_apertus(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minimax_01 : public llama_model_base {
    llama_model_minimax_01(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minimax_m2 : public llama_model_base {
    llama_model_minimax_m2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct msa_params {
    int blk;
    int topk_blocks;
    int local;
};

struct llama_model_minimax_m3 : public llama_model_base {
    llama_model_minimax_m3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    msa_params msa_p;
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_attn_msa_fa(
                ggml_tensor * q_cur,   // [D, HQ, S] f32
                ggml_tensor * k,       // [D, n_keys, 1, C]  C = HKV or HKV*n_stream
                ggml_tensor * v,       // [D, n_keys, 1, C]
                ggml_tensor * mask,    // [n_keys, R, 1, C] f16, R = HQ*T/(Gp*C)
                int64_t Gp, float kq_scale, int il) const;
    };
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_cogvlm : public llama_model_base {
    llama_model_cogvlm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_pangu_embed : public llama_model_base {
    llama_model_pangu_embed(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3next : public llama_model_base {
    llama_model_qwen3next(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen35 : public llama_model_base {
    llama_model_qwen35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen4exp : public llama_model_base {
    llama_model_qwen4exp(const struct llama_model_params & params) : llama_model_base(params) {}

    class llm_graph_input_qsa;

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    protected:
        // The MTP block reuses the builders below, so graph_mtp chains to this ctor
        // (which does NOT build the trunk graph) and then builds its own.
        graph(const llama_model & model, const llm_graph_params & params, bool /*mtp*/);

        // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
        ggml_tensor * build_hc_mix(
                    ggml_tensor * x,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_down,
                    ggml_tensor * w_up,
                    ggml_tensor * w_inject,
                    ggml_tensor ** inject,
                            int   il);

        ggml_tensor * build_hc_combine(
                    ggml_tensor * residual,
                    ggml_tensor * block_out,
                    ggml_tensor * inject,
                            int   il);

        ggml_tensor * build_layer_attn(
              llm_graph_input_attn_kv * inp_attn,
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        // dense self-attention restricted to the cells that top_k names
        ggml_tensor * build_attn_qsa(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * top_k,
                          float   kq_scale,
                            int   il);

        // the QSA cache layout inputs do not depend on the layer, only on its compress ratio,
        // so the layers sharing a ratio share one input set
        std::map<uint32_t, llm_graph_input_qsa *> qsa_inps;

        // QSA: token indices this layer's queries may attend to, or nullptr for dense
        ggml_tensor * build_qsa_top_k(
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                    ggml_tensor * kq_mask,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // build_rs writes the state tensor in place, so one gather per cache tensor is reused
        std::map<ggml_tensor *, ggml_tensor *> rs_rows;

        // one conv history per cache tensor: delta-net and PLE each have their own
        ggml_tensor * build_conv_state_at(
             llm_graph_input_rs * inp,
                    ggml_tensor * conv_states_all,
                    ggml_tensor * x,
                        int64_t   state_cols,
                        int64_t   channels,
                            int   il);

        ggml_tensor * build_ple(
             llm_graph_input_rs * inp,
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * hidden,
                            int   il);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        // WP_QWEN4EXP_LAYER_CUT (Stage 3, pipelined-prefill): the trunk-layer body
        // (post-FFN res_hc boundary) and the terminal head, factored out of the
        // whole-graph constructor so the staged driver (build_trunk_staged) can
        // build one at a time without duplicating either. The whole-graph
        // constructor calls these from its loop too, so the default-off path is
        // byte-identical -- only the call site moved, the code did not change.
        ggml_tensor * build_trunk_layer(
              llm_graph_input_mem_hybrid * inp,
  const llama_memory_hybrid_idx_context * mctx_hyb,
                    ggml_tensor * inp_pos,
                    ggml_tensor * inp_out_ids,
                            int * sections,
                        int64_t   hc,
                           bool   keep_all_rows,
                    ggml_tensor * res_hc,
                            int   il);

        // lines that used to run once after the loop: h_nextn publication, final
        // HC mixer, output-row selection, result norm, embeddings, logits. No cut
        // inside this stage (plan item 3).
        ggml_tensor * build_trunk_head(
                    ggml_tensor * res_hc,
                        int64_t   hc,
                           bool   keep_all_rows,
                    ggml_tensor * inp_out_ids);

        // WP_QWEN4EXP_LAYER_CUT driver: builds exactly ONE stage (stage_il in
        // [0, n_layer) is a trunk layer, stage_il == n_layer is the terminal
        // head) into this llm_graph_context's ctx0/gf. Shared QSA/hybrid-memory
        // inputs are rebuilt within this call (not cached across stages, since
        // each stage is produced by a separate model.build_graph() call from
        // llama_context::process_ubatch_staged) -- see the "not rebound per
        // stage" deviation noted in the Stage 3 summary.
        void build_trunk_staged(int64_t hc, int * sections, int32_t stage_il);

        const llama_model & model;
    };

    // The MTP/NextN draft block: one dense-attention layer that predicts the
    // next-next token from the target's hyper-connection streams plus the
    // embedding of the token the target just produced. ref: ggml-org#27739
    struct graph_mtp : public graph {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_qwen35moe : public llama_model_base {
    llama_model_qwen35moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen35_mtp : public llama_model_base {
    llama_model_qwen35_mtp(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen35moe_mtp : public llama_model_base {
    llama_model_qwen35moe_mtp(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mistral3 : public llama_model_base {
    llama_model_mistral3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mimo2 : public llama_model_base {
    llama_model_mimo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_kimi_k3 : public llama_model_base {
    llama_model_kimi_k3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);

        const llama_model & model;

        // Cross-layer residual attention (K3's `_apply_attn_res`).
        ggml_tensor * resi_stack = nullptr;

        void          res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens);
        ggml_tensor * res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                              int64_t n_tokens, int il);

        ggml_tensor * build_kda_layer(ggml_tensor * cur, const llama_layer & layer,
                                      llm_graph_input_rs * inp_rs,
                                      int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
                                      int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il);

        ggml_tensor * build_mla_layer(ggml_tensor * cur, const llama_layer & layer,
                                      llm_graph_input_attn_k  * inp_attn_k,
                                      llm_graph_input_attn_kv * inp_attn_kv,
                                      int64_t n_embd_head_k_mla, int64_t n_embd_head_v_mla,
                                      int64_t kv_lora_rank, int64_t n_embd_head_qk_rope,
                                      int64_t n_embd_head_qk_nope, float kq_scale, int il);

        ggml_tensor * build_latent_moe(ggml_tensor * cur, const llama_layer & layer,
                                       int64_t n_embd_latent, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_kimi_linear : public llama_model_base {
    llama_model_kimi_linear(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                            int   il);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                    ggml_tensor * causal_mask,
                    ggml_tensor * identity,
                    ggml_tensor * diag_mask,
                            int   il);

        const llama_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_step35 : public llama_model_base {
    llama_model_step35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_spark2_5 : public llama_model_base {
    llama_model_spark2_5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
