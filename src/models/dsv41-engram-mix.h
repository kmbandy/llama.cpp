#pragma once

// WP_SPINE_ROW_GROUPS (S7): the engram mix, factored out of llama_model_deepseek41::graph::build_engram
// so the CPU test (tests/test-dsv4-rowgroup-engram.cpp) runs the same ops the model graph runs.
//
// Inputs: x  = stream [n_embd, hc, nt] (computed tensor)
//         kv = wkv applied to the gathered engram rows, [hc*n_embd + n_embd, nt] (computed tensor)
// Output: x + v * gate, with the per-(token, hc copy) gate and the shared value from kv.
// Every op is per token row, so for a row range [r0, r0+w) of x and kv the result equals the
// same row range of the whole-batch result (the row-group path relies on this).

#include "ggml.h"

#include <cmath>
#include <cstdint>

inline ggml_tensor * dsv41_engram_mix(
        ggml_context * ctx0,
        ggml_cgraph * gf,
        ggml_tensor * x,
        ggml_tensor * kv,
        ggml_tensor * w_k,
        ggml_tensor * w_q,
        int64_t       n_embd,
        int64_t       hc,
        float         norm_rms_eps,
        ggml_tensor ** gate_out) {
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * key   = ggml_cont_3d(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0), n_embd, hc, nt);
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));
    // Pin the order: both copies out of kv first so kv dies before the norms, and the gate
    // before the value's repeat below, so at most three [n_embd, hc, nt] tensors are live.
    ggml_build_forward_expand(gf, value);
    ggml_build_forward_expand(gf, key);

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before
    // llama-quant.cpp learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        GGML_ASSERT(t->ne[0] == n_embd && t->ne[1] == hc && t->ne[2] == nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        return ggml_mul(ctx0, t, ggml_reshape_3d(ctx0, w, n_embd, hc, 1));
    };

    ggml_tensor * k = grouped_norm(key, as_f32(w_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(w_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    ggml_build_forward_expand(gf, gate);
    if (gate_out) {
        *gate_out = gate;
    }

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}
