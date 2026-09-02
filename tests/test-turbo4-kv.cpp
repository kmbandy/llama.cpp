// CPU numerical audit for the non-paged turbo4 KV representation.

#include "ggml-common.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int D = 256;
constexpr int N_ROWS = 96;
constexpr int N_BLOCKS_TURBO4 = D / QK_TURBO4;
constexpr int N_BLOCKS_Q4 = D / QK4_0;
constexpr int N_BLOCKS_Q8 = D / QK8_0;

// The test prints these values so a run records which centroid table was audited.
const float turbo4_centroids[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
    0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

const float turbo_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

const float turbo_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

void turbo_wht(float * x, bool inverse) {
    const float * first  = inverse ? turbo_wht_s2 : turbo_wht_s1;
    const float * second = inverse ? turbo_wht_s1 : turbo_wht_s2;

    for (int i = 0; i < 128; ++i) {
        x[i] *= first[i];
    }
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        x[i] *= inv_sqrt_128 * second[i];
    }
}

void turbo_wht_row(float * row, bool inverse) {
    for (int block = 0; block < N_BLOCKS_TURBO4; ++block) {
        turbo_wht(row + block * QK_TURBO4, inverse);
    }
}

void fill_rows(std::vector<float> & rows, uint32_t seed, bool heavy_tail) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int row = 0; row < N_ROWS; ++row) {
        float * dst = rows.data() + row * D;
        for (int i = 0; i < D; ++i) {
            dst[i] = normal(rng);
        }
        if (heavy_tail) {
            dst[(37 * row + 17) % D] = 10.0f * normal(rng);
            dst[(73 * row + 53) % D] = 10.0f * normal(rng);
        }
    }
}

struct stats {
    double ref_sq = 0.0;
    double err_sq = 0.0;
    float max_abs = 0.0f;
};

void add_stats(stats & out, const float * ref, const float * got, int n) {
    for (int i = 0; i < n; ++i) {
        const float diff = got[i] - ref[i];
        out.ref_sq += double(ref[i]) * double(ref[i]);
        out.err_sq += double(diff) * double(diff);
        out.max_abs = std::max(out.max_abs, std::fabs(diff));
    }
}

float relative_l2(const stats & value) {
    return value.ref_sq > 0.0 ? float(std::sqrt(value.err_sq / value.ref_sq)) : 0.0f;
}

void softmax_attention(const float * q, const std::vector<float> & k_rows,
                       const std::vector<float> & v_rows, float * out) {
    std::vector<float> logits(N_ROWS);
    float max_logit = -INFINITY;
    const float scale = 1.0f / std::sqrt(float(D));

    for (int row = 0; row < N_ROWS; ++row) {
        const float * k = k_rows.data() + row * D;
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) {
            dot += q[i] * k[i];
        }
        logits[row] = dot * scale;
        max_logit = std::max(max_logit, logits[row]);
    }

    float sum = 0.0f;
    for (int row = 0; row < N_ROWS; ++row) {
        logits[row] = std::exp(logits[row] - max_logit);
        sum += logits[row];
    }

    for (int i = 0; i < D; ++i) {
        out[i] = 0.0f;
    }
    for (int row = 0; row < N_ROWS; ++row) {
        const float weight = logits[row] / sum;
        const float * v = v_rows.data() + row * D;
        for (int i = 0; i < D; ++i) {
            out[i] += weight * v[i];
        }
    }
}

void quantize_decode(const std::vector<float> & input, std::vector<float> & turbo,
                     std::vector<float> & q4, std::vector<float> & q8) {
    std::vector<block_turbo4_0> turbo_blocks(N_ROWS * N_BLOCKS_TURBO4);
    std::vector<block_q4_0> q4_blocks(N_ROWS * N_BLOCKS_Q4);
    std::vector<block_q8_0> q8_blocks(N_ROWS * N_BLOCKS_Q8);

    for (int row = 0; row < N_ROWS; ++row) {
        const float * src = input.data() + row * D;
        quantize_row_turbo4_0_ref(src, turbo_blocks.data() + row * N_BLOCKS_TURBO4, D);
        quantize_row_q4_0_ref(src, q4_blocks.data() + row * N_BLOCKS_Q4, D);
        quantize_row_q8_0_ref(src, q8_blocks.data() + row * N_BLOCKS_Q8, D);

        dequantize_row_turbo4_0(turbo_blocks.data() + row * N_BLOCKS_TURBO4, turbo.data() + row * D, D);
        dequantize_row_q4_0(q4_blocks.data() + row * N_BLOCKS_Q4, q4.data() + row * D, D);
        dequantize_row_q8_0(q8_blocks.data() + row * N_BLOCKS_Q8, q8.data() + row * D, D);
    }
}

void report_attention(const char * name, const std::vector<float> & q,
                      const std::vector<float> & k, const std::vector<float> & v,
                      const std::vector<float> & ref_out, bool turbo_domain,
                      bool rotate_q) {
    std::vector<float> q_used = q;
    if (turbo_domain && rotate_q) {
        turbo_wht_row(q_used.data(), false);
    }

    std::vector<float> out(D);
    softmax_attention(q_used.data(), k, v, out.data());
    if (turbo_domain) {
        turbo_wht_row(out.data(), true);
    }

    stats value;
    add_stats(value, ref_out.data(), out.data(), D);
    std::printf("  attention %-7s rel_l2=%.7g max_abs=%.7g\n", name,
                relative_l2(value), value.max_abs);

}

void run_case(const char * name, bool heavy_tail) {
    std::vector<float> k(N_ROWS * D);
    std::vector<float> v(N_ROWS * D);
    fill_rows(k, heavy_tail ? 0x4B564B31u : 0x4B564B30u, heavy_tail);
    fill_rows(v, heavy_tail ? 0x4B565631u : 0x4B565630u, heavy_tail);

    std::mt19937 rng(heavy_tail ? 0x51554C31u : 0x51554C30u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> q(D);
    for (float & value : q) {
        value = normal(rng);
    }
    if (heavy_tail) {
        q[29] = 10.0f * normal(rng);
        q[211] = 10.0f * normal(rng);
    }

    std::vector<float> k_turbo(N_ROWS * D), v_turbo(N_ROWS * D);
    std::vector<float> k_q4(N_ROWS * D), v_q4(N_ROWS * D);
    std::vector<float> k_q8(N_ROWS * D), v_q8(N_ROWS * D);
    quantize_decode(k, k_turbo, k_q4, k_q8);
    quantize_decode(v, v_turbo, v_q4, v_q8);

    std::vector<float> k_turbo_recon = k_turbo;
    std::vector<float> v_turbo_recon = v_turbo;
    stats turbo_raw;
    stats turbo_recon;
    stats q4_error;
    stats q8_error;
    for (int row = 0; row < N_ROWS; ++row) {
        const float * k_ref = k.data() + row * D;
        const float * v_ref = v.data() + row * D;
        const float * k_turbo_row = k_turbo.data() + row * D;
        const float * v_turbo_row = v_turbo.data() + row * D;

        add_stats(turbo_raw, k_ref, k_turbo_row, D);
        add_stats(turbo_raw, v_ref, v_turbo_row, D);
        add_stats(q4_error, k_ref, k_q4.data() + row * D, D);
        add_stats(q4_error, v_ref, v_q4.data() + row * D, D);
        add_stats(q8_error, k_ref, k_q8.data() + row * D, D);
        add_stats(q8_error, v_ref, v_q8.data() + row * D, D);

        turbo_wht_row(k_turbo_recon.data() + row * D, true);
        turbo_wht_row(v_turbo_recon.data() + row * D, true);
        add_stats(turbo_recon, k_ref, k_turbo_recon.data() + row * D, D);
        add_stats(turbo_recon, v_ref, v_turbo_recon.data() + row * D, D);
    }

    std::printf("\ncase %s\n", name);
    std::printf("  values turbo4(raw cache domain) rel_l2=%.7g max_abs=%.7g\n",
                relative_l2(turbo_raw), turbo_raw.max_abs);
    std::printf("  values turbo4(inverse WHT)     rel_l2=%.7g max_abs=%.7g\n",
                relative_l2(turbo_recon), turbo_recon.max_abs);
    std::printf("  values q4_0                    rel_l2=%.7g max_abs=%.7g\n",
                relative_l2(q4_error), q4_error.max_abs);
    std::printf("  values q8_0                    rel_l2=%.7g max_abs=%.7g\n",
                relative_l2(q8_error), q8_error.max_abs);

    std::vector<float> reference(D);
    softmax_attention(q.data(), k, v, reference.data());
    report_attention("q4_0", q, k_q4, v_q4, reference, false, false);
    report_attention("q8_0", q, k_q8, v_q8, reference, false, false);
    report_attention("turbo4", q, k_turbo, v_turbo, reference, true, true);
    report_attention("no_q_wht", q, k_turbo, v_turbo, reference, true, false);
}

} // namespace

int main() {
    std::printf("turbo4 KV numerical audit: head_dim=%d rows=%d blocks/head=%d\n", D, N_ROWS, N_BLOCKS_TURBO4);
    std::printf("turbo4 centroids:");
    for (float centroid : turbo4_centroids) {
        std::printf(" %.7g", centroid);
    }
    std::printf("\n");
    run_case("gaussian", false);
    run_case("heavy-tail", true);
    return 0;
}
