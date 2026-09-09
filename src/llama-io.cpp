#include "llama-io.h"

#include "ggml-backend.h"
#include "llama-impl.h"

#include <cmath>
#include <cstdlib>
#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

// Generic re-encoding path: pull the rows to host, quantize, write the bytes.
// Only the byte-counting writer overrides this; every real writer wants exactly
// this behaviour, and doing it here keeps the encoded size definition in one
// place so the sizing pass and the writing pass can never disagree.
void llama_io_write_i::write_tensor_as(ggml_tensor * tensor, size_t offset, size_t n_rows, size_t n_per_row, ggml_type dst_type) {
    if (dst_type == tensor->type) {
        write_tensor(tensor, offset, n_rows * ggml_row_size(tensor->type, n_per_row));
        return;
    }

    // Only f32 sources are re-encoded; recurrent state is f32 by construction.
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);

    const size_t src_size = n_rows * ggml_row_size(GGML_TYPE_F32, n_per_row);
    const size_t dst_size = n_rows * ggml_row_size(dst_type, n_per_row);

    std::vector<uint8_t> src(src_size);
    std::vector<uint8_t> dst(dst_size);

    ggml_backend_tensor_get(tensor, src.data(), offset, src_size);

    const size_t n = ggml_quantize_chunk(dst_type, (const float *) src.data(), dst.data(),
                                         0, n_rows, n_per_row, nullptr);
    GGML_ASSERT(n == dst_size);

    // Opt-in fidelity probe: round-trip the block we just encoded and report the
    // error actually injected into this state, on real data. Off by default --
    // it costs a dequantize per tensor. This is the number that decides whether
    // an encoding is safe to restore from, so it is measured rather than
    // assumed, and it works for any codec the type registry can decode.
    static const bool probe = [] {
        const char * e = getenv("LLAMA_STATE_CODEC_PROBE");
        return e != nullptr && e[0] == '1';
    }();
    if (probe) {
        const ggml_type_traits * qt = ggml_get_type_traits(dst_type);
        if (qt != nullptr && qt->to_float != nullptr) {
            std::vector<float> back(n_rows * n_per_row);
            for (size_t ir = 0; ir < n_rows; ++ir) {
                qt->to_float((const char *) dst.data() + ir * ggml_row_size(dst_type, n_per_row),
                             back.data() + ir * n_per_row, (int64_t) n_per_row);
            }
            const float * ref = (const float *) src.data();
            double se = 0.0, ss = 0.0, amax = 0.0;
            for (size_t i = 0; i < back.size(); ++i) {
                const double d = (double) back[i] - (double) ref[i];
                se += d * d;
                ss += (double) ref[i] * (double) ref[i];
                amax = std::max(amax, std::fabs(d));
            }
            LLAMA_LOG_INFO("state codec probe: %s rows=%zu n_per_row=%zu rel_rms=%.6f max_abs_err=%.6g rms_signal=%.6g\n",
                          ggml_type_name(dst_type), n_rows, n_per_row,
                          ss > 0.0 ? std::sqrt(se / ss) : 0.0, amax,
                          std::sqrt(ss / (double) back.size()));
        }
    }

    write(dst.data(), dst_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

void llama_io_read_i::read_tensor_as(ggml_tensor * tensor, size_t offset, size_t n_rows, size_t n_per_row, ggml_type src_type) {
    if (src_type == tensor->type) {
        read_tensor(tensor, offset, n_rows * ggml_row_size(tensor->type, n_per_row));
        return;
    }

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);

    const size_t src_size = n_rows * ggml_row_size(src_type, n_per_row);
    const size_t dst_size = n_rows * ggml_row_size(GGML_TYPE_F32, n_per_row);

    std::vector<uint8_t> src(src_size);
    std::vector<uint8_t> dst(dst_size);

    read(src.data(), src_size);

    const ggml_type_traits * qt = ggml_get_type_traits(src_type);
    GGML_ASSERT(qt->to_float != nullptr);
    for (size_t ir = 0; ir < n_rows; ++ir) {
        qt->to_float((const char *) src.data() + ir * ggml_row_size(src_type, n_per_row),
                     (float *) dst.data() + ir * n_per_row,
                     (int64_t) n_per_row);
    }

    ggml_backend_tensor_set(tensor, dst.data(), offset, dst_size);
}
