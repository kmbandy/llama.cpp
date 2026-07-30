// Round-trip test for wp-stage-split (Phase 1b of the cross-machine pipeline
// spec): build a tiny synthetic GGUF, split it into two stage files, re-read
// them with the gguf reader, and verify the tensor set, the tensor DATA bytes,
// and the pipeline.layer_first/last metadata.
//
// Links against ggml (gguf + ggml-core). Manual build against the in-tree
// prebuilt libs:
//   g++ -std=c++17 -I include -I ggml/include -I src -I . \
//       tests/test-wp-stage-split.cpp tools/wp-stage-split/wp-stage-split-lib.cpp \
//       src/llama-pipeline.cpp -L bin -lggml -lggml-base -Wl,-rpath,$PWD/bin -o /tmp/t-split

#include "wp-stage-split-lib.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

namespace {

constexpr int32_t N_LAYER = 4;
constexpr int64_t N_EMBD  = 8;

// deterministic payload so the split files can be checked byte-for-byte
float payload_value(const std::string & name, int64_t i) {
    uint32_t h = 0;
    for (char c : name) {
        h = h*31u + (uint32_t) c;
    }
    return (float) ((int) (h % 1000) - 500) + (float) i;
}

void add_tensor_f32(gguf_context * ctx, ggml_context * ctx_meta, const char * name,
                    int64_t ne0, int64_t ne1, std::vector<float> & data) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    gguf_add_tensor(ctx, t);
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        data.push_back(payload_value(name, i));
    }
}

void set_source_metadata(gguf_context * ctx, bool has_mtp = true) {
    gguf_set_val_str(ctx, "general.architecture", "glm-dsa");
    gguf_set_val_u32(ctx, "glm-dsa.attention.head_count", 2);
    gguf_set_val_str(ctx, "general.name", "wp-stage-split-test");
    gguf_set_val_u32(ctx, "glm-dsa.block_count", N_LAYER + (has_mtp ? 1 : 0));
    if (has_mtp) {
        gguf_set_val_u32(ctx, "glm-dsa.nextn_predict_layers", 1);
    }
}

// write a synthetic 4-layer model with token_embd + output_norm + output
void write_source_gguf(const std::string &        path,
                       std::vector<std::string> & names,
                       std::vector<float> &       data,
                       int32_t                    split_no    = -1,
                       int32_t                    split_count = 1,
                       bool                       has_mtp      = true) {
    gguf_context *   ctx = gguf_init_empty();
    ggml_init_params mp = { /*.mem_size =*/64 * ggml_tensor_overhead(), /*.mem_buffer =*/nullptr, /*.no_alloc =*/true };
    ggml_context *   ctx_meta = ggml_init(mp);

    set_source_metadata(ctx, has_mtp);

    add_tensor_f32(ctx, ctx_meta, "token_embd.weight",   N_EMBD, 16, data); names.push_back("token_embd.weight");
    for (int32_t il = 0; il < N_LAYER; ++il) {
        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        add_tensor_f32(ctx, ctx_meta, name, N_EMBD, N_EMBD, data); names.push_back(name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
        add_tensor_f32(ctx, ctx_meta, name, N_EMBD, 4*N_EMBD, data); names.push_back(name);
    }
    if (has_mtp) {
        // NextN/MTP tensors live at blk.N_LAYER, past the real layers
        add_tensor_f32(ctx, ctx_meta, "blk.4.nextn_eh_proj.weight", N_EMBD, N_EMBD, data);
        names.push_back("blk.4.nextn_eh_proj.weight");
    }
    add_tensor_f32(ctx, ctx_meta, "output_norm.weight",  N_EMBD, 1,  data); names.push_back("output_norm.weight");
    add_tensor_f32(ctx, ctx_meta, "output.weight",       N_EMBD, 16, data); names.push_back("output.weight");

    if (split_no >= 0) {
        gguf_set_val_u16(ctx, "split.no", (uint16_t) split_no);
        gguf_set_val_u16(ctx, "split.count", (uint16_t) split_count);
        gguf_set_val_i32(ctx, "split.tensors.count", (int32_t) names.size());
    }

    // metadata, then data in info order (same idiom as the splitter)
    std::ofstream fout(path, std::ios::binary);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    std::vector<uint8_t> meta(gguf_get_meta_size(ctx));
    gguf_get_meta_data(ctx, meta.data());
    fout.write((const char *) meta.data(), meta.size());

    size_t pos = 0;
    for (const std::string & name : names) {
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, name.c_str());
        const size_t n_bytes = ggml_nbytes(t);
        fout.write((const char *) (data.data() + pos/sizeof(float)), n_bytes);
        pos += n_bytes;
        // pad to GGUF_DEFAULT_ALIGNMENT
        const size_t pad = GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes;
        for (size_t i = 0; i < pad; ++i) {
            char z = 0;
            fout.write(&z, 1);
        }
    }
    fout.close();

    ggml_free(ctx_meta);
    gguf_free(ctx);
}

void write_metadata_only_shard(const std::string & path, int32_t split_count, int32_t n_tensors) {
    gguf_context * ctx = gguf_init_empty();
    set_source_metadata(ctx);
    gguf_set_val_u16(ctx, "split.no", 0);
    gguf_set_val_u16(ctx, "split.count", (uint16_t) split_count);
    gguf_set_val_i32(ctx, "split.tensors.count", n_tensors);

    std::ofstream fout(path, std::ios::binary);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    std::vector<uint8_t> meta(gguf_get_meta_size(ctx));
    gguf_get_meta_data(ctx, meta.data());
    fout.write((const char *) meta.data(), meta.size());
    fout.close();
    gguf_free(ctx);
}

struct stage_contents {
    std::set<std::string> names;
    int32_t first = -1;
    int32_t last  = -1;
    int32_t n_kv  = 0;
    std::string arch;
    uint32_t block_count = 0;
    bool                  has_split_no            = false;
    bool                  has_split_count         = false;
    bool                  has_split_tensors_count = false;
};

stage_contents read_stage(const std::string & path, const std::vector<float> & src_data,
                          const std::vector<std::string> & src_names) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params ip = {/*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta};
    gguf_context * ctx = gguf_init_from_file(path.c_str(), ip);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to re-read stage file " + path);
    }

    stage_contents sc;
    sc.n_kv = (int32_t) gguf_get_n_kv(ctx);
    const int64_t kf = gguf_find_key(ctx, "pipeline.layer_first");
    const int64_t kl = gguf_find_key(ctx, "pipeline.layer_last");
    CHECK(kf >= 0 && kl >= 0);
    sc.first = gguf_get_val_i32(ctx, kf);
    sc.last  = gguf_get_val_i32(ctx, kl);

    // The band the splitter WROTE must be exactly the band the pipeline tool
    // READS back before loading. These are the two halves of the same
    // contract: if they ever disagree, a stage launched without an explicit
    // --pipeline-layers silently runs the wrong band.
    int32_t peek_first = -1;
    int32_t peek_last  = -1;
    CHECK(llama_pipeline_peek_band_from_file(path.c_str(), &peek_first, &peek_last));
    CHECK(peek_first == sc.first);
    CHECK(peek_last  == sc.last);
    sc.arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    sc.block_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "glm-dsa.block_count"));
    sc.has_split_no            = gguf_find_key(ctx, "split.no") >= 0;
    sc.has_split_count         = gguf_find_key(ctx, "split.count") >= 0;
    sc.has_split_tensors_count = gguf_find_key(ctx, "split.tensors.count") >= 0;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        sc.names.insert(name);

        // verify the payload bytes round-tripped
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        const int64_t n = ggml_nelements(t);
        std::vector<float> got(n);
        const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        std::ifstream fin(path, std::ios::binary);
        fin.seekg(offset);
        fin.read((char *) got.data(), n*sizeof(float));
        CHECK(fin.good());
        for (int64_t j = 0; j < n; ++j) {
            if (got[j] != payload_value(name, j)) {
                std::fprintf(stderr, "FAIL %s:%d: payload mismatch in %s at element %lld\n",
                             __FILE__, __LINE__, name, (long long) j);
                ++g_failed;
                break;
            }
        }
    }
    GGML_UNUSED(src_data);
    GGML_UNUSED(src_names);

    gguf_free(ctx);
    ggml_free(ctx_meta);
    return sc;
}

std::set<std::string> expected_stage(
        int32_t first, int32_t last, bool head, bool tail, bool has_mtp = true) {
    std::set<std::string> exp;
    for (int32_t il = first; il <= last; ++il) {
        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        exp.insert(name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
        exp.insert(name);
    }
    if (head) exp.insert("token_embd.weight");
    if (tail || (head && has_mtp)) exp.insert("output_norm.weight");
    if (tail || (head && has_mtp)) exp.insert("output.weight");
    if (head && has_mtp) exp.insert("blk.4.nextn_eh_proj.weight");
    return exp;
}

} // namespace

int main() {
    const std::string dir  = "/tmp/wp-stage-split-test";
    const std::string src  = dir + "/src.gguf";
    const std::string head = dir + "/head.gguf";
    const std::string tail = dir + "/tail.gguf";
    const std::string split_first  = dir + "/src-split-00001-of-00002.gguf";
    const std::string split_second = dir + "/src-split-00002-of-00002.gguf";
    const std::string split_head   = dir + "/split-head.gguf";
    const std::string split_tail   = dir + "/split-tail.gguf";
    const std::string no_mtp       = dir + "/no-mtp.gguf";

    std::string cmd = "mkdir -p " + dir + " && rm -f " + src + " " + head + " " + tail + " " + split_first + " " +
                      split_second + " " + split_head + " " + split_tail + " " + no_mtp;
    if (std::system(cmd.c_str()) != 0) {
        std::fprintf(stderr, "setup failed\n");
        return 1;
    }

    std::vector<std::string> src_names;
    std::vector<float>     src_data;
    write_source_gguf(src, src_names, src_data);

    // split head [0,1] and tail [2,3] of the 4-layer model
    const wp_stage_split::result r_head = wp_stage_split::split_stage(src, head, 0, 1, false);
    const wp_stage_split::result r_tail = wp_stage_split::split_stage(src, tail, 2, 3, false);

    CHECK(r_head.n_layer == N_LAYER && r_tail.n_layer == N_LAYER);
    CHECK(r_head.n_tensors_in == (int64_t) src_names.size());
    CHECK(r_head.n_tensors_out == (int64_t) expected_stage(0, 1, true,  false).size());
    CHECK(r_tail.n_tensors_out == (int64_t) expected_stage(2, 3, false, true ).size());

    const stage_contents sc_head = read_stage(head, src_data, src_names);
    const stage_contents sc_tail = read_stage(tail, src_data, src_names);

    CHECK(sc_head.first == 0 && sc_head.last == 1);
    CHECK(sc_tail.first == 2 && sc_tail.last == 3);
    CHECK(sc_head.names == expected_stage(0, 1, true,  false));
    CHECK(sc_tail.names == expected_stage(2, 3, false, true ));

    // all KV metadata is preserved (the stage adds exactly the two band keys)
    CHECK(sc_head.arch == "glm-dsa");
    CHECK(sc_head.block_count == N_LAYER + 1);
    CHECK(sc_tail.block_count == N_LAYER + 1);

    // Layer tensors partition exactly. The output tensors are the only
    // intentionally duplicated globals.
    {
        const std::set<std::string> expected_duplicates = {
            "output.weight",
            "output_norm.weight",
        };
        std::set<std::string> uni = sc_head.names;
        uni.insert(sc_tail.names.begin(), sc_tail.names.end());
        CHECK(uni.size() == src_names.size());

        std::set<std::string> duplicates;
        for (const auto & n : sc_head.names) {
            if (sc_tail.names.count(n)) duplicates.insert(n);
        }
        CHECK(duplicates == expected_duplicates);
        for (const std::string & name : duplicates) {
            std::printf("intentionally duplicated global: %s\n", name.c_str());
        }

        for (const std::string & name : src_names) {
            const int owners = (int) sc_head.names.count(name) + (int) sc_tail.names.count(name);
            if (llama_pipeline_tensor_block_index(name.c_str()) >= 0) {
                CHECK(owners == 1);
            } else if (expected_duplicates.count(name)) {
                CHECK(owners == 2);
            } else {
                CHECK(owners == 1);
            }
        }
    }

    // split input with a metadata-only first shard
    {
        std::vector<std::string> split_names;
        std::vector<float>       split_data;
        write_source_gguf(split_second, split_names, split_data, 1, 2);
        write_metadata_only_shard(split_first, 2, (int32_t) split_names.size());

        const wp_stage_split::result r_split_head = wp_stage_split::split_stage(split_first, split_head, 0, 1, false);
        const wp_stage_split::result r_split_tail = wp_stage_split::split_stage(split_first, split_tail, 2, 3, false);
        CHECK(r_split_head.n_tensors_in == (int64_t) split_names.size());
        CHECK(r_split_tail.n_tensors_in == (int64_t) split_names.size());

        const stage_contents sc_split_head = read_stage(split_head, split_data, split_names);
        const stage_contents sc_split_tail = read_stage(split_tail, split_data, split_names);
        CHECK(sc_split_head.names == expected_stage(0, 1, true, false));
        CHECK(sc_split_tail.names == expected_stage(2, 3, false, true));
        CHECK(!sc_split_head.has_split_no);
        CHECK(!sc_split_head.has_split_count);
        CHECK(!sc_split_head.has_split_tensors_count);
        CHECK(!sc_split_tail.has_split_no);
        CHECK(!sc_split_tail.has_split_count);
        CHECK(!sc_split_tail.has_split_tensors_count);
    }

    // A model without MTP metadata keeps output tensors tail-only.
    {
        std::vector<std::string> no_mtp_names;
        std::vector<float>       no_mtp_data;
        write_source_gguf(no_mtp, no_mtp_names, no_mtp_data, -1, 1, false);

        const wp_stage_split::result r_no_mtp_head =
            wp_stage_split::split_stage(no_mtp, "", 0, 1, true);
        const wp_stage_split::result r_no_mtp_tail =
            wp_stage_split::split_stage(no_mtp, "", 2, 3, true);
        const std::set<std::string> no_mtp_head_names(
            r_no_mtp_head.tensor_names.begin(), r_no_mtp_head.tensor_names.end());
        const std::set<std::string> no_mtp_tail_names(
            r_no_mtp_tail.tensor_names.begin(), r_no_mtp_tail.tensor_names.end());

        CHECK(no_mtp_head_names == expected_stage(0, 1, true, false, false));
        CHECK(no_mtp_tail_names == expected_stage(2, 3, false, true, false));
        std::vector<std::string> no_mtp_duplicates;
        for (const std::string & name : no_mtp_head_names) {
            if (no_mtp_tail_names.count(name)) no_mtp_duplicates.push_back(name);
        }
        CHECK(no_mtp_duplicates.empty());
    }

    // a middle band must refuse neither head nor tail tensors
    {
        const wp_stage_split::result r_mid =
            wp_stage_split::split_stage(src, dir + "/mid.gguf", 1, 2, true);
        std::set<std::string> mid_names(r_mid.tensor_names.begin(), r_mid.tensor_names.end());
        CHECK(mid_names == expected_stage(1, 2, false, false));
    }

    // invalid bands refuse loudly
    {
        bool threw = false;
        try { wp_stage_split::split_stage(src, dir + "/x.gguf", 3, 2, true); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw);
    }
    {
        bool threw = false;
        try { wp_stage_split::split_stage(src, dir + "/x.gguf", 0, 4, true); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw);
    }
    // existing output refuses to overwrite
    {
        bool threw = false;
        try { wp_stage_split::split_stage(src, head, 0, 1, false); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw);
    }

    if (g_failed == 0) {
        std::printf("OK: all wp-stage-split tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s) failed\n", g_failed);
    return 1;
}
