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

// write a synthetic 4-layer model with token_embd + output_norm + output
void write_source_gguf(const std::string & path, std::vector<std::string> & names,
                       std::vector<float> & data) {
    gguf_context * ctx  = gguf_init_empty();
    ggml_init_params mp = {/*.mem_size =*/ 64*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true};
    ggml_context * ctx_meta = ggml_init(mp);

    gguf_set_val_str(ctx, "general.architecture", "glm-dsa");
    gguf_set_val_u32(ctx, "glm-dsa.attention.head_count", 2);
    gguf_set_val_str(ctx, "general.name", "wp-stage-split-test");

    // block_count includes the NextN layer, as in glm-dsa
    gguf_set_val_u32(ctx, "glm-dsa.block_count", N_LAYER + 1);
    gguf_set_val_u32(ctx, "glm-dsa.nextn_predict_layers", 1);

    add_tensor_f32(ctx, ctx_meta, "token_embd.weight",   N_EMBD, 16, data); names.push_back("token_embd.weight");
    for (int32_t il = 0; il < N_LAYER; ++il) {
        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        add_tensor_f32(ctx, ctx_meta, name, N_EMBD, N_EMBD, data); names.push_back(name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
        add_tensor_f32(ctx, ctx_meta, name, N_EMBD, 4*N_EMBD, data); names.push_back(name);
    }
    // NextN/MTP tensors live at blk.N_LAYER, past the real layers
    add_tensor_f32(ctx, ctx_meta, "blk.4.nextn_eh_proj.weight", N_EMBD, N_EMBD, data);
    names.push_back("blk.4.nextn_eh_proj.weight");
    add_tensor_f32(ctx, ctx_meta, "output_norm.weight",  N_EMBD, 1,  data); names.push_back("output_norm.weight");
    add_tensor_f32(ctx, ctx_meta, "output.weight",       N_EMBD, 16, data); names.push_back("output.weight");

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

struct stage_contents {
    std::set<std::string> names;
    int32_t first = -1;
    int32_t last  = -1;
    int32_t n_kv  = 0;
    std::string arch;
    uint32_t block_count = 0;
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
    sc.arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    sc.block_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "glm-dsa.block_count"));

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

std::set<std::string> expected_stage(int32_t first, int32_t last, bool head, bool tail) {
    std::set<std::string> exp;
    for (int32_t il = first; il <= last; ++il) {
        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);
        exp.insert(name);
        std::snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", il);
        exp.insert(name);
    }
    if (head) exp.insert("token_embd.weight");
    if (tail) exp.insert("output_norm.weight");
    if (tail) exp.insert("output.weight");
    if (tail) exp.insert("blk.4.nextn_eh_proj.weight"); // NextN belongs to the tail
    return exp;
}

} // namespace

int main() {
    const std::string dir  = "/tmp/wp-stage-split-test";
    const std::string src  = dir + "/src.gguf";
    const std::string head = dir + "/head.gguf";
    const std::string tail = dir + "/tail.gguf";

    std::string cmd = "mkdir -p " + dir + " && rm -f " + src + " " + head + " " + tail;
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

    // the two stages together cover every source tensor exactly once,
    // except output_norm/output which only the tail owns -- i.e. union ==
    // source set and the intersection is empty
    {
        std::set<std::string> uni = sc_head.names;
        uni.insert(sc_tail.names.begin(), sc_tail.names.end());
        CHECK(uni.size() == src_names.size());
        std::vector<std::string> inter;
        for (const auto & n : sc_head.names) {
            if (sc_tail.names.count(n)) inter.push_back(n);
        }
        CHECK(inter.empty());
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
