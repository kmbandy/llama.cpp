#include "pipe-prefetch-hints.h"

#include <algorithm>
#include <cmath>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace pipe_expert_dispatcher;

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_u16(std::ofstream & output, uint16_t value) {
    output.put((char) value);
    output.put((char) (value >> 8));
}

void write_u32(std::ofstream & output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.put((char) (value >> shift));
    }
}

void write_u64(std::ofstream & output, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        output.put((char) (value >> shift));
    }
}

struct table_entry {
    uint16_t expert;
    uint32_t count;
};

void write_row(std::ofstream & output, uint32_t total, const std::vector<table_entry> & entries) {
    write_u32(output, total);
    write_u16(output, (uint16_t) entries.size());
    write_u16(output, 0);
    for (const table_entry & entry : entries) {
        write_u16(output, entry.expert);
        write_u32(output, entry.count);
    }
}

void write_test_table(const fs::path & path) {
    std::ofstream output(path, std::ios::binary);
    require((bool) output, "failed to create n-gram test table");
    output.write("WPNGRAM\0", 8);
    write_u32(output, 1);
    write_u32(output, 2);
    write_u32(output, 4);
    write_u32(output, 4);
    write_u64(output, 2);

    write_row(output, 100,
              {
                  { 3, 40 },
                  { 2, 30 },
                  { 1, 20 },
                  { 0, 10 }
    });
    write_row(output, 10,
              {
                  { 0, 4 },
                  { 1, 3 },
                  { 2, 2 },
                  { 3, 1 }
    });

    write_u32(output, 10);
    write_u16(output, 0);
    write_u16(output, 0);
    write_row(output, 100,
              {
                  { 0, 60 },
                  { 1, 20 }
    });

    write_u32(output, 20);
    write_u16(output, 0);
    write_u16(output, 0);
    write_row(output, 10,
              {
                  { 2, 9 },
                  { 1, 1 }
    });
}

void test_router2_per_token_union() {
    // Token 0 wants e0 then e2; token 1 wants e1 then e2. Max-pool-then-top-2
    // kept {0,1} and dropped e2, which is in BOTH tokens' top-2 -- the set the
    // target actually dispatches. Per-token top-2 union is {0,1,2}.
    const float weights[] = {
        4.0f, 0.0f, 0.0f, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f,
    };
    const float bias[]        = { 0.0f, 0.0f, 0.0f, 0.0f };
    const float activations[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const std::vector<int32_t> top = router2_top_experts(weights, bias, activations, 2, 4, 2, 2);
    require(top == std::vector<int32_t>({ 0, 1, 2 }),
            "router2 must union each token's top-M, not max-pool then top-M");
}

void test_router2_confidence_gate() {
    // 4 experts, 2 dims. Expert 0 is strongly aligned with the activation,
    // expert 1 weakly, experts 2 and 3 not at all -- a PEAKED layer.
    const float weights[] = {
        8.0f, 0.0f,   // e0 . h = 8
        2.0f, 0.0f,   // e1 . h = 2
        0.0f, 0.0f,   // e2 . h = 0
        0.0f, 0.0f,   // e3 . h = 0
    };
    const float bias[]        = { 0.0f, 0.0f, 0.0f, 0.0f };
    const float activations[] = { 1.0f, 0.0f };

    const std::vector<int32_t> ungated =
        router2_top_experts(weights, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.0f);
    require(ungated.size() == 4, "ungated router2 must still emit the full top-M");

    // All-or-nothing: best expert clears 0.2, so the WHOLE top-M is emitted.
    // Truncating to only the ids above the floor leaves a layer partially
    // covered, which still demand-pages.
    const std::vector<int32_t> gated =
        router2_top_experts(weights, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.2f);
    require(gated == ungated, "peaked layer that clears the floor must emit the full top-M");
    require(std::is_sorted(gated.begin(), gated.end()),
            "gated router2 output must stay ascending for the wire");

    // A FLAT layer -- the router is undecided, every expert scores the same, so
    // no expert can clear a floor above 1/n_expert. Emitting nothing here is the
    // entire point: an undecided layer is where speculative reads are wasted.
    const float flat_w[] = {
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
    };
    const std::vector<int32_t> flat =
        router2_top_experts(flat_w, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.5f);
    require(flat.empty(), "confidence gate emitted experts on a layer with no signal");

    const std::vector<int32_t> flat_ungated =
        router2_top_experts(flat_w, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.0f);
    require(flat_ungated.size() == 4, "flat layer must emit when the gate is off");
}

void test_ngram_format_and_scoring(const fs::path & path) {
    write_test_table(path);
    const ngram_hint_table table(path.string());
    require(table.n_layers() == 2, "n-gram layer count is wrong");
    require(table.n_experts() == 4, "n-gram expert count is wrong");
    require(table.row_width() == 4, "n-gram row width is wrong");
    require(table.row_count() == 2, "n-gram token row count is wrong");

    const int32_t tokens[] = { 10, 20 };
    require(table.top_experts(tokens, 2, 0, 2) == std::vector<int32_t>({ 0, 2 }),
            "n-gram rows were not normalized per token before summing");

    const int32_t missing[] = { 999 };
    require(table.top_experts(missing, 1, 0, 2) == std::vector<int32_t>({ 2, 3 }),
            "n-gram popularity fallback is wrong");
}

}  // namespace

void test_pscore(const fs::path & path) {
    {
        std::ofstream out(path);
        out << "# c\nbias -1.0\nrank 0 1 0.5\nprob 0.1 2 1.5\nmargin 1 99 9\n";
    }
    pscore_model m;
    std::string  err;
    require(parse_pscore_file(path.string(), m, &err) && m.buckets.size() == 3, "pscore parse");
    pscore_features f;
    f.v[PSF_RANK] = 0; f.v[PSF_PROB] = 0.1f; f.v[PSF_MARGIN] = 0.5f;
    const double p = pscore_eval(m, f);   // z = -1 + 0.5 + 1.5 = 1 (hi is exclusive, lo inclusive)
    require(std::abs(p - 1.0 / (1.0 + std::exp(-1.0))) < 1e-9, "pscore eval");
    f.v[PSF_RANK] = 1; f.v[PSF_PROB] = 2.0f;
    require(std::abs(pscore_eval(m, f) - 1.0 / (1.0 + std::exp(1.0))) < 1e-9, "pscore bucket bounds");
    {   // age factor: buckets parse, 99 = never, lo inclusive / hi exclusive; files without age still work
        std::ofstream out(path);
        out << "bias 0\nage 1 2 -1.0\nage 16 99 2.0\nage 99 100 0.25\nrank 0 1 0.5\n";
    }
    pscore_model ma;
    require(parse_pscore_file(path.string(), ma, &err) && ma.buckets.size() == 4, "pscore age parse");
    pscore_features fa;
    fa.v[PSF_AGE] = 1;
    require(std::abs(pscore_eval(ma, fa) - 1.0 / (1.0 + std::exp(-(-1.0 + 0.5)))) < 1e-9, "pscore age=1");
    fa.v[PSF_AGE] = 98;
    require(std::abs(pscore_eval(ma, fa) - 1.0 / (1.0 + std::exp(-(2.0 + 0.5)))) < 1e-9, "pscore age=98");
    fa.v[PSF_AGE] = 99;
    require(std::abs(pscore_eval(ma, fa) - 1.0 / (1.0 + std::exp(-(0.25 + 0.5)))) < 1e-9, "pscore age=99 never");
    fa.v[PSF_AGE] = 5;
    require(std::abs(pscore_eval(ma, fa) - 1.0 / (1.0 + std::exp(-0.5))) < 1e-9, "pscore age gap bucket");
    require(parse_pscore_file(path.string(), m, &err) && m.buckets.size() == 4, "pscore reparse");
    {   // old-format file (no age) is unaffected by the new feature value
        std::ofstream out(path);
        out << "bias 0\nrank 0 1 0.5\n";
    }
    require(parse_pscore_file(path.string(), m, &err), "pscore old file parses");
    pscore_features fo;
    fo.v[PSF_AGE] = 99;
    require(std::abs(pscore_eval(m, fo) - 1.0 / (1.0 + std::exp(-0.5))) < 1e-9, "pscore old file ignores age");
    {
        std::ofstream out(path);
        out << "bias 0\nbogus 0 1 1\n";
    }
    require(!parse_pscore_file(path.string(), m, &err) && !err.empty(), "pscore must reject bad factor");
}

// Per-distance margin / top-M overrides (WP_HINT_ROUTER2_MARGIN_BY_D / _TOPM_BY_D).
// The env is parsed once on first use, so main() sets it before anything calls these.
void test_router2_per_distance_knobs() {
    const float base = router2_margin();
    require(std::abs(router2_margin_for_d(1) - 0.3f) < 1e-6f, "margin_by_d element 1");
    require(router2_margin_for_d(2) == base, "empty element keeps the default margin");
    require(std::abs(router2_margin_for_d(3) - 0.5f) < 1e-6f, "margin_by_d element 3");
    require(router2_margin_for_d(4) == 0.0f, "explicit 0 turns the gate off for that distance");
    require(router2_margin_for_d(5) == base && router2_margin_for_d(0) == base, "outside the list -> default");
    require(std::abs(router2_margin_late_for_d(3) - 0.5f) < 1e-6f, "late margin follows the per-d margin when unset");
    require(router2_topm_for_d(1, 6) == 6 && router2_topm_for_d(2, 6) == 10, "topm_by_d override");
    require(router2_topm_for_d(3, 6) == PREFETCH_HINT_MAX_EXPERTS, "topm_by_d clamps to the wire cap");
    require(router2_topm_for_d(4, 6) == 6, "past the list -> base top-M");
}

int main(int argc, char ** argv) {
    if (argc == 2) {
        const ngram_hint_table table(argv[1]);
        require(table.n_layers() == 43, "built DS4 table has the wrong layer count");
        require(table.n_experts() == 256, "built DS4 table has the wrong expert count");
        require(table.row_width() == 16, "built DS4 table has the wrong row width");
        require(table.row_count() > 0, "built DS4 table has no token rows");
        return 0;
    }
    require(argc == 1, "usage: test-wp-prefetch-hints [table]");
    setenv("WP_HINT_ROUTER2_MARGIN_BY_D", "0.3,,0.5,0", 1);
    setenv("WP_HINT_ROUTER2_TOPM_BY_D", "-,10,99", 1);
    unsetenv("WP_HINT_ROUTER2_MARGIN_LATE");
    const fs::path path = fs::temp_directory_path() / ("wp-prefetch-hints-" + std::to_string((long) getpid()) + ".bin");
    try {
        test_router2_per_distance_knobs();
        test_router2_per_token_union();
        test_router2_confidence_gate();
        test_ngram_format_and_scoring(path);
        test_pscore(path);
        std::error_code ignored;
        fs::remove(path, ignored);
        return 0;
    } catch (...) {
        std::error_code ignored;
        fs::remove(path, ignored);
        throw;
    }
}
