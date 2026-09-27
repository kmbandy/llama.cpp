// system1-rows: batched hidden-state row extraction for mneme's System-1 labeller.
//
// Reads JSONL on stdin -- one "entry" per line:
//   {"id": str, "state": [int], "branches": [{"ids": [int], "outs": [int]}]}
// "outs" are offsets into that branch's own "ids" array.
//
// For each entry, the tokens in "state" are decoded first (the shared prefix / prompt
// context), then each branch's "ids" are decoded as a continuation of that state. Only
// the positions named in "outs" are extracted as hidden-state rows (via the
// embd_sparse_outputs context option), in position order.
//
// Entries are grouped by state length (up to --max-g per group, within the context's
// token budget). Each group decodes its states together on sequences 0..G-1, then for
// each "question slot" (branch index) copies every sequence's state to a scratch
// sequence (G+s) and decodes all of that slot's branches in one call. This is the
// batching scheme validated in the mneme system1 spike (kev_rows2.cpp).
//
// See tools/system1-rows brief: .superpowers/sdd/2026-09-27-system1-2b-plan/task-1-brief.md
#include "llama.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

// ---------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------

struct Args {
    std::string model;
    std::string lora;      // empty = none
    std::string out;
    uint32_t    n_ctx           = 8192;
    uint32_t    n_ubatch        = 2048;
    int32_t     max_g           = 5;
    int64_t     vram_margin_mib = 512;
    int32_t     n_threads       = 4;
    bool        no_copy         = false;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model PATH --out PATH [options] < input.jsonl\n"
        "\n"
        "  --model PATH             GGUF model to load (required)\n"
        "  --lora PATH              optional LoRA adapter to apply\n"
        "  --out PATH               output file for binary rows (required)\n"
        "  --ctx N                  context size (default 8192)\n"
        "  --ubatch N                logical/physical batch size (default 2048)\n"
        "  --max-g N                 largest group size to try (default 5)\n"
        "  --vram-margin-mib N       required free device memory after context creation (default 512)\n"
        "  --threads N               CPU threads for generation (default 4)\n"
        "  --no-copy                 control mode: decode every branch as state+branch from\n"
        "                            scratch, without the sequence-copy batching scheme\n"
        "  --help                    show this message and exit\n"
        "\n"
        "Reads JSONL on stdin: {\"id\": str, \"state\": [int], \"branches\": [{\"ids\": [int], \"outs\": [int]}]}\n"
        "Writes binary float32 rows to --out (n_embd floats per row, one row per flagged\n"
        "position, entries in input order, branches in order, positions in position order).\n"
        "Writes one JSON line per input entry to stdout: {\"id\", \"ok\", \"rows\", \"error\"}.\n",
        argv0);
}

// Parses a positive integer CLI value. Returns false (and leaves *out untouched) if the
// string isn't a valid non-negative integer.
bool parse_uint(const std::string & s, int64_t & out) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || v < 0) {
        return false;
    }
    out = v;
    return true;
}

// Prints a usage error and exits 2. Used for CLI problems -- never after the model load.
[[noreturn]] void fatal_usage(const std::string & msg) {
    fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(2);
}

// Prints an input error naming the (1-based) input line and exits 2. Never called after
// the model load has started.
[[noreturn]] void fatal_line(size_t line_no, const std::string & msg) {
    fprintf(stderr, "error: line %zu: %s\n", line_no, msg.c_str());
    std::exit(2);
}

[[noreturn]] void fatal_runtime(const std::string & msg) {
    fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

std::optional<Args> parse_args(int argc, char ** argv) {
    Args a;
    bool have_model = false;
    bool have_out   = false;

    auto need_value = [&](int & i, const char * flag) -> std::string {
        if (i + 1 >= argc) {
            fatal_usage(std::string(flag) + " requires a value");
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") {
            a.model = need_value(i, "--model");
            have_model = true;
        } else if (arg == "--lora") {
            a.lora = need_value(i, "--lora");
        } else if (arg == "--out") {
            a.out = need_value(i, "--out");
            have_out = true;
        } else if (arg == "--ctx") {
            int64_t v;
            if (!parse_uint(need_value(i, "--ctx"), v) || v == 0) {
                fatal_usage("--ctx must be a positive integer");
            }
            a.n_ctx = (uint32_t) v;
        } else if (arg == "--ubatch") {
            int64_t v;
            if (!parse_uint(need_value(i, "--ubatch"), v) || v == 0) {
                fatal_usage("--ubatch must be a positive integer");
            }
            a.n_ubatch = (uint32_t) v;
        } else if (arg == "--max-g") {
            int64_t v;
            if (!parse_uint(need_value(i, "--max-g"), v) || v == 0) {
                fatal_usage("--max-g must be a positive integer");
            }
            a.max_g = (int32_t) v;
        } else if (arg == "--vram-margin-mib") {
            int64_t v;
            if (!parse_uint(need_value(i, "--vram-margin-mib"), v)) {
                fatal_usage("--vram-margin-mib must be a non-negative integer");
            }
            a.vram_margin_mib = v;
        } else if (arg == "--threads") {
            int64_t v;
            if (!parse_uint(need_value(i, "--threads"), v) || v == 0) {
                fatal_usage("--threads must be a positive integer");
            }
            a.n_threads = (int32_t) v;
        } else if (arg == "--no-copy") {
            a.no_copy = true;
        } else {
            fatal_usage("unrecognized argument: " + arg);
        }
    }

    if (!have_model) {
        fatal_usage("--model is required");
    }
    if (!have_out) {
        fatal_usage("--out is required");
    }

    return a;
}

// ---------------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------------

struct BranchIn {
    std::vector<llama_token> ids;
    std::vector<int32_t>     outs; // offsets into ids, position order not guaranteed
};

struct EntryIn {
    std::string           id;
    std::vector<llama_token> state;
    std::vector<BranchIn>    branches;
    size_t                   line_no = 0;
};

struct EntryOut {
    std::string id;
    bool        ok    = false;
    std::string error;
    // rows[branch] = flat n_embd * outs.size() floats, in position order.
    std::vector<std::vector<float>> rows;
};

// Validates and parses every line of stdin into `entries`. Exits 2 (via fatal_line) on
// the first structural problem, naming the offending line. Never touches the model.
std::vector<EntryIn> read_and_validate_input(std::istream & in) {
    std::vector<EntryIn> entries;
    std::set<std::string> seen_ids;

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        line_no++;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue; // skip blank lines
        }

        json j;
        try {
            j = json::parse(line);
        } catch (const std::exception & e) {
            fatal_line(line_no, std::string("invalid JSON: ") + e.what());
        }

        if (!j.is_object()) {
            fatal_line(line_no, "expected a JSON object");
        }

        if (!j.contains("id") || !j["id"].is_string()) {
            fatal_line(line_no, "missing or invalid \"id\" (expected string)");
        }
        EntryIn e;
        e.id = j["id"].get<std::string>();
        e.line_no = line_no;

        if (seen_ids.count(e.id)) {
            fatal_line(line_no, "duplicate id '" + e.id + "'");
        }
        seen_ids.insert(e.id);

        if (!j.contains("state") || !j["state"].is_array()) {
            fatal_line(line_no, "missing or invalid \"state\" (expected array of int)");
        }
        for (const auto & tok : j["state"]) {
            if (!tok.is_number_integer()) {
                fatal_line(line_no, "\"state\" must contain only integers");
            }
            const int64_t v = tok.get<int64_t>();
            if (v < 0 || v > INT32_MAX) {
                fatal_line(line_no, "state token id " + std::to_string(v) +
                        " is out of range (must be a non-negative int32)");
            }
            e.state.push_back((llama_token) v);
        }
        if (e.state.empty()) {
            fatal_line(line_no, "empty state");
        }

        if (!j.contains("branches") || !j["branches"].is_array()) {
            fatal_line(line_no, "missing or invalid \"branches\" (expected array)");
        }
        for (const auto & jb : j["branches"]) {
            if (!jb.is_object() || !jb.contains("ids") || !jb["ids"].is_array() ||
                !jb.contains("outs") || !jb["outs"].is_array()) {
                fatal_line(line_no, "each branch requires \"ids\" (array of int) and \"outs\" (array of int)");
            }
            BranchIn b;
            for (const auto & tok : jb["ids"]) {
                if (!tok.is_number_integer()) {
                    fatal_line(line_no, "branch \"ids\" must contain only integers");
                }
                const int64_t v = tok.get<int64_t>();
                if (v < 0 || v > INT32_MAX) {
                    fatal_line(line_no, "branch token id " + std::to_string(v) +
                            " is out of range (must be a non-negative int32)");
                }
                b.ids.push_back((llama_token) v);
            }
            std::set<int32_t> seen_outs;
            for (const auto & off : jb["outs"]) {
                if (!off.is_number_integer()) {
                    fatal_line(line_no, "branch \"outs\" must contain only integers");
                }
                const int64_t o = off.get<int64_t>();
                if (o < 0 || o >= (int64_t) b.ids.size()) {
                    fatal_line(line_no, "outs offset " + std::to_string(o) +
                            " is outside its branch (branch has " + std::to_string(b.ids.size()) + " ids)");
                }
                if (!seen_outs.insert((int32_t) o).second) {
                    fatal_line(line_no, "duplicate outs offset " + std::to_string(o) + " in branch");
                }
                b.outs.push_back((int32_t) o);
            }
            e.branches.push_back(std::move(b));
        }

        entries.push_back(std::move(e));
    }

    return entries;
}

// Marks (only) entries that reference a token id outside [0, n_vocab) as failed, naming
// the offending id. Must run after the model is loaded (n_vocab isn't known before
// then) and before grouping/decoding, so a single bad id in one entry can't take down
// an otherwise-valid group (llama_decode would fail the whole batch otherwise).
void mark_out_of_vocab_entries(const std::vector<EntryIn> & entries, std::vector<EntryOut> & results, int32_t n_vocab) {
    for (size_t i = 0; i < entries.size(); i++) {
        if (!results[i].error.empty()) {
            continue; // already excluded for another reason
        }
        llama_token bad = -1;
        for (llama_token t : entries[i].state) {
            if (t < 0 || t >= n_vocab) {
                bad = t;
                break;
            }
        }
        if (bad == -1) {
            for (const auto & b : entries[i].branches) {
                for (llama_token t : b.ids) {
                    if (t < 0 || t >= n_vocab) {
                        bad = t;
                        break;
                    }
                }
                if (bad != -1) {
                    break;
                }
            }
        }
        if (bad != -1) {
            results[i].error = "token id " + std::to_string(bad) + " is outside the model vocab [0, " +
                    std::to_string(n_vocab) + ")";
        }
    }
}

// ---------------------------------------------------------------------------------
// Context / device sizing
// ---------------------------------------------------------------------------------

// Returns the device free-memory query for the first GPU backend device, or
// std::nullopt if none is present (e.g. a CPU-only build).
std::optional<size_t> gpu_free_bytes() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        return std::nullopt;
    }
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    return free_bytes;
}

// Tries context creation for G = max_g, max_g-1, ..., 1 and keeps the first (largest) G
// whose context creation succeeds AND leaves at least vram_margin_mib free afterwards.
// Returns {ctx, G} on success; ctx is nullptr and G is 0 if every G failed.
struct SizedContext {
    llama_context * ctx = nullptr;
    int32_t         g   = 0;
};

SizedContext make_sized_context(llama_model * model, const Args & args) {
    for (int32_t g = args.max_g; g >= 1; g--) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx               = args.n_ctx;
        cp.n_batch              = args.n_ubatch;
        cp.n_ubatch             = args.n_ubatch;
        cp.n_seq_max            = (uint32_t) (2 * g);
        cp.kv_unified           = true;
        cp.embeddings           = true;
        cp.pooling_type         = LLAMA_POOLING_TYPE_NONE;
        cp.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.embd_sparse_outputs  = true;
        cp.n_threads            = args.n_threads;
        cp.n_threads_batch      = args.n_threads;

        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr) {
            fprintf(stderr, "system1-rows: G=%d context creation failed, trying smaller G\n", g);
            continue;
        }

        const auto free_bytes = gpu_free_bytes();
        if (!free_bytes.has_value()) {
            // No GPU device to query (e.g. CPU-only build) -- can't check the margin,
            // so accept the first G that creates a context at all.
            fprintf(stderr, "system1-rows: no GPU backend device found; skipping VRAM margin check\n");
            return { ctx, g };
        }

        const int64_t free_mib = (int64_t) (*free_bytes / (1024 * 1024));
        if (free_mib >= args.vram_margin_mib) {
            return { ctx, g };
        }

        fprintf(stderr, "system1-rows: G=%d leaves only %lld MiB free (< --vram-margin-mib %lld), trying smaller G\n",
                g, (long long) free_mib, (long long) args.vram_margin_mib);
        llama_free(ctx);
    }
    return { nullptr, 0 };
}

// ---------------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------------

struct Tok {
    llama_token  t;
    llama_pos    pos;
    llama_seq_id seq;
    bool         out;
    size_t       entry;  // index into entries/results
    int32_t      branch; // -1 for state tokens
};

struct Runner {
    llama_context * ctx;
    uint32_t         n_batch;
    int32_t          n_embd;
    long             n_tok_done = 0;

    // Decodes `toks` in chunks of <= n_batch tokens. On success, flagged rows are
    // appended (in the order decoded, i.e. position order) to results[entry].rows[branch].
    // Returns false (leaving results untouched for the failing chunk onward) if any
    // llama_decode call fails.
    bool run(const std::vector<Tok> & toks, std::vector<EntryOut> & results) {
        for (size_t s = 0; s < toks.size(); s += n_batch) {
            const int n = (int) std::min(toks.size() - s, (size_t) n_batch);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; i++) {
                const Tok & k = toks[s + i];
                b.token[i]     = k.t;
                b.pos[i]       = k.pos;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = k.seq;
                b.logits[i]    = k.out ? 1 : 0;
            }
            b.n_tokens = n;

            const int rc = llama_decode(ctx, b);
            if (rc != 0) {
                llama_batch_free(b);
                return false;
            }

            for (int i = 0; i < n; i++) {
                const Tok & k = toks[s + i];
                if (!k.out) {
                    continue;
                }
                const float * e = llama_get_embeddings_ith(ctx, i);
                if (e == nullptr) {
                    llama_batch_free(b);
                    return false;
                }
                auto & row = results[k.entry].rows[k.branch];
                row.insert(row.end(), e, e + n_embd);
            }

            llama_batch_free(b);
            n_tok_done += n;
        }
        return true;
    }
};

// Marks every branch of `entries[idx]` as failed with `reason`, for the indices in
// `group` (indices into entries/results).
void mark_group_failed(const std::vector<size_t> & group, std::vector<EntryOut> & results, const std::string & reason) {
    for (size_t idx : group) {
        if (!results[idx].ok && results[idx].error.empty()) {
            results[idx].error = reason;
        }
    }
}

// Runs the batched (default) scheme: groups entries by state length (<= G per group,
// state+longest-branch tokens within --ctx), decodes each group's states together, then
// decodes each question slot's branches together via a sequence copy.
long run_batched(llama_context * ctx, const std::vector<EntryIn> & entries, std::vector<EntryOut> & results,
                  int32_t g, uint32_t n_ctx, uint32_t n_ubatch, int32_t n_embd) {
    Runner runner{ ctx, n_ubatch, n_embd };
    llama_memory_t mem = llama_get_memory(ctx);

    // Entries that cannot possibly fit (their state + their own longest branch alone
    // exceeds the context) are marked failed up front and excluded from grouping.
    // Entries already marked failed (e.g. an out-of-vocab token id) are skipped here too
    // -- their error is left as-is, and they never enter a group.
    std::vector<size_t> eligible;
    for (size_t i = 0; i < entries.size(); i++) {
        if (!results[i].error.empty()) {
            continue;
        }
        uint32_t longest_branch = 0;
        for (const auto & b : entries[i].branches) {
            longest_branch = std::max<uint32_t>(longest_branch, (uint32_t) b.ids.size());
        }
        const uint64_t need = (uint64_t) entries[i].state.size() + longest_branch;
        if (need > n_ctx) {
            results[i].error = "entry does not fit in context (state + longest branch = " +
                    std::to_string(need) + " > --ctx " + std::to_string(n_ctx) + ")";
            continue;
        }
        eligible.push_back(i);
    }

    // sort by state length so groups hold similarly-sized states (fewer wasted positions
    // when interleaving state decode across sequences)
    std::stable_sort(eligible.begin(), eligible.end(), [&](size_t a, size_t b) {
        return entries[a].state.size() < entries[b].state.size();
    });

    std::vector<std::vector<size_t>> groups;
    for (size_t idx : eligible) {
        uint32_t longest_branch = 0;
        for (const auto & b : entries[idx].branches) {
            longest_branch = std::max<uint32_t>(longest_branch, (uint32_t) b.ids.size());
        }
        const uint64_t need = entries[idx].state.size() + longest_branch;

        bool start_new = groups.empty() || (int32_t) groups.back().size() >= g;
        if (!start_new) {
            uint64_t total = need;
            for (size_t k : groups.back()) {
                uint32_t lb = 0;
                for (const auto & b : entries[k].branches) {
                    lb = std::max<uint32_t>(lb, (uint32_t) b.ids.size());
                }
                total += entries[k].state.size() + lb;
            }
            if (total > n_ctx) {
                start_new = true;
            }
        }
        if (start_new) {
            groups.push_back({});
        }
        groups.back().push_back(idx);
    }

    for (const auto & group : groups) {
        llama_memory_clear(mem, true);

        // phase 1: decode states, interleaved by position across seqs 0..group.size()-1
        std::vector<Tok> toks;
        size_t longest = 0;
        for (size_t idx : group) {
            longest = std::max(longest, entries[idx].state.size());
        }
        for (size_t p = 0; p < longest; p++) {
            for (size_t s = 0; s < group.size(); s++) {
                const auto & state = entries[group[s]].state;
                if (p < state.size()) {
                    toks.push_back({ state[p], (llama_pos) p, (llama_seq_id) s, false, group[s], -1 });
                }
            }
        }
        if (!runner.run(toks, results)) {
            mark_group_failed(group, results, "decode failed (state phase)");
            continue;
        }

        // phase 2: for each question slot, copy state -> scratch seq and decode branches
        size_t max_branches = 0;
        for (size_t idx : group) {
            max_branches = std::max(max_branches, entries[idx].branches.size());
        }

        bool group_ok = true;
        for (size_t q = 0; q < max_branches && group_ok; q++) {
            for (size_t s = 0; s < group.size(); s++) {
                if (q >= entries[group[s]].branches.size()) {
                    continue;
                }
                const llama_seq_id dst = (llama_seq_id) (g + (int32_t) s);
                llama_memory_seq_rm(mem, dst, -1, -1);
                llama_memory_seq_cp(mem, (llama_seq_id) s, dst, -1, -1);
            }

            toks.clear();
            for (size_t s = 0; s < group.size(); s++) {
                const auto & entry = entries[group[s]];
                if (q >= entry.branches.size()) {
                    continue;
                }
                const auto & branch = entry.branches[q];
                std::vector<char> want(branch.ids.size(), 0);
                for (int32_t o : branch.outs) {
                    want[o] = 1;
                }
                const llama_seq_id seq = (llama_seq_id) (g + (int32_t) s);
                for (size_t p = 0; p < branch.ids.size(); p++) {
                    toks.push_back({ branch.ids[p], (llama_pos) (entry.state.size() + p), seq,
                            (bool) want[p], group[s], (int32_t) q });
                }
            }

            if (!toks.empty() && !runner.run(toks, results)) {
                group_ok = false;
            }
        }
        if (!group_ok) {
            mark_group_failed(group, results, "decode failed (branch phase)");
        }
    }

    return runner.n_tok_done;
}

// Control mode (--no-copy): every branch is decoded as state+branch from scratch on a
// single sequence, with no sequence-copy batching. Used to validate the batched scheme
// produces the same rows.
long run_no_copy(llama_context * ctx, const std::vector<EntryIn> & entries, std::vector<EntryOut> & results,
                  uint32_t n_ctx, uint32_t n_ubatch, int32_t n_embd) {
    Runner runner{ ctx, n_ubatch, n_embd };
    llama_memory_t mem = llama_get_memory(ctx);

    for (size_t i = 0; i < entries.size(); i++) {
        if (!results[i].error.empty()) {
            continue; // already excluded (e.g. an out-of-vocab token id)
        }
        const auto & entry = entries[i];
        for (size_t bi = 0; bi < entry.branches.size(); bi++) {
            const auto & branch = entry.branches[bi];
            const uint64_t need = entry.state.size() + branch.ids.size();
            if (need > n_ctx) {
                results[i].error = "entry does not fit in context (state + branch = " +
                        std::to_string(need) + " > --ctx " + std::to_string(n_ctx) + ")";
                results[i].rows.assign(entry.branches.size(), {});
                break;
            }

            llama_memory_clear(mem, true);

            std::vector<Tok> toks;
            toks.reserve(entry.state.size() + branch.ids.size());
            for (size_t p = 0; p < entry.state.size(); p++) {
                toks.push_back({ entry.state[p], (llama_pos) p, 0, false, i, (int32_t) bi });
            }
            std::vector<char> want(branch.ids.size(), 0);
            for (int32_t o : branch.outs) {
                want[o] = 1;
            }
            for (size_t p = 0; p < branch.ids.size(); p++) {
                toks.push_back({ branch.ids[p], (llama_pos) (entry.state.size() + p), 0,
                        (bool) want[p], i, (int32_t) bi });
            }

            if (!runner.run(toks, results)) {
                if (results[i].error.empty()) {
                    results[i].error = "decode failed";
                }
            }
        }
    }

    return runner.n_tok_done;
}

} // namespace

int main(int argc, char ** argv) {
    auto maybe_args = parse_args(argc, argv);
    if (!maybe_args.has_value()) {
        return 2; // unreachable: parse_args exits directly on error
    }
    Args args = *maybe_args;

    // 1) validate every input line before touching the model.
    std::vector<EntryIn> entries = read_and_validate_input(std::cin);

    std::vector<EntryOut> results(entries.size());
    for (size_t i = 0; i < entries.size(); i++) {
        results[i].id = entries[i].id;
        results[i].rows.resize(entries[i].branches.size());
    }

    // 2) load the model.
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mp);
    if (model == nullptr) {
        fatal_runtime("failed to load model: " + args.model);
    }

    // Reject entries that reference a token id outside the model's vocab now that we
    // know n_vocab, before any grouping/decoding happens -- one bad id fails only its
    // own entry, not the group it would otherwise have shared a decode call with.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    mark_out_of_vocab_entries(entries, results, llama_vocab_n_tokens(vocab));

    llama_adapter_lora * lora = nullptr;
    if (!args.lora.empty()) {
        lora = llama_adapter_lora_init(model, args.lora.c_str());
        if (lora == nullptr) {
            fatal_runtime("failed to load LoRA adapter: " + args.lora);
        }
    }

    // 3) pick the largest G that fits in VRAM with the required margin.
    SizedContext sized = make_sized_context(model, args);
    if (sized.ctx == nullptr) {
        fatal_runtime("context creation failed for every G in [1, " + std::to_string(args.max_g) + "]");
    }
    llama_context * ctx = sized.ctx;
    const int32_t g = sized.g;
    fprintf(stderr, "system1-rows: using G=%d\n", g);

    if (lora != nullptr) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &lora, 1, &scale) != 0) {
            fatal_runtime("failed to apply LoRA adapter");
        }
    }

    const int32_t n_embd = llama_model_n_embd(model);

    // 4) run.
    auto t0 = std::chrono::steady_clock::now();
    long n_tok_done = 0;
    if (args.no_copy) {
        n_tok_done = run_no_copy(ctx, entries, results, args.n_ctx, args.n_ubatch, n_embd);
    } else {
        n_tok_done = run_batched(ctx, entries, results, g, args.n_ctx, args.n_ubatch, n_embd);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // 5) finalize per-entry status: an entry with no error set and every branch's row
    // count matching its outs count is ok; a row-count mismatch is a decode bug, not a
    // silent success.
    long total_rows = 0;
    long n_entries_ok = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        EntryOut & r = results[i];
        if (!r.error.empty()) {
            r.ok = false;
            continue;
        }
        bool mismatch = false;
        for (size_t bi = 0; bi < entries[i].branches.size(); bi++) {
            if (r.rows[bi].size() != entries[i].branches[bi].outs.size() * (size_t) n_embd) {
                mismatch = true;
                break;
            }
        }
        if (mismatch) {
            r.ok = false;
            r.error = "row count mismatch";
            continue;
        }
        r.ok = true;
        n_entries_ok++;
        for (const auto & row : r.rows) {
            total_rows += (long) (row.size() / (size_t) n_embd);
        }
    }
    // 6) write the binary rows file: successful entries, in input order. fwrite/fclose
    // are checked -- a short write or a failed close (e.g. disk full) must not leave a
    // truncated file while stdout still claims ok:true for rows that never made it to
    // disk, since the downstream consumer slices the file by those counts.
    bool write_ok = true;
    size_t confirmed_upto = 0; // prefix [0, confirmed_upto) is durably on disk
    {
        FILE * out = fopen(args.out.c_str(), "wb");
        if (out == nullptr) {
            fatal_runtime("failed to open --out for writing: " + args.out);
        }
        for (size_t i = 0; i < results.size(); i++) {
            if (!results[i].ok) {
                confirmed_upto = i + 1;
                continue;
            }
            bool entry_ok = true;
            for (const auto & row : results[i].rows) {
                if (row.empty()) {
                    continue;
                }
                const size_t wrote = fwrite(row.data(), sizeof(float), row.size(), out);
                if (wrote != row.size()) {
                    entry_ok = false;
                    break;
                }
            }
            if (!entry_ok) {
                write_ok = false;
                break;
            }
            confirmed_upto = i + 1;
        }
        // fclose flushes buffered writes; if it fails we cannot trust that anything
        // buffered since the last successful flush actually reached disk, so nothing
        // is "confirmed" beyond this point either.
        if (fclose(out) != 0) {
            write_ok = false;
        }
    }
    if (!write_ok) {
        for (size_t i = confirmed_upto; i < results.size(); i++) {
            if (results[i].ok) {
                results[i].ok    = false;
                results[i].error = "output write failed";
            }
        }
        fprintf(stderr, "system1-rows: failed writing --out (%s); rows past entry %zu are not on disk\n",
                args.out.c_str(), confirmed_upto);
    }

    // 7) stdout: one JSON line per input entry, input order.
    for (const auto & r : results) {
        long n_rows = 0;
        if (r.ok) {
            for (const auto & row : r.rows) {
                n_rows += (long) (row.size() / (size_t) n_embd);
            }
        }
        json j;
        j["id"]    = r.id;
        j["ok"]    = r.ok;
        j["rows"]  = n_rows;
        j["error"] = r.ok ? json(nullptr) : json(r.error);
        std::cout << j.dump() << "\n";
    }
    std::cout.flush();

    // 8) stderr summary.
    const double tok_per_s = elapsed > 0.0 ? (double) n_tok_done / elapsed : 0.0;
    fprintf(stderr, "system1-rows: %zu entries (%ld ok), %ld tok, %ld rows, %.1f s, %.0f tok/s, G=%d\n",
            entries.size(), n_entries_ok, n_tok_done, total_rows, elapsed, tok_per_s, g);

    if (lora != nullptr) {
        llama_adapter_lora_free(lora);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return write_ok ? 0 : 1;
}
