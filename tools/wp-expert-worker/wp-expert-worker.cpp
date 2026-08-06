#include "wp-expert-worker.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "pipe-protocol.h"
#include "pipe-transport.h"
#include "weight-pager/wp-host-tier.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <utility>
#include <vector>

// poll() for the keepalive pump. POSIX only; the worker is already POSIX-only
// (O_DIRECT, posix_memalign) so this adds no new portability constraint.
#include <poll.h>            // ppoll needs _GNU_SOURCE, which glibc sets via -std=gnu++

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace wp_expert_worker {

static constexpr uint64_t DEFAULT_STAGING_BUFFERS = 16;

struct RequestStats {
    uint64_t ns_lookup  = 0;
    uint64_t ns_read    = 0;
    uint64_t ns_compute = 0;
    uint64_t ns_send    = 0;
    uint64_t n_resident      = 0;
    uint64_t n_pagein     = 0;
    uint64_t n_host_hit = 0;
    uint64_t n_host_demote = 0;
    uint64_t bytes_read = 0;
    uint64_t n_pagein_reserved = 0;
    uint64_t n_pagein_general = 0;
    uint64_t ns_host_get = 0;
    uint64_t host_bytes = 0;
    uint64_t n_graph_submits = 0;
    uint64_t n_device_allocs = 0;
    uint64_t ns_graph_build = 0;
    uint64_t ns_submit = 0;
    uint64_t ns_readback = 0;
    uint64_t ns_prep = 0;
    uint64_t ns_prep_setup = 0;   // ggml_init + new_tensor + buft queries
    uint64_t ns_prep_grow = 0;    // grow_io_buffer (device alloc when it grows)
    uint64_t ns_prep_attach = 0;  // attach_weight -> buffer_init_tensor
    uint64_t ns_prep_set = 0;     // the activation upload itself          // prepare_io: activation upload + io buffer growth
    uint64_t ns_hits = 0;          // compute_batch(hits) end to end
    uint64_t ns_wait = 0;          // batch.complete(): reader-thread join + I/O + H2D
    uint64_t ns_pagein_compute = 0;  // compute_batch(pageins) end to end
    uint64_t ns_result = 0;        // read_result
    uint64_t ns_encode = 0;        // fp32 -> fp16 of the reply
    uint64_t ns_h2d = 0;
    uint64_t bytes_h2d = 0;
    // ROUTING DENSITY (2026-08-04). compute_batch runs the FULL FFN for every
    // assigned expert over ALL request.n_tokens and then multiplies by a
    // per-token router weight that is ZERO for tokens not routed to that expert
    // (pipe-protocol.h: "one final router weight per token"). So the useful
    // fraction of the expert FLOPs is exactly the nonzero fraction of those
    // weights. Counting it directly rather than inferring it from
    // n_expert_used/n_expert, because the assignment list only contains experts
    // that got at least one token, which biases the naive estimate.
    uint64_t n_weight_nonzero = 0;   // token-expert pairs actually routed
    uint64_t n_weight_total   = 0;   // token-expert pairs actually COMPUTED
};

// Forward declarations: the probe itself is defined further down, next to
// run_self_bench, but WorkerStats::report() needs to read it.
void self_bench_tick(ggml_backend_t backend);
bool self_bench_stats(uint64_t & n, uint64_t & min_us, uint64_t & mean_us);

class WorkerStats {
public:
    WorkerStats() :
        enabled_(std::getenv("WP_WORKER_STATS") != nullptr &&
                 std::strcmp(std::getenv("WP_WORKER_STATS"), "1") == 0),
        next_report_(clock::now() + std::chrono::seconds(5)) {
    }

    bool enabled() const {
        return enabled_;
    }

    // Set once at startup from ExpertSlotPool::staging_kind() -- which
    // allocation path the staging pool actually used, not what was requested.
    void set_probe_backend(ggml_backend_t b) { probe_backend_ = b; }

    void set_staging_kind(const char * kind) {
        staging_kind_ = kind;
    }

    ~WorkerStats() {
        report();
    }

    void record(const RequestStats & request, size_t n_experts) {
        if (!enabled_) {
            return;
        }

        ns_lookup_ += request.ns_lookup;
        ns_read_ += request.ns_read;
        ns_compute_ += request.ns_compute;
        ns_send_ += request.ns_send;
        n_resident_ += request.n_resident;
        n_pagein_ += request.n_pagein;
        n_pagein_reserved_ += request.n_pagein_reserved;
        n_pagein_general_ += request.n_pagein_general;
        n_host_hit_ += request.n_host_hit;
        n_host_demote_ += request.n_host_demote;
        bytes_read_ += request.bytes_read;
        ns_host_get_ += request.ns_host_get;
        host_bytes_ = request.host_bytes;
        n_graph_submits_ += request.n_graph_submits;
        n_device_allocs_ += request.n_device_allocs;
        ns_graph_build_ += request.ns_graph_build;
        ns_submit_ += request.ns_submit;
        // PER-REQUEST DISTRIBUTION, not just the total. The cumulative ns_submit
        // cannot distinguish "every request costs 2.1 ms" from "most cost 0.2 ms
        // and a few cost 50 ms", and those have completely different fixes. An
        // isolated rebuild of this exact graph at this exact shape measures
        // ~190 us/expert on Vulkan0 and ~150 on CUDA, against ~1510 and ~254
        // in the worker -- so the shape of this histogram is the open question.
        if (request.n_graph_submits > 0) {
            const uint64_t per = request.ns_submit / request.n_graph_submits;
            if (per < submit_min_ns_) submit_min_ns_ = per;
            if (per > submit_max_ns_) submit_max_ns_ = per;
            // log2-ish buckets in us: <125, <250, <500, <1k, <2k, <4k, <8k, >=8k
            size_t b = 0;
            for (uint64_t edge = 125000; b < 7 && per >= edge; edge *= 2) ++b;
            ++submit_bucket_[b];
        }
        ns_readback_ += request.ns_readback;
        ns_prep_ += request.ns_prep;
        ns_prep_setup_ += request.ns_prep_setup;
        ns_prep_grow_ += request.ns_prep_grow;
        ns_prep_attach_ += request.ns_prep_attach;
        ns_prep_set_ += request.ns_prep_set;
        ns_hits_ += request.ns_hits;
        ns_wait_ += request.ns_wait;
        ns_pagein_compute_ += request.ns_pagein_compute;
        ns_result_ += request.ns_result;
        ns_encode_ += request.ns_encode;
        n_weight_nonzero_ += request.n_weight_nonzero;
        n_weight_total_   += request.n_weight_total;
        ns_h2d_ += request.ns_h2d;
        bytes_h2d_ += request.bytes_h2d;
        ++n_requests_;
        n_experts_ += n_experts;

        // WP_SELF_BENCH_EVERY=N: re-time the static graph every N requests while
        // serving. Costs one extra compute per N requests; default off.
        if (self_bench_every_ > 0 && probe_backend_ != nullptr &&
            n_requests_ % self_bench_every_ == 0) {
            self_bench_tick(probe_backend_);
        }

        const clock::time_point now = clock::now();
        if (now < next_report_) {
            return;
        }
        next_report_ = now + std::chrono::seconds(5);
        report();
    }

private:
    using clock = std::chrono::steady_clock;

    void report() const {
        if (!enabled_ || n_requests_ == 0) {
            return;
        }
        std::cout << "wp expert worker stats"
                  << " n_requests=" << n_requests_
                  << " n_experts=" << n_experts_
                  << " n_resident=" << n_resident_
                  << " n_pagein=" << n_pagein_
                  << " n_pagein_reserved=" << n_pagein_reserved_
                  << " n_pagein_general=" << n_pagein_general_
                  << " n_host_hit=" << n_host_hit_
                  << " n_host_demote=" << n_host_demote_
                  << " bytes_read=" << bytes_read_
                  << " ns_recv=unavailable"
                  << " ns_lookup=" << ns_lookup_
                  << " ns_read=" << ns_read_
                  << " ns_h2d=" << ns_h2d_
                  << " bytes_h2d=" << bytes_h2d_
                  << " gb_s_h2d=" << (ns_h2d_ == 0 ? 0.0 :
                        (double) bytes_h2d_ / (double) ns_h2d_)
                  << " staging_kind=" << staging_kind_
                  << " ns_host_get=" << ns_host_get_
                  << " ns_compute=" << ns_compute_
                  << " n_graph_submits=" << n_graph_submits_
                  << " n_device_allocs=" << n_device_allocs_
                  << " ns_graph_build=" << ns_graph_build_
                  << " ns_submit=" << ns_submit_
                  << " ns_readback=" << ns_readback_
                  << " ns_send=" << ns_send_
                  << " host_bytes=" << host_bytes_
                  << " ns_prep=" << ns_prep_
                  << " ns_prep_setup=" << ns_prep_setup_
                  << " ns_prep_grow=" << ns_prep_grow_
                  << " ns_prep_attach=" << ns_prep_attach_
                  << " ns_prep_set=" << ns_prep_set_
                  << " ns_hits=" << ns_hits_
                  << " ns_wait=" << ns_wait_
                  << " ns_pagein_compute=" << ns_pagein_compute_
                  << " ns_result=" << ns_result_
                  << " ns_encode=" << ns_encode_
                  << " n_weight_nonzero=" << n_weight_nonzero_
                  << " n_weight_total=" << n_weight_total_
                  << " submit_us_min=" << (submit_min_ns_ == UINT64_MAX ? 0 : submit_min_ns_ / 1000)
                  << " submit_us_max=" << (submit_max_ns_ / 1000)
                  << " submit_hist_us[<125,<250,<500,<1k,<2k,<4k,<8k,>=8k]="
                  << submit_bucket_[0] << ',' << submit_bucket_[1] << ','
                  << submit_bucket_[2] << ',' << submit_bucket_[3] << ','
                  << submit_bucket_[4] << ',' << submit_bucket_[5] << ','
                  << submit_bucket_[6] << ',' << submit_bucket_[7];
        {
            uint64_t pn = 0, pmin = 0, pmean = 0;
            if (self_bench_stats(pn, pmin, pmean)) {
                std::cout << " probe_n=" << pn
                          << " probe_static_us_min=" << pmin
                          << " probe_static_us_mean=" << pmean;
            }
        }
        std::cout << std::endl;
    }

    bool              enabled_ = false;
    clock::time_point next_report_;
    uint64_t          self_bench_every_ =
        std::getenv("WP_SELF_BENCH_EVERY") ? (uint64_t) atoll(std::getenv("WP_SELF_BENCH_EVERY")) : 0;
    ggml_backend_t    probe_backend_ = nullptr;
    uint64_t          submit_min_ns_ = UINT64_MAX;
    uint64_t          submit_max_ns_ = 0;
    uint64_t          submit_bucket_[8] = {0,0,0,0,0,0,0,0};
    uint64_t          ns_lookup_  = 0;
    uint64_t          ns_read_    = 0;
    uint64_t          ns_compute_ = 0;
    uint64_t          ns_send_    = 0;
    uint64_t          n_resident_      = 0;
    uint64_t          n_pagein_     = 0;
    uint64_t          n_pagein_reserved_ = 0;
    uint64_t          n_pagein_general_ = 0;
    uint64_t          n_host_hit_ = 0;
    uint64_t          n_host_demote_ = 0;
    uint64_t          bytes_read_ = 0;
    uint64_t          ns_host_get_ = 0;
    uint64_t          host_bytes_ = 0;
    uint64_t          n_graph_submits_ = 0;
    uint64_t          n_device_allocs_ = 0;
    uint64_t          ns_graph_build_ = 0;
    uint64_t          ns_submit_ = 0;
    uint64_t          ns_readback_ = 0;
    uint64_t          ns_prep_ = 0;
    uint64_t          ns_prep_setup_ = 0;
    uint64_t          ns_prep_grow_ = 0;
    uint64_t          ns_prep_attach_ = 0;
    uint64_t          ns_prep_set_ = 0;
    uint64_t          ns_hits_ = 0;
    uint64_t          ns_wait_ = 0;
    uint64_t          ns_pagein_compute_ = 0;
    uint64_t          ns_result_ = 0;
    uint64_t          ns_encode_ = 0;
    uint64_t          n_weight_nonzero_ = 0;
    uint64_t          n_weight_total_ = 0;
    uint64_t          ns_h2d_     = 0;
    uint64_t          bytes_h2d_  = 0;
    std::string       staging_kind_ = "unknown";
    uint64_t          n_requests_ = 0;
    uint64_t          n_experts_  = 0;
};

ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes,
        uint64_t pinned_bytes,
        const std::vector<int> & reserve_blocks,
        uint64_t reserve_bytes) {
    if (requested_slots <= 0) {
        throw std::invalid_argument("invalid expert resource plan dimensions");
    }

    std::map<uint64_t, int> histogram;
    std::map<uint64_t, std::map<int, int>> layer_counts;
    uint64_t max_page_size = 0;
    for (const ResourcePage & page : pages) {
        if (page.layer < 0 || page.size == 0 ||
            (!page.pinned && (histogram[page.size] == std::numeric_limits<int>::max() ||
             layer_counts[page.size][page.layer] == std::numeric_limits<int>::max()))) {
            throw std::invalid_argument("invalid expert resource page");
        }
        if (!page.pinned) {
            ++histogram[page.size];
            ++layer_counts[page.size][page.layer];
        }
        max_page_size = std::max(max_page_size, page.size);
    }
    if (max_page_size >
        std::numeric_limits<uint64_t>::max() / (uint64_t) requested_slots) {
        throw std::overflow_error("expert device budget overflows");
    }

    ResourcePlan result;
    result.requested_slots      = requested_slots;
    result.device_budget_bytes =
        max_page_size * (uint64_t) requested_slots;
    result.pinned_bytes = pinned_bytes;
    if (pinned_bytes > result.device_budget_bytes) {
        throw std::invalid_argument("resident expert bytes exceed device budget");
    }
    result.slot_budget_bytes = result.device_budget_bytes - pinned_bytes;

    if (histogram.empty()) {
        result.host_budget_bytes = host_budget_bytes;
        result.staging_buffers = 0;
        return result;
    }

    uint64_t    total_pages   = 0;
    long double weighted_bytes = 0.0;
    uint64_t    floor_bytes   = 0;
    std::vector<SlotClass> classes;
    classes.reserve(histogram.size());
    for (const auto & item : histogram) {
        int floor = 0;
        for (const auto & layer : layer_counts.at(item.first)) {
            floor = std::max(floor, layer.second);
        }
        if ((uint64_t) floor >
            (std::numeric_limits<uint64_t>::max() - floor_bytes) / item.first) {
            throw std::overflow_error("expert pin floor overflows");
        }
        floor_bytes += item.first * (uint64_t) floor;
        total_pages += (uint64_t) item.second;
        weighted_bytes +=
            (long double) item.first * (long double) item.second;
        classes.push_back({ item.first, 0, floor, item.second });
    }

    bool use_size_classes = floor_bytes <= result.slot_budget_bytes;
    if (use_size_classes) {
        const long double average =
            weighted_bytes / (long double) total_pages;
        const long double remaining =
            (long double) (result.slot_budget_bytes - floor_bytes);
        const long double remaining_slots = remaining / average;
        for (SlotClass & slot_class : classes) {
            const long double fraction =
                (long double) slot_class.pages / (long double) total_pages;
            long long count =
                (long long) slot_class.pin_floor +
                std::llround(fraction * remaining_slots);
            count = std::max<long long>(1, count);
            count = std::max<long long>(slot_class.pin_floor, count);
            count = std::min<long long>(slot_class.pages, count);
            slot_class.slots = (int) count;
        }

        auto planned_bytes = [&]() {
            uint64_t bytes = 0;
            for (const SlotClass & slot_class : classes) {
                bytes += slot_class.size * (uint64_t) slot_class.slots;
            }
            return bytes;
        };
        while (planned_bytes() > result.slot_budget_bytes) {
            SlotClass * trim = nullptr;
            for (SlotClass & slot_class : classes) {
                const int keep = std::max(1, slot_class.pin_floor);
                if (slot_class.slots > keep &&
                    (trim == nullptr || slot_class.size > trim->size)) {
                    trim = &slot_class;
                }
            }
            if (trim == nullptr) {
                use_size_classes = false;
                break;
            }
            --trim->slots;
        }
    }

    if (!use_size_classes) {
        int max_layer_pages = 0;
        std::map<int, int> pages_by_layer;
        for (const ResourcePage & page : pages) {
            if (page.pinned) {
                continue;
            }
            max_layer_pages =
                std::max(max_layer_pages, ++pages_by_layer[page.layer]);
        }
        if (result.slot_budget_bytes / max_page_size < (uint64_t) max_layer_pages) {
            throw std::invalid_argument(
                "expert slot budget is smaller than the largest layer request");
        }
        classes.clear();
        classes.push_back({
            max_page_size, (int) (result.slot_budget_bytes / max_page_size),
            max_layer_pages, (int) total_pages
        });
    }

    result.size_classes = use_size_classes;
    result.slot_classes = std::move(classes);
    for (const SlotClass & slot_class : result.slot_classes) {
        if (slot_class.slots >
            std::numeric_limits<int>::max() - result.slot_count) {
            throw std::overflow_error("expert slot count overflows");
        }
        result.slot_count += slot_class.slots;
        result.device_bytes +=
            slot_class.size * (uint64_t) slot_class.slots;
    }
    if (reserve_bytes != 0 && !reserve_blocks.empty()) {
        result.requested_reserved_bytes = reserve_bytes;
        uint64_t named_bytes = 0;
        for (const ResourcePage & page : pages) {
            if (!page.pinned && std::binary_search(reserve_blocks.begin(), reserve_blocks.end(), page.layer)) {
                if (named_bytes > UINT64_MAX - page.size) named_bytes = UINT64_MAX;
                else named_bytes += page.size;
            }
        }
        const uint64_t target = std::min(reserve_bytes, named_bytes);
        result.named_reservable_bytes = named_bytes;
        uint64_t remaining = target;
        int slot_index = 0;
        for (const SlotClass & slot_class : result.slot_classes) {
            const int class_start = slot_index;
            for (int i = 0; i < slot_class.slots && remaining >= slot_class.size; ++i) {
                result.reserved_slot_indices.push_back(class_start + i);
                result.reserved_bytes += slot_class.size;
                remaining -= slot_class.size;
            }
            slot_index += slot_class.slots;
        }
        result.reserved_slot_count = (int) result.reserved_slot_indices.size();
        result.general_slot_count = result.slot_count - result.reserved_slot_count;
        if (target != 0 && result.reserved_bytes == 0) {
            result.reserved_slot_indices.clear();
            result.reserved_slot_count = 0;
            result.general_slot_count = result.slot_count;
        }
    }
    if (result.general_slot_count == 0 && result.reserved_slot_count == 0) {
        result.general_slot_count = result.slot_count;
    }

    if (max_page_size >
        std::numeric_limits<uint64_t>::max() / DEFAULT_STAGING_BUFFERS) {
        throw std::overflow_error("default expert host budget overflows");
    }
    result.host_budget_bytes = host_budget_bytes == 0
        ? max_page_size * std::min<uint64_t>(
              (uint64_t) result.slot_count, DEFAULT_STAGING_BUFFERS)
        : host_budget_bytes;
    if (result.host_budget_bytes < max_page_size) {
        throw std::invalid_argument(
            "expert host budget is smaller than the largest page");
    }
    const uint64_t staging_count =
        std::min<uint64_t>(
            (uint64_t) result.slot_count,
            result.host_budget_bytes / max_page_size);
    if (staging_count == 0 ||
        staging_count > (uint64_t) std::numeric_limits<int>::max()) {
        throw std::overflow_error("invalid expert staging buffer count");
    }
    result.staging_buffers      = (int) staging_count;
    result.staging_buffer_bytes = max_page_size;
    result.staging_bytes        = max_page_size * staging_count;
    return result;
}

namespace {

static constexpr const char * MANIFEST_FORMAT =
    "llama.cpp.weight-pager.expert-shard-manifest";
static constexpr const char * INDEX_FORMAT =
    "llama.cpp.weight-pager.expert-shard-index";
static constexpr const char * DESCRIPTOR_FORMAT =
    "llama.cpp.weight-pager.expert-descriptor";
static constexpr size_t DIRECT_ALIGNMENT = 4096;

struct RoleSpec {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t        ne0 = 0;
    int64_t        ne1 = 0;
    uint64_t       bytes = 0;
    std::string    source_tensor_name;
};

struct HParams {
    int n_layer       = 0;
    int n_embd        = 0;
    int n_ff_exp      = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
};

struct Descriptor {
    HParams                                      hparams;
    int                                          expert_first = -1;
    int                                          expert_last  = -1;
    std::string                                  input_model;
    std::string                                  identity_algorithm;
    std::string                                  identity_value;
    std::vector<std::string>                     model_files;
    std::map<int, std::map<std::string, RoleSpec>> layers;
};

struct MemberSpan {
    uint64_t offset = 0;
    uint64_t size   = 0;
};

struct ExpertPage {
    int                               cache_id = -1;
    int                               layer  = -1;
    int                               expert = -1;
    fs::path                          blob;
    uint64_t                          offset = 0;
    uint64_t                          size   = 0;
    std::map<std::string, MemberSpan> roles;
    bool                              is_resident = false;
    ggml_backend_buffer_t             resident_buffer = nullptr;
    void *                            resident_base = nullptr;
};

struct Catalog {
    Descriptor                                      descriptor;
    std::map<std::pair<int, int>, ExpertPage>       pages;
    std::vector<int>                                layers;
    uint64_t                                        max_page_size = 0;
};

struct backend_deleter {
    void operator()(ggml_backend * backend) const {
        ggml_backend_free(backend);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        ggml_backend_buffer_free(buffer);
    }
};

struct context_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

struct galloc_deleter {
    void operator()(ggml_gallocr * galloc) const {
        ggml_gallocr_free(galloc);
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using galloc_ptr  = std::unique_ptr<ggml_gallocr, galloc_deleter>;

json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    json value;
    try {
        input >> value;
    } catch (const json::exception & error) {
        throw std::runtime_error("failed to parse " + path.string() + ": " + error.what());
    }
    return value;
}

template <typename T>
T get_value(const json & value, const char * key, const fs::path & path) {
    try {
        return value.at(key).get<T>();
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

const json & get_array(const json & value, const char * key, const fs::path & path) {
    try {
        const json & result = value.at(key);
        if (!result.is_array()) {
            throw std::runtime_error(path.string() + ": " + key + " is not an array");
        }
        return result;
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

void check_format(const json & value, const char * expected, const fs::path & path) {
    if (get_value<std::string>(value, "format", path) != expected ||
        get_value<int>(value, "version", path) != 1) {
        throw std::runtime_error(path.string() + ": unsupported format or version");
    }
}

int checked_int(uint64_t value, const char * name) {
    if (value == 0 || value > INT32_MAX) {
        throw std::runtime_error(std::string(name) + " is out of range");
    }
    return (int) value;
}

void sha_update_u64(sha256_t & hash, uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = (unsigned char) ((value >> (i * 8)) & 0xffu);
    }
    sha256_update(&hash, bytes.data(), bytes.size());
}

void sha_update_string(sha256_t & hash, const std::string & value) {
    sha_update_u64(hash, value.size());
    sha256_update(
        &hash, reinterpret_cast<const unsigned char *>(value.data()), value.size());
}

std::string source_model_identity(
        const std::string & input_model,
        const std::vector<std::string> & model_files) {
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-expert.source-model.v1");
    sha_update_string(hash, input_model);
    sha_update_u64(hash, model_files.size());
    for (const std::string & model_file : model_files) {
        sha_update_string(hash, model_file);
    }

    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream result;
    result << "sha256:" << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        result << std::setw(2) << (unsigned int) byte;
    }
    return result.str();
}

RoleSpec parse_role(const json & value, const fs::path & path) {
    RoleSpec role;
    const int type = get_value<int>(value, "ggml_type", path);
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        throw std::runtime_error(path.string() + ": ggml_type is out of range");
    }
    role.type = (enum ggml_type) type;
    const std::string type_name = get_value<std::string>(value, "ggml_type_name", path);
    if (type_name != ggml_type_name(role.type)) {
        throw std::runtime_error(path.string() + ": ggml type name and enum disagree");
    }
    const json & shape = get_array(value, "shape", path);
    if (shape.size() != 2) {
        throw std::runtime_error(path.string() + ": expert role shape is not 2D");
    }
    role.ne0   = shape.at(0).get<int64_t>();
    role.ne1   = shape.at(1).get<int64_t>();
    role.bytes = get_value<uint64_t>(value, "bytes_per_expert", path);
    role.source_tensor_name =
        get_value<std::string>(value, "source_tensor_name", path);
    if (role.ne0 <= 0 || role.ne1 <= 0 ||
        role.source_tensor_name.empty() ||
        ggml_row_size(role.type, role.ne0) * (uint64_t) role.ne1 != role.bytes) {
        throw std::runtime_error(path.string() + ": expert role shape/type byte count is invalid");
    }
    return role;
}

Descriptor load_descriptor(const fs::path & path) {
    const json value = read_json(path);
    check_format(value, DESCRIPTOR_FORMAT, path);
    Descriptor result;

    const json & hparams = value.at("hparams");
    result.hparams.n_layer =
        checked_int(get_value<uint64_t>(hparams, "n_layer", path), "n_layer");
    result.hparams.n_embd =
        checked_int(get_value<uint64_t>(hparams, "n_embd", path), "n_embd");
    result.hparams.n_ff_exp =
        checked_int(get_value<uint64_t>(hparams, "n_ff_exp", path), "n_ff_exp");
    result.hparams.n_expert =
        checked_int(get_value<uint64_t>(hparams, "n_expert", path), "n_expert");
    result.hparams.n_expert_used =
        checked_int(get_value<uint64_t>(hparams, "n_expert_used", path), "n_expert_used");
    if (result.hparams.n_expert_used > result.hparams.n_expert ||
        get_value<std::string>(hparams, "activation", path) != "silu") {
        throw std::runtime_error(path.string() + ": unsupported expert hparams");
    }

    const json & range = value.at("retained_expert_range");
    result.expert_first = get_value<int>(range, "first", path);
    result.expert_last  = get_value<int>(range, "last", path);
    if (result.expert_first < 0 || result.expert_last < result.expert_first ||
        result.expert_last >= result.hparams.n_expert) {
        throw std::runtime_error(path.string() + ": invalid retained expert range");
    }

    const json & identity = value.at("shard_manifest_identity");
    result.identity_algorithm = get_value<std::string>(identity, "algorithm", path);
    result.identity_value     = get_value<std::string>(identity, "value", path);
    if (result.identity_algorithm.empty() || result.identity_value.empty()) {
        throw std::runtime_error(path.string() + ": empty shard manifest identity");
    }
    const json & source_model = value.at("source_model");
    result.input_model = get_value<std::string>(source_model, "input_model", path);
    for (const json & model_file : get_array(source_model, "model_files", path)) {
        result.model_files.push_back(model_file.get<std::string>());
    }
    if (result.input_model.empty() || result.model_files.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no source model");
    }

    const json & layers = get_array(value, "layers", path);
    for (const json & layer_value : layers) {
        const int layer = get_value<int>(layer_value, "layer", path);
        if (layer < 0 || layer >= result.hparams.n_layer || result.layers.count(layer) != 0) {
            throw std::runtime_error(path.string() + ": invalid or repeated layer descriptor");
        }
        const json & roles = layer_value.at("roles");
        std::map<std::string, RoleSpec> parsed;
        for (const std::string name : { "gate", "up", "down" }) {
            parsed.emplace(name, parse_role(roles.at(name), path));
        }
        if (parsed.at("gate").ne0 != result.hparams.n_embd ||
            parsed.at("gate").ne1 != result.hparams.n_ff_exp ||
            parsed.at("up").ne0 != result.hparams.n_embd ||
            parsed.at("up").ne1 != result.hparams.n_ff_exp ||
            parsed.at("down").ne0 != result.hparams.n_ff_exp ||
            parsed.at("down").ne1 != result.hparams.n_embd) {
            throw std::runtime_error(path.string() + ": descriptor role shapes disagree with hparams");
        }
        result.layers.emplace(layer, std::move(parsed));
    }
    if (result.layers.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no layers");
    }
    return result;
}

std::string role_from_mask(uint64_t mask) {
    switch (mask) {
        case 1: return "up";
        case 2: return "gate";
        case 4: return "down";
        default:
            throw std::runtime_error("sidecar has an unknown role mask " + std::to_string(mask));
    }
}

Catalog load_catalog(const fs::path & manifest_path, const fs::path & descriptor_path) {
    Catalog result;
    result.descriptor = load_descriptor(descriptor_path);

    const json manifest = read_json(manifest_path);
    check_format(manifest, MANIFEST_FORMAT, manifest_path);
    if (get_value<std::string>(manifest, "sharding_mode", manifest_path) !=
        "expert-index-range") {
        throw std::runtime_error("worker requires an expert-index-range shard manifest");
    }
    const json & identity = manifest.at("content_hash");
    if (get_value<std::string>(identity, "algorithm", manifest_path) !=
            result.descriptor.identity_algorithm ||
        get_value<std::string>(identity, "value", manifest_path) !=
            result.descriptor.identity_value) {
        throw std::runtime_error("descriptor does not match shard manifest identity");
    }
    if (get_array(manifest, "model_files", manifest_path) !=
        json(result.descriptor.model_files)) {
        throw std::runtime_error("descriptor and shard manifest model files disagree");
    }
    if (get_value<std::string>(manifest, "input_model", manifest_path) !=
        result.descriptor.input_model) {
        throw std::runtime_error("descriptor and shard manifest input models disagree");
    }
    const json & range = manifest.at("retained_expert_range");
    if (get_value<int>(range, "first", manifest_path) != result.descriptor.expert_first ||
        get_value<int>(range, "last", manifest_path) != result.descriptor.expert_last) {
        throw std::runtime_error("descriptor and shard manifest expert ranges disagree");
    }

    const json & shards = get_array(manifest, "shards", manifest_path);
    if (shards.size() != get_value<uint64_t>(manifest, "shard_count", manifest_path)) {
        throw std::runtime_error("manifest shard count mismatch");
    }
    std::set<int> seen_layers;
    std::set<int> seen_shard_indices;
    uint64_t total_groups = 0;
    uint64_t total_bytes  = 0;
    for (const json & shard : shards) {
        const fs::path index_path = manifest_path.parent_path() /
            get_value<std::string>(shard, "index_file", manifest_path);
        const json index = read_json(index_path);
        check_format(index, INDEX_FORMAT, index_path);
        const int shard_index = get_value<int>(index, "shard_index", index_path);
        const int layer_first = get_value<int>(index, "layer_first", index_path);
        const int layer_last  = get_value<int>(index, "layer_last", index_path);
        if (shard_index < 0 || (uint64_t) shard_index >= shards.size() ||
            !seen_shard_indices.insert(shard_index).second ||
            get_value<uint64_t>(index, "shard_count", index_path) != shards.size() ||
            get_value<int>(shard, "shard_index", manifest_path) != shard_index ||
            get_value<int>(shard, "layer_first", manifest_path) != layer_first ||
            get_value<int>(shard, "layer_last", manifest_path) != layer_last ||
            get_value<uint64_t>(shard, "group_count", manifest_path) !=
                get_value<uint64_t>(index, "group_count", index_path) ||
            get_value<uint64_t>(shard, "blob_bytes", manifest_path) !=
                get_value<uint64_t>(index, "blob_bytes", index_path) ||
            get_value<std::string>(shard, "blob_file", manifest_path) !=
                get_value<std::string>(index, "blob_file", index_path) ||
            get_array(index, "model_files", index_path) !=
                get_array(manifest, "model_files", manifest_path) ||
            layer_first != layer_last || !seen_layers.insert(layer_first).second ||
            result.descriptor.layers.count(layer_first) == 0) {
            throw std::runtime_error(index_path.string() + ": invalid or undescribed layer");
        }

        const fs::path blob_path = manifest_path.parent_path() /
            get_value<std::string>(index, "blob_file", index_path);
        std::error_code size_error;
        const uint64_t actual_size = fs::file_size(blob_path, size_error);
        const uint64_t blob_bytes  = get_value<uint64_t>(index, "blob_bytes", index_path);
        if (size_error || actual_size != blob_bytes) {
            throw std::runtime_error(blob_path.string() + ": blob size does not match sidecar");
        }

        const json & groups = get_array(index, "groups", index_path);
        if (groups.size() != get_value<uint64_t>(index, "group_count", index_path)) {
            throw std::runtime_error(index_path.string() + ": group count mismatch");
        }
        uint64_t next_offset = 0;
        int expected_expert = result.descriptor.expert_first;
        for (const json & group : groups) {
            ExpertPage page;
            if (total_groups > (uint64_t) INT32_MAX) {
                throw std::runtime_error("expert page cache index is out of range");
            }
            page.cache_id = (int) total_groups;
            page.layer  = get_value<int>(group, "block_idx", index_path);
            page.expert = get_value<int>(group, "expert_idx", index_path);
            page.blob   = blob_path;
            page.offset = next_offset;
            if (page.layer != layer_first || page.expert != expected_expert++) {
                throw std::runtime_error(index_path.string() + ": expert groups are not dense and ordered");
            }
            const json & members = get_array(group, "members", index_path);
            if (members.size() != 3 ||
                get_value<uint64_t>(group, "member_count", index_path) != 3) {
                throw std::runtime_error(index_path.string() + ": expert group does not have three members");
            }
            const auto & role_specs = result.descriptor.layers.at(page.layer);
            for (const json & member : members) {
                const std::string role =
                    role_from_mask(get_value<uint64_t>(member, "role_mask", index_path));
                const uint64_t offset = get_value<uint64_t>(member, "offset", index_path);
                const uint64_t size   = get_value<uint64_t>(member, "size", index_path);
                const std::string source_tensor_name =
                    get_value<std::string>(member, "source_tensor_name", index_path);
                if (offset != next_offset || size != role_specs.at(role).bytes ||
                    source_tensor_name != role_specs.at(role).source_tensor_name ||
                    page.roles.count(role) != 0) {
                    throw std::runtime_error(
                        index_path.string() + ": member spans disagree with descriptor");
                }
                page.roles.emplace(role, MemberSpan{ offset - page.offset, size });
                next_offset += size;
                page.size += size;
            }
            if (page.offset % DIRECT_ALIGNMENT != 0 ||
                page.size % DIRECT_ALIGNMENT != 0) {
                throw std::runtime_error(
                    index_path.string() +
                    ": expert page is not aligned for one O_DIRECT read");
            }
            result.max_page_size = std::max(result.max_page_size, page.size);
            if (!result.pages.emplace(
                    std::make_pair(page.layer, page.expert), std::move(page)).second) {
                throw std::runtime_error(index_path.string() + ": repeated expert page");
            }
            ++total_groups;
        }
        if (next_offset != blob_bytes) {
            throw std::runtime_error(index_path.string() + ": groups do not cover blob");
        }
        total_bytes += blob_bytes;
    }
    if (seen_layers.size() != result.descriptor.layers.size()) {
        throw std::runtime_error("descriptor and shard manifest layer sets disagree");
    }
    if (total_groups != get_value<uint64_t>(manifest, "total_group_count", manifest_path) ||
        total_bytes != get_value<uint64_t>(manifest, "total_blob_bytes", manifest_path)) {
        throw std::runtime_error("shard catalog totals disagree with manifest");
    }
    result.layers.assign(seen_layers.begin(), seen_layers.end());
    return result;
}

void run_self_bench(ggml_backend_t backend, uint32_t n_embd, uint32_t n_ff);
void run_self_bench_early(ggml_backend_t backend) {
    // DS4-Flash expert shape, hardcoded: init_backend has no catalog yet and
    // this path is diagnostic-only (WP_SELF_BENCH=2).
    const char * saved = std::getenv("WP_SELF_BENCH");
    (void) saved;
    setenv("WP_SELF_BENCH", "1", 1);   // run_self_bench gates on '1'
    run_self_bench(backend, 4096, 2048);
    setenv("WP_SELF_BENCH", "2", 1);
}

backend_ptr init_backend(const std::string & device) {
    std::string lower = device;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    ggml_backend_t backend = nullptr;
    if (lower == "cpu") {
        backend = ggml_backend_cpu_init();
        if (backend != nullptr) {
            const unsigned int hw = std::thread::hardware_concurrency();
            ggml_backend_cpu_set_n_threads(backend, hw == 0 ? 1 : (int) hw);
        }
    } else {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name(device.c_str(), nullptr);
    }
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize device " + device);
    }
    // WP_SELF_BENCH=2 benchmarks HERE -- backend freshly created, no slot pool,
    // no staging pool, nothing else in the process. WP_SELF_BENCH=1 benchmarks
    // in the Worker ctor AFTER both pools exist. The pair bisects the ctor:
    // fast here + slow there => a pool is responsible; slow in both => the
    // backend/process is slow from birth. Standalone reference is ~190 us on
    // Vulkan0, and the post-pool measurement is 1327 us (2026-08-01).
    {
        const char * mode = std::getenv("WP_SELF_BENCH");
        if (mode != nullptr && mode[0] == '2') {
            run_self_bench_early(backend);
        }
    }
    return backend_ptr(backend);
}


// WP_SELF_BENCH=1: time a STATIC pre-built expert graph inside THIS process,
// on THIS backend instance, before serving any request.
//
// WHY THIS EXISTS. A standalone benchmark rebuilding the same graph at the same
// shape measures ~190 us/compute on the RX 480, but the worker's FASTEST of
// 1996 live submits is 618 us and its median is ~1800 us (2026-08-01). Twelve
// reconstructed differences were tested and none reproduced the gap: shader,
// per-graph overhead, buffer layout, 400 live buffers, graph rotation, gallocr
// realloc per request, readback, H2D (per-BYTE, ~36 us weighted), backend
// construction, within-process degradation, second-GPU and CPU contention.
// So the remaining difference is either the worker's PER-REQUEST GRAPH or its
// PROCESS/BACKEND STATE, and a standalone binary cannot tell those apart.
// This does: same process, same backend, same device, static graph.
//   ~190 us  -> the backend is fine; the per-request graph path is the cost.
//   ~618 us  -> the graph is fine; something about this process/backend is.
// Persistent handle for the periodic in-serving probe (WP_SELF_BENCH_EVERY=N).
// The static graph is built once and re-timed every N requests WHILE the worker
// serves real traffic. It separates two things the live ns_submit cannot:
//   static stays ~190 us while real requests cost ~647 us => the difference is
//     the worker's per-request GRAPH CONTENT (cpy into io buffer, gallocr-owned
//     routing weights, attach_weight into slot buffers).
//   static ALSO degrades to ~647 us => it is process/backend state under load
//     (in-flight transfers, reader threads, queue depth), not the graph.
struct SelfBenchProbe {
    ggml_context *        wctx  = nullptr;
    ggml_backend_buffer_t wbuf  = nullptr;
    ggml_context *        gctx  = nullptr;
    ggml_cgraph *         graph = nullptr;
    ggml_gallocr_t        ga    = nullptr;
    uint64_t              min_ns = UINT64_MAX;
    uint64_t              total_ns = 0;
    uint64_t              n = 0;
    bool                  ready = false;
};
static SelfBenchProbe g_probe;

void run_self_bench(ggml_backend_t backend, uint32_t n_embd, uint32_t n_ff) {
    const char * env = std::getenv("WP_SELF_BENCH");
    if (env == nullptr || env[0] != '1') {
        return;
    }
    const ggml_init_params wp_params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 16,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * wctx = ggml_init(wp_params);
    ggml_tensor * gate  = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff);
    ggml_tensor * up    = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff);
    ggml_tensor * down  = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_ff,  n_embd);
    ggml_tensor * input = ggml_new_tensor_2d(wctx, GGML_TYPE_F32,   n_embd, 1);
    ggml_tensor * rw    = ggml_new_tensor_2d(wctx, GGML_TYPE_F32,   1,      1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (wbuf == nullptr) {
        std::cout << "wp self-bench: alloc failed" << std::endl;
        ggml_free(wctx);
        return;
    }
    {
        std::vector<uint8_t> junk(ggml_nbytes(gate), 0x5a);
        ggml_backend_tensor_set(gate, junk.data(), 0, ggml_nbytes(gate));
        ggml_backend_tensor_set(up,   junk.data(), 0, ggml_nbytes(up));
        ggml_backend_tensor_set(down, junk.data(), 0, ggml_nbytes(down));
        std::vector<float> f(n_embd, 0.01f);
        ggml_backend_tensor_set(input, f.data(), 0, ggml_nbytes(input));
        const float one = 0.5f;
        ggml_backend_tensor_set(rw, &one, 0, sizeof(float));
    }
    const size_t nodes = 32;
    const ggml_init_params gp = {
        /* .mem_size   = */ ggml_tensor_overhead() * nodes
                            + ggml_graph_overhead_custom(nodes, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * g = ggml_mul_mat(gctx, gate, input);
    ggml_tensor * u = ggml_mul_mat(gctx, up,   input);
    ggml_tensor * h = ggml_swiglu_split(gctx, g, u);
    ggml_tensor * o = ggml_mul_mat(gctx, down, h);
    ggml_tensor * w = ggml_mul(gctx, o, rw);
    ggml_cgraph * graph = ggml_new_graph_custom(gctx, nodes, false);
    ggml_build_forward_expand(graph, w);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, graph)) {
        std::cout << "wp self-bench: graph alloc failed" << std::endl;
        ggml_gallocr_free(ga); ggml_free(gctx);
        ggml_backend_buffer_free(wbuf); ggml_free(wctx);
        return;
    }
    for (int i = 0; i < 5; ++i) {          // warm up: first Vulkan submit compiles pipelines
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    uint64_t best = UINT64_MAX, total = 0;
    const int iters = 300;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (ns < best) best = ns;
        total += ns;
    }
    std::cout << "wp self-bench: static 1-expert graph on this backend"
              << " iters=" << iters
              << " min_us=" << best / 1000
              << " mean_us=" << total / iters / 1000
              << std::endl;
    if (std::getenv("WP_SELF_BENCH_EVERY") != nullptr) {
        g_probe.wctx = wctx; g_probe.wbuf = wbuf;
        g_probe.gctx = gctx; g_probe.graph = graph; g_probe.ga = ga;
        g_probe.ready = true;
        return;   // deliberately leaked for the process lifetime; diagnostic only
    }
    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}

// One timed compute of the static graph, called from the serving path.

std::vector<ResourcePage> resource_pages(const Catalog & catalog) {
    std::vector<ResourcePage> result;
    result.reserve(catalog.pages.size());
    for (const auto & item : catalog.pages) {
        result.push_back({ item.second.layer, item.second.size,
                           item.second.is_resident });
    }
    return result;
}

struct free_deleter {
    void operator()(void * p) const {
        std::free(p);
    }
};

// WP_STAGING_PINNED=1 opts IN to page-locked staging via
// ggml_backend_dev_host_buffer_type. DEFAULT IS OFF (posix_memalign) because
// measurement on 2026-07-31 showed pinning is INERT AND UNSAFE:
//   - inert: 1070 gb_s_h2d 2.971 pinned vs 3.006 pageable; R9700 14.99 vs
//     16.10; throughput 0.865 vs 0.889 tok/s. All inside the +/-3% band. The
//     "1.29 GB/s H2D" that motivated this was DERIVED from a subtraction, never
//     measured; the first real ns_h2d says the 1070 was always at ~85% of its
//     gen3 x4 ceiling. There was no bounce-copy cost to recover.
//   - unsafe: Vulkan's host buffer type returns memory that is 4096-aligned but
//     NOT O_DIRECT-readable (host-visible device/BAR memory, not host RAM).
//     read() returns -1 and prefill dies at layer 3. The alignment and
//     buffer-type checks below both PASS on it, so the pool reports
//     staging_kind=pinned truthfully and then fails on first read.
// Read at startup only (not a struct Options field) so it cannot reintroduce
// the ABI mismatch that broke every worker on 2026-07-30 -- and so it stays
// settable per worker process, which is what made the A/B above possible.
bool staging_pinned_env_enabled() {
    const char * env = std::getenv("WP_STAGING_PINNED");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

class StagingPool {
public:
    StagingPool(const ResourcePlan & resources, ggml_backend_t backend) :
        buffer_bytes_(resources.staging_buffer_bytes),
        buffer_count_(resources.staging_buffers) {
        buffers_.reserve((size_t) resources.staging_buffers);
        host_buffers_.reserve((size_t) resources.staging_buffers);
        available_.reserve((size_t) resources.staging_buffers);

        ggml_backend_buffer_type_t host_buft = nullptr;
        bool try_pinned = staging_pinned_env_enabled();
        if (try_pinned && backend != nullptr) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev != nullptr) {
                // Portable entry point (ggml-backend.h:187). Never call a
                // backend-specific host-alloc symbol here: this worker binary
                // serves ROCm, CUDA and Vulkan, and #if GGML_USE_* branching
                // in this path is the exact bug class that has recurred six
                // times in this codebase.
                host_buft = ggml_backend_dev_host_buffer_type(dev);
            }
        }

        // Some backends provide no host buffer type; the allocation can also
        // fail or come back misaligned. The O_DIRECT reads need 4096-byte
        // alignment, so verify it now rather than discovering EINVAL at
        // read() time. Any of these cases falls back to posix_memalign for
        // the whole pool.
        //
        // Also verify the returned buffer's actual type matches host_buft.
        // ggml_backend_cuda_host_buffer_type_alloc_buffer silently falls back
        // to a plain CPU buffer (ggml_backend_cpu_buffer_from_ptr) when
        // cudaHostAlloc fails (e.g. GGML_CUDA_NO_PINNED, or the allocator is
        // just out of pinned memory) -- and only stamps buffer->buft with the
        // host type on the success path, so the fallback buffer keeps
        // ggml_backend_cpu_buffer_type(). That buffer is non-null, has a
        // valid base pointer, and is 4096-aligned, so the checks above all
        // pass on it even though the memory is pageable. Without this check
        // we would report staging_kind=pinned while actually running
        // pageable, and derive gb_s_h2d against a mechanism that never ran.
        bool pinned = try_pinned && host_buft != nullptr;
        if (pinned) {
            for (int i = 0; i < resources.staging_buffers; ++i) {
                buffer_ptr buf(ggml_backend_buft_alloc_buffer(
                    host_buft, (size_t) buffer_bytes_));
                void * raw = buf ? ggml_backend_buffer_get_base(buf.get()) : nullptr;
                if (!buf || raw == nullptr ||
                        (reinterpret_cast<uintptr_t>(raw) % DIRECT_ALIGNMENT) != 0 ||
                        ggml_backend_buffer_get_type(buf.get()) != host_buft) {
                    pinned = false;
                    break;
                }
                host_buffers_.push_back(std::move(buf));
            }
            if (!pinned) {
                host_buffers_.clear();
            }
        }

        if (pinned) {
            for (const buffer_ptr & buf : host_buffers_) {
                available_.push_back(ggml_backend_buffer_get_base(buf.get()));
            }
        } else {
            for (int i = 0; i < resources.staging_buffers; ++i) {
                void * raw = nullptr;
                if (posix_memalign(
                        &raw, DIRECT_ALIGNMENT, (size_t) buffer_bytes_) != 0) {
                    throw std::runtime_error(
                        "failed to allocate aligned O_DIRECT staging buffer");
                }
                buffers_.emplace_back(raw);
                available_.push_back(raw);
            }
        }

        pinned_ = pinned;
        // Logged once: one StagingPool is constructed per worker process.
        std::cerr << "wp expert worker: staging_kind="
                  << (pinned_ ? "pinned" : "pageable") << std::endl;
    }

    class Lease {
    public:
        Lease(Lease && other) noexcept :
            owner_(other.owner_), data_(other.data_) {
            other.owner_ = nullptr;
            other.data_  = nullptr;
        }

        ~Lease() {
            if (owner_ != nullptr) {
                owner_->release(data_);
            }
        }

        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        Lease & operator=(Lease &&) = delete;

        void * get() const {
            return data_;
        }

    private:
        friend class StagingPool;

        Lease(StagingPool * owner, void * data) :
            owner_(owner), data_(data) {
        }

        StagingPool * owner_ = nullptr;
        void *        data_  = nullptr;
    };

    Lease borrow() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_cv_.wait(lock, [&]() { return !available_.empty(); });
        void * result = available_.back();
        available_.pop_back();
        return Lease(this, result);
    }

    int buffer_count() const {
        return buffer_count_;
    }

    uint64_t buffer_bytes() const {
        return buffer_bytes_;
    }

    // Which allocation path actually ran, not what was requested.
    bool pinned() const {
        return pinned_;
    }

private:
    void release(void * data) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push_back(data);
        }
        available_cv_.notify_one();
    }

    uint64_t                                        buffer_bytes_ = 0;
    int                                              buffer_count_ = 0;
    bool                                             pinned_ = false;
    std::vector<std::unique_ptr<void, free_deleter>> buffers_;
    std::vector<buffer_ptr>                         host_buffers_;
    std::vector<void *>                             available_;
    std::mutex                                      mutex_;
    std::condition_variable                         available_cv_;
};

class ResidentExpertPool {
public:
    ResidentExpertPool(ggml_backend_t backend, Catalog & catalog,
                       const std::vector<int> & blocks) :
        backend_(backend) {
        for (auto & item : catalog.pages) {
            ExpertPage & page = item.second;
            if (!std::binary_search(blocks.begin(), blocks.end(), page.layer)) {
                continue;
            }
            Allocation allocation;
            allocation.buffer.reset(ggml_backend_alloc_buffer(
                backend_, (size_t) page.size));
            if (!allocation.buffer) {
                throw std::runtime_error("failed to allocate resident expert page");
            }
            ggml_backend_buffer_set_usage(
                allocation.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            allocation.ctx.reset(ggml_init({
                /* .mem_size = */ ggml_tensor_overhead() * 2,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc = */ true,
            }));
            if (!allocation.ctx) {
                throw std::runtime_error("failed to allocate resident expert metadata");
            }
            allocation.raw = ggml_new_tensor_1d(
                allocation.ctx.get(), GGML_TYPE_I8, (int64_t) page.size);
            allocation.raw->buffer = allocation.buffer.get();
            allocation.raw->data = ggml_backend_buffer_get_base(allocation.buffer.get());
            if (allocation.raw->data == nullptr ||
                ggml_backend_buffer_init_tensor(
                    allocation.buffer.get(), allocation.raw) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("failed to initialize resident expert page");
            }

            void * host = nullptr;
            if (posix_memalign(&host, DIRECT_ALIGNMENT, (size_t) page.size) != 0) {
                throw std::runtime_error("failed to allocate resident expert staging");
            }
            try {
                read_once(page, host);
                ggml_backend_tensor_set(
                    allocation.raw, host, 0, (size_t) page.size);
            } catch (...) {
                std::free(host);
                throw;
            }
            std::free(host);

            page.is_resident = true;
            page.resident_buffer = allocation.buffer.get();
            page.resident_base = allocation.raw->data;
            pinned_bytes_ += page.size;
            ++pinned_pages_;
            allocations_.push_back(std::move(allocation));
        }
        if (pinned_pages_ != 0) {
            ggml_backend_synchronize(backend_);
        }
    }

    uint64_t pinned_bytes() const { return pinned_bytes_; }
    int pinned_pages() const { return pinned_pages_; }

private:
    struct Allocation {
        buffer_ptr buffer;
        context_ptr ctx;
        ggml_tensor * raw = nullptr;
    };

    static void read_once(const ExpertPage & page, void * dst) {
#if defined(__linux__)
        const int fd = open(page.blob.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "failed to open resident expert shard " + page.blob.string() +
                ": " + std::strerror(errno));
        }
        ssize_t n = -1;
        do {
            n = pread(fd, dst, (size_t) page.size, (off_t) page.offset);
        } while (n < 0 && errno == EINTR);
        const int saved_errno = errno;
        close(fd);
        if (n < 0 || (uint64_t) n != page.size) {
            throw std::runtime_error(
                "short resident expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " +
                std::to_string(page.size) + " (" + std::strerror(saved_errno) + ")");
        }
#else
        (void) page;
        (void) dst;
        throw std::runtime_error("resident expert loading requires Linux");
#endif
    }

    ggml_backend_t backend_ = nullptr;
    std::vector<Allocation> allocations_;
    uint64_t pinned_bytes_ = 0;
    int pinned_pages_ = 0;
};

class ExpertSlotPool {
private:
    struct PageIn {
        size_t             entry_index = 0;
        size_t             slot_index  = 0;
        const ExpertPage * page        = nullptr;
        int                fd          = -1;
    };

    // One STRIPE of one page-in. A page is read in WP_EXPERT_READ_STRIPES
    // aligned pieces so the dispatch thread can upload stripe k while the
    // reader thread is still reading stripe k+1. With one stripe this is
    // exactly the old whole-page result.
    //
    // WHY THE LEASE IS SHARED: every stripe of a page reads into a different
    // offset of the SAME staging buffer, so the lease must outlive the last
    // stripe's upload, not the first.
    struct ReadResult {
        size_t                              pagein_indexb = 0;
        std::shared_ptr<StagingPool::Lease> staging;
        size_t                              offset = 0;   // byte offset within the page
        size_t                              len    = 0;   // bytes in this stripe
        bool                                last   = true; // last stripe of this page
        std::exception_ptr                  error;
        std::chrono::steady_clock::time_point read_started;
        std::chrono::steady_clock::time_point read_finished;
        bool                                read_timed = false;
    };

    struct BatchState {
        std::vector<PageIn>                       pageins;
        std::atomic<size_t>                     next{0};
        std::mutex                              mutex;
        std::condition_variable                 cv;
        std::deque<std::unique_ptr<ReadResult>> ready;
        bool                                    start  = false;
        bool                                    cancel = false;
        bool                                    measure = false;
    };

public:
    ExpertSlotPool(
            ggml_backend_t backend, ResourcePlan resources,
            uint64_t host_victim_bytes, TestHooks * test_hooks,
            const std::vector<int> & reserve_blocks) :
        backend_(backend),
        resources_(std::move(resources)),
        staging_(resources_, backend),
        test_hooks_(test_hooks) {
        reserve_blocks_ = reserve_blocks;
        std::sort(reserve_blocks_.begin(), reserve_blocks_.end());
        if (resources_.slot_count < 0 ||
            (resources_.slot_count == 0 && resources_.pinned_bytes == 0) ||
            (resources_.slot_count > 0 && resources_.staging_buffer_bytes == 0) ||
            resources_.staging_buffer_bytes >
                (uint64_t) std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("invalid expert slot pool dimensions");
        }
        // WP_SELF_BENCH=3 benchmarks HERE: staging_ is already built (member
        // init list) but NO slot buffers exist yet. With =2 (before everything)
        // measuring 183 us and =1 (after everything) measuring 1341 us, this
        // splits the constructor: fast here => the SLOT buffers do it; slow here
        // => the STAGING pool does it. Staging is a fixed 16 buffers regardless
        // of --slots, which matches the observed flatness across 100..400 slots.
        {
            const char * mode = std::getenv("WP_SELF_BENCH");
            if (mode != nullptr && mode[0] == '3') {
                setenv("WP_SELF_BENCH", "1", 1);
                run_self_bench(backend_, 4096, 2048);
                setenv("WP_SELF_BENCH", "3", 1);
            }
        }
        slots_.reserve((size_t) resources_.slot_count);
        resources_.device_bytes = 0;
        std::set<int> reserved_indices(resources_.reserved_slot_indices.begin(),
                                       resources_.reserved_slot_indices.end());
        for (const SlotClass & slot_class : resources_.slot_classes) {
            for (int i = 0; i < slot_class.slots; ++i) {
                slots_.push_back(make_slot(slot_class.size));
                slots_.back().reserved = reserved_indices.count((int) slots_.size() - 1) != 0;
                resources_.device_bytes +=
                    ggml_backend_buffer_get_size(slots_.back().buffer.get());
            }
        }
        resources_.staging_buffers      = staging_.buffer_count();
        resources_.staging_buffer_bytes = staging_.buffer_bytes();
        resources_.staging_bytes =
            resources_.staging_buffer_bytes *
            (uint64_t) resources_.staging_buffers;

        if (host_victim_bytes != 0) {
            if (host_victim_bytes >
                (uint64_t) std::numeric_limits<size_t>::max() ||
                !host_tier_.init((size_t) host_victim_bytes, 0)) {
                throw std::runtime_error("failed to initialize host victim tier");
            }
            host_tier_.set_device_reader(
                [this](void * dst_host, const void * src_device, size_t n) {
                    for (const Slot & slot : slots_) {
                        if (slot.raw != nullptr && slot.raw->data == src_device) {
                            ggml_backend_tensor_get(slot.raw, dst_host, 0, n);
                            return true;
                        }
                    }
                    return false;
                });
            host_victim_enabled_ = true;
            // Arm HostTier's Pass 0 so an unconfirmed prediction is drained
            // before anything VRAM actually touched. Without it spec_tier_ is
            // false and a guess competes with a known-good victim -- the tier's
            // own comment calls that "prefetch actively degrading the tier it is
            // meant to fill". This line silently failed to apply once already:
            // a str.replace with the wrong indentation is a no-op, not an error.
            host_tier_.set_speculative_tier(true);
        }
    }

    ~ExpertSlotPool() {
        // A landing thread outliving the pool would std::terminate on a joinable
        // thread, and it holds raw pointers into catalog pages and the staging
        // arena. Join before anything it touches goes away.
        if (host_thread_.joinable()) {
            host_thread_.join();
        }
#if defined(__linux__)
        for (const auto & item : fds_) {
            close(item.second);
        }
#endif
    }

    struct Loaded {
        ggml_backend_buffer_t buffer = nullptr;
        void *                base   = nullptr;
    };

    class Batch {
    public:
        Batch(Batch && other) noexcept;
        ~Batch();

        Batch(const Batch &) = delete;
        Batch & operator=(const Batch &) = delete;
        Batch & operator=(Batch &&) = delete;

        bool is_resident(size_t index) const {
            return entries_.at(index).hit;
        }

        // Slot this entry landed in, or SIZE_MAX for a pinned-resident page that
        // occupies no pool slot. Used by the speculative page-in path to re-stamp the LRU tick.
        size_t slot_index(size_t index) const {
            return entries_.at(index).slot_index;
        }

        const Loaded & loaded(size_t index) const {
            const Entry & entry = entries_.at(index);
            if (!entry.ready) {
                throw std::logic_error("expert batch entry is not ready");
            }
            return entry.loaded;
        }

        void complete();

        // Drain reads until every entry with index < entry_end that needs a
        // page-in has landed, leaving the rest in flight. Lets the caller
        // compute a leading chunk of experts while the tail is still being read
        // -- see WP_EXPERT_COMPUTE_CHUNKS at the dispatch site. complete() must
        // still be called afterwards to join the reader threads and finalise
        // the read timing, and remains safe to call after any number of these.
        void complete_upto(size_t entry_end);

        uint64_t lookup_ns() const {
            return ns_lookup_;
        }

        uint64_t read_ns() const {
            return ns_read_;
        }

        uint64_t ns_h2d() const {
            return ns_h2d_;
        }

        uint64_t bytes_h2d() const {
            return bytes_h2d_;
        }

        uint64_t n_resident() const {
            return n_resident_;
        }

        uint64_t n_pagein() const {
            return n_pagein_;
        }

        uint64_t n_pagein_reserved() const { return n_pagein_reserved_; }
        uint64_t n_pagein_general() const { return n_pagein_general_; }

        uint64_t bytes_read() const {
            return bytes_read_;
        }

        uint64_t n_host_hit() const {
            return n_host_hit_;
        }

        uint64_t n_host_demote() const {
            return n_host_demote_;
        }

        uint64_t ns_host_get() const {
            return ns_host_get_;
        }

        uint64_t host_bytes() const {
            return host_bytes_;
        }

    private:
        friend class ExpertSlotPool;

        struct Entry {
            Loaded loaded;
            size_t slot_index = std::numeric_limits<size_t>::max();
            bool   hit        = false;
            bool   ready      = false;
        };

        explicit Batch(ExpertSlotPool * owner, size_t count) :
            owner_(owner), entries_(count) {
        }

        ExpertSlotPool *           owner_ = nullptr;
        std::vector<Entry>         entries_;
        std::shared_ptr<BatchState> state_;
        std::vector<std::thread>   workers_;
        bool                       completed_ = false;
        uint64_t                   ns_lookup_ = 0;
        uint64_t                   ns_read_   = 0;
        uint64_t                   n_resident_     = 0;
        uint64_t                   n_pagein_    = 0;
        uint64_t                   n_pagein_reserved_ = 0;
        uint64_t                   n_pagein_general_ = 0;
        uint64_t                   bytes_read_ = 0;
        uint64_t                   n_host_hit_ = 0;
        uint64_t                   n_host_demote_ = 0;
        uint64_t                   ns_host_get_ = 0;
        uint64_t                   host_bytes_ = 0;
        uint64_t                   ns_h2d_    = 0;
        uint64_t                   bytes_h2d_ = 0;
        // Drain state. Lives on the Batch rather than in complete_batch's frame
        // so a drain can stop part-way (complete_upto) and be resumed. A read
        // that FAILED never sets entry.ready, so every drain loop is bounded by
        // received_ < pageins.size(), never by readiness alone.
        size_t                     received_  = 0;
        std::exception_ptr         first_error_;
        std::chrono::steady_clock::time_point first_read_;
        std::chrono::steady_clock::time_point last_read_;
        bool                       have_read_time_ = false;
    };

    Batch ensure_batch(
            const std::vector<const ExpertPage *> & pages,
            bool measure,
            std::chrono::steady_clock::time_point lookup_started) {
        // Take anything the reader threads already landed -- free residency for
        // this request, and it frees the pins. Non-blocking.
        if (spec_batch_ && !spec_recursion_) {
            spec_recursion_ = true;
            spec_pagein_poll(false);
            // An in-flight speculative slot is not yet valid, so find_slot below
            // cannot see it and this request would issue a SECOND read of the
            // same page. Wait for the bounded read already in flight instead.
            if (spec_batch_) {
                for (const ExpertPage * page : pages) {
                    if (page != nullptr && spec_in_flight_for(*page)) {
                        spec_pagein_poll(true);
                        break;
                    }
                }
            }
            spec_recursion_ = false;
        }
        Batch batch(this, pages.size());
        try {
            std::vector<size_t> pageins;
            pageins.reserve(pages.size());

            // Resolve and pin every hit before selecting a victim. A hit is
            // immediately usable while sibling pageins are read.
            for (size_t i = 0; i < pages.size(); ++i) {
                const ExpertPage & page = *pages[i];
                const size_t slot_index = find_slot(page);
                if (slot_index == slots_.size()) {
                    if (page.is_resident) {
                        batch.entries_[i] = {
                            { page.resident_buffer, page.resident_base },
                            std::numeric_limits<size_t>::max(),
                            true,
                            true,
                        };
                        ++batch.n_resident_;
                        continue;
                    }
                    pageins.push_back(i);
                    continue;
                }
                Slot & slot = slots_[slot_index];
                ++slot.pin_count;
                slot.tick = ++tick_;
                ++slot.uses;
                batch.entries_[i] = {
                    { slot.buffer.get(),
                      ggml_backend_buffer_get_base(slot.buffer.get()) },
                    slot_index,
                    true,
                    true,
                };
                ++batch.n_resident_;
            }

            // Reserve and pin all pagein slots before starting any read. Later
            // allocations in this request cannot select an earlier pagein.
            for (size_t entry_index : pageins) {
                const ExpertPage & page = *pages[entry_index];
                const size_t slot_index = select_victim(page);
                if (slot_index == slots_.size()) {
                    throw std::runtime_error(
                        "no expert slot can hold requested page");
                }
                // The floor rises with what we are actually throwing away, so a
                // page admitted now starts level with the pool instead of at the
                // bottom of it. This is the only thing keeping old counts from
                // locking newcomers out forever, and it needs no tuned constant.
                if (slots_[slot_index].valid) {
                    evict_age_ = slots_[slot_index].uses;
                    ++evictions_;
                }
                slots_[slot_index].lease_until = 0;
                ++slots_[slot_index].pin_count;
                batch.entries_[entry_index].slot_index = slot_index;
            }

            if (measure) {
                batch.ns_lookup_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - lookup_started).count();
            }

            struct HostHit {
                const void *                   src = nullptr;
                wp::HostTier::BorrowHandle borrow =
                    wp::HostTier::kInvalidBorrowHandle;
            };
            std::vector<HostHit> host_hits(pages.size());
            if (host_victim_enabled_) {
                for (size_t entry_index : pageins) {
                    const ExpertPage & page = *pages[entry_index];
                    if (host_tier_.borrow(
                            page.cache_id, &host_hits[entry_index].src,
                            (size_t) page.size, &host_hits[entry_index].borrow)) {
                        continue;
                    }
                }
            }

            auto release_host_hits = [&]() {
                for (size_t entry_index : pageins) {
                    HostHit & host_hit = host_hits[entry_index];
                    if (host_hit.borrow != wp::HostTier::kInvalidBorrowHandle) {
                        host_tier_.release(
                            pages[entry_index]->cache_id, host_hit.borrow);
                        host_hit.borrow = wp::HostTier::kInvalidBorrowHandle;
                    }
                }
            };

            batch.state_ = std::make_shared<BatchState>();
            batch.state_->pageins.reserve(pageins.size());
            try {
                for (size_t entry_index : pageins) {
                    const ExpertPage & page = *pages[entry_index];
                    const size_t slot_index =
                        batch.entries_[entry_index].slot_index;
                    Slot & slot = slots_[slot_index];
                    if (slot.valid && demote_slot(slot)) {
                        ++batch.n_host_demote_;
                    }
                    slot.valid = false;

                    HostHit & host_hit = host_hits[entry_index];
                    if (host_hit.borrow != wp::HostTier::kInvalidBorrowHandle) {
                        const std::chrono::steady_clock::time_point host_get_started =
                            measure ? std::chrono::steady_clock::now() :
                                      std::chrono::steady_clock::time_point();
                        ggml_backend_tensor_set(
                            slot.raw, host_hit.src, 0, (size_t) page.size);
                        host_tier_.release(page.cache_id, host_hit.borrow);
                        host_hit.borrow = wp::HostTier::kInvalidBorrowHandle;
                        host_tier_.erase(page.cache_id);
                        slot.valid    = true;
                        slot.key      = { page.layer, page.expert };
                        slot.cache_id = page.cache_id;
                        slot.size     = page.size;
                        slot.tick     = ++tick_;
                        // Admit at the age of the last page we evicted, not at
                        // 1. Otherwise a genuinely hot expert is thrown out
                        // before it can ever prove itself, and stale pages that
                        // were hot early squat forever. Those two are the only
                        // reasons plain use-counting loses to LRU.
                        slot.uses     = evict_age_ + 1;
                        batch.entries_[entry_index].loaded = {
                            slot.buffer.get(),
                            ggml_backend_buffer_get_base(slot.buffer.get())
                        };
                        batch.entries_[entry_index].hit   = true;
                        batch.entries_[entry_index].ready = true;
                        ++batch.n_host_hit_;
                        if (measure) {
                            batch.ns_host_get_ +=
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - host_get_started).count();
                        }
                    } else {
                        batch.state_->pageins.push_back({
                            entry_index, slot_index, &page, fd_for(page.blob)
                        });
                        ++batch.n_pagein_;
                        if (std::binary_search(reserve_blocks_.begin(), reserve_blocks_.end(), page.layer)) {
                            ++batch.n_pagein_reserved_;
                        } else {
                            ++batch.n_pagein_general_;
                        }
                        batch.bytes_read_ += page.size;
                        if (test_hooks_ != nullptr &&
                            test_hooks_->slot_reserved) {
                            test_hooks_->slot_reserved(
                                page.layer, page.expert, (int) slot_index);
                        }
                    }
                }
            } catch (...) {
                release_host_hits();
                throw;
            }
            batch.host_bytes_ = host_victim_enabled_ ? host_tier_.used_bytes() : 0;

            if (batch.state_->pageins.empty()) {
                batch.completed_ = true;
                return batch;
            }

            batch.state_->measure = measure;

            const size_t worker_count = std::min<size_t>(
                batch.state_->pageins.size(), (size_t) staging_.buffer_count());
            batch.workers_.reserve(worker_count);
            for (size_t i = 0; i < worker_count; ++i) {
                batch.workers_.emplace_back(
                    [this, state = batch.state_]() {
                        read_worker(state);
                    });
            }
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->start = true;
            }
            batch.state_->cv.notify_all();
            return batch;
        } catch (...) {
            cancel_workers(batch);
            release_pins(batch);
            throw;
        }
    }

    // Read `pages` into slots WITHOUT building a compute batch and WITHOUT
    // promoting anything in the LRU. This is the SPECULATIVE PAGE-IN path.
    //
    // IT REUSES ensure_batch + complete_batch DELIBERATELY. A second read path
    // would be a second place for O_DIRECT alignment, striping, staging leases
    // and slot bookkeeping to drift, and the whole point of a prefetch is that
    // the page it leaves behind is indistinguishable from a demand-paged one.
    //
    // *** THE PREFETCH LRU BAND IS THE READ-AMPLIFICATION GUARD. ***
    // select_victim evicts invalid slots first, then the lowest tick. A speculatively paged-in
    // slot must be valid or the next request re-reads it -- but if it also took
    // a FRESH tick it would outrank pages demand actually touched, and an unused
    // prefetch would evict a hot page. That is pool pollution, and it is exactly
    // what made the 2026-07 cross-layer attempt cost 2.7-3.1x the bytes at every
    // width, with a 0.973-precision predictor. So every page read here is
    // stamped from the prefetch band (see kDemandTickBase): strictly older than
    // anything demand has ever touched, hence the first victim if the guess was
    // wrong. A speculative page that is then actually USED gets ++tick_ from
    // ensure_batch's hit path and is promoted into the demand band like any
    // other. A wrong guess therefore costs one read and nothing else.
    //
    // *** SPECULATIVE READS ARE ASYNCHRONOUS. THIS IS THE WHOLE POINT. ***
    //
    // The first version of this called batch.complete() right here -- a blocking
    // join on the reader thread -- from the keepalive pump, which only runs when
    // NO request is pending. So a speculative read could never overlap with
    // compute, which is the only thing prefetching is for, and a request that
    // arrived mid-read waited for it. That is not a prefetch, it is a stall with
    // extra steps, and every speculation measurement taken against it is void.
    //
    // Now: submit and return. The reader threads carry the read while the
    // dispatch thread goes back to poll() and, when work arrives, computes. The
    // pages land underneath the request.
    //
    // SLOT SAFETY: ensure_batch reserves AND PINS every page-in slot before any
    // read is issued, and the pins are held for as long as we hold the Batch. An
    // in-flight speculative slot therefore cannot be recycled out from under its
    // own read -- which is exactly the corruption found on 2026-07-21, when the
    // speculative path allocated slots without pinning them.
    //
    // Returns the number of pages submitted (0 if they were all already present
    // or a batch is still in flight).
    size_t spec_pagein_submit(const std::vector<const ExpertPage *> & pages,
                              const std::vector<uint64_t> & leases) {
        if (pages.empty() || spec_batch_) {
            return 0;   // one speculative batch in flight at a time
        }
        std::vector<const ExpertPage *> cold;
        cold.reserve(pages.size());
        spec_leases_.clear();
        spec_leases_.reserve(pages.size());
        for (size_t i = 0; i < pages.size(); ++i) {
            const ExpertPage * page = pages[i];
            if (page == nullptr || page->is_resident || find_slot(*page) != slots_.size()) {
                continue;   // pinned resident, or already in a slot: nothing to do
            }
            cold.push_back(page);
            spec_leases_.push_back(i < leases.size() ? leases[i] : spec_lease_);
        }
        if (cold.empty()) {
            return 0;
        }
        try {
            // measure=false: a speculative read's cost belongs to the spec
            // counters, not to a request's phase timers. Mixing them would make
            // ns_read on the dispatch path stop meaning "time this request spent
            // reading".
            spec_batch_ = std::make_unique<Batch>(ensure_batch(cold, false, {}));
        } catch (const std::exception &) {
            // Advisory: a failed speculative read must not fail the worker. The
            // same error will surface on the demand path, where it belongs.
            ++spec_errors_;
            spec_batch_.reset();
            return 0;
        }
        spec_inflight_ = std::move(cold);
        // LOG AT SUBMIT, NOT AT HARVEST. The read is issued here, so this is when
        // the cost is paid and when the position in the stream is meaningful.
        // Logging at harvest inverts the order against R: an async batch can be
        // harvested INSIDE ensure_batch, i.e. after the dispatch's reference line
        // has already been written, and the classifier -- which matches S to the
        // next R -- then cannot credit the page to the request that used it. That
        // alone moved USED from 686 to 424 with no change in behaviour.
        if (spec_log_ != nullptr) {
            for (const ExpertPage * page : spec_inflight_) {
                fprintf(spec_log_, "S %d %d\n", page->layer, page->expert);
            }
            fflush(spec_log_);
        }
        return spec_inflight_.size();
    }

    bool spec_in_flight() const { return spec_batch_ != nullptr; }

    // Is there anywhere for a predicted page to land? Without the host tier the
    // answer is no and the caller must keep it on the VRAM path.
    bool host_landing_available() const { return host_victim_enabled_; }

    // *** LAND A GUESS IN HOST RAM, NOT IN VRAM. ***
    //
    // A predicted page in a VRAM slot is a slot a CERTAIN page cannot have, and
    // no lease fixes that -- a short lease only lets the guess give the slot up
    // AFTER it has taken it. Landing in the host arena costs no slot at all,
    // and if the guess turns out right the demand path promotes it over PCIe
    // (~1-2 ms) instead of re-reading NVMe (~5 ms) via the borrow path that
    // already exists in ensure_batch.
    //
    // Runs on a reader thread, not the dispatch thread: there is no GPU work
    // here, so the Vulkan command-pool affinity that constrains every other read
    // path does not apply. HostTier is mutex-guarded throughout.
    //
    // Returns pages queued. Requires the host tier -- without it there is
    // nowhere to land and the caller falls back to the VRAM path.
    size_t spec_host_submit(const std::vector<const ExpertPage *> & pages) {
        if (!host_victim_enabled_ || pages.empty() || host_thread_.joinable()) {
            return 0;
        }
        std::vector<const ExpertPage *> cold;
        cold.reserve(pages.size());
        for (const ExpertPage * page : pages) {
            // Counted, not lumped. host_landed=0 with host_errors=0 says the
            // filter ate everything and nothing about WHICH condition did it.
            if (page == nullptr || page->cache_id < 0) { ++host_skip_bad_;  continue; }
            if (page->is_resident)                     { ++host_skip_pin_;  continue; }
            if (find_slot(*page) != slots_.size())     { ++host_skip_vram_; continue; }
            if (host_tier_.contains(page->cache_id))   { ++host_skip_tier_; continue; }
            cold.push_back(page);
        }
        if (cold.empty()) {
            return 0;
        }
        host_pending_.store(cold.size(), std::memory_order_release);
        host_thread_ = std::thread([this, cold]() {
            for (const ExpertPage * page : cold) {
                try {
                    StagingPool::Lease lease = staging_.borrow();
                    // Fire the same hooks as every other read path. A host
                    // landing IS a read -- instrumentation that cannot see it
                    // would under-report exactly the bytes this feature spends.
                    if (test_hooks_ != nullptr && test_hooks_->read_started) {
                        test_hooks_->read_started(page->layer, page->expert);
                    }
                    read_page_range(*page, fd_for(page->blob), (char *) lease.get(),
                                    0, (size_t) page->size);
                    if (test_hooks_ != nullptr && test_hooks_->read_finished) {
                        test_hooks_->read_finished(page->layer, page->expert);
                    }
                    if (host_tier_.store(page->cache_id, lease.get(),
                                         (size_t) page->size, /*speculative=*/true)) {
                        host_landed_.fetch_add(1, std::memory_order_relaxed);
                        host_bytes_.fetch_add(page->size, std::memory_order_relaxed);
                    }
                } catch (const std::exception &) {
                    // Advisory, exactly like the VRAM speculative path: a failed
                    // guess must not fail the worker. The same error surfaces on
                    // the demand path, which is where it belongs.
                    host_errors_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            host_pending_.store(0, std::memory_order_release);
        });
        return cold.size();
    }

    bool spec_host_in_flight() const {
        return host_pending_.load(std::memory_order_acquire) != 0;
    }

    // Join a finished landing thread so the next chunk can start. Non-blocking:
    // only reaps a thread that has already drained its queue.
    void spec_host_reap() {
        if (host_thread_.joinable() && !spec_host_in_flight()) {
            host_thread_.join();
        }
    }

    uint64_t host_landed() const { return host_landed_.load(std::memory_order_relaxed); }
    uint64_t host_spec_bytes() const { return host_bytes_.load(std::memory_order_relaxed); }
    uint64_t host_spec_errors() const { return host_errors_.load(std::memory_order_relaxed); }
    uint64_t host_spec_promotions() const { return host_tier_.speculative_promotions(); }
    uint64_t host_spec_wasted() const { return host_tier_.speculative_evicted_unused(); }
    uint64_t host_skip_bad()   const { return host_skip_bad_; }
    uint64_t host_skip_pin()   const { return host_skip_pin_; }
    uint64_t host_skip_vram()  const { return host_skip_vram_; }
    uint64_t host_skip_tier()  const { return host_skip_tier_; }

    // Is `page` currently being read speculatively? The demand path has to ask,
    // because an in-flight slot is not yet valid, so find_slot cannot see it and
    // the request would issue a SECOND read of the same page.
    bool spec_in_flight_for(const ExpertPage & page) const {
        for (const ExpertPage * p : spec_inflight_) {
            if (p->layer == page.layer && p->expert == page.expert) {
                return true;
            }
        }
        return false;
    }

    // Harvest whatever has landed, WITHOUT BLOCKING. Returns true when the batch
    // finished and was retired. Safe to call from the idle pump and from the
    // dispatch path; both run on the dispatch thread, which is required because
    // drain_one_read performs the H2D upload and Vulkan command pools have
    // thread affinity.
    bool spec_pagein_poll(bool block) {
        if (!spec_batch_) {
            return false;
        }
        Batch & batch = *spec_batch_;
        while (batch.received_ < batch.state_->pageins.size()) {
            if (!block) {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                if (batch.state_->ready.empty()) {
                    return false;   // still in flight; come back later
                }
            }
            drain_one_read(batch);
        }
        retire_spec_batch();
        return true;
    }

    // Called when a demand request needs a page this batch is reading. Bounded:
    // at most WP_EXPERT_SPEC_CHUNK pages, and the read is already in flight.
    void spec_pagein_finish() { spec_pagein_poll(true); }

private:
    void retire_spec_batch() {
        Batch & batch = *spec_batch_;
        uint64_t n_read = 0;
        try {
            complete_batch(batch);
            n_read = batch.n_pagein();
            spec_bytes_ += batch.bytes_read();

            // Stamp by CHECKING THE SLOT, not by trusting an entry flag. `hit`
            // is only set for pages already present or from the host tier -- a
            // fresh disk page-in sets `ready`, never `hit` -- so keying off
            // is_resident() here would silently skip exactly the pages this path
            // exists to stamp, leaving every speculative page with a FRESH tick
            // and reintroducing the pollution the stale tick prevents.
            for (size_t i = 0; i < spec_inflight_.size(); ++i) {
                const size_t slot_index = batch.slot_index(i);
                if (slot_index >= slots_.size()) {
                    continue;
                }
                Slot & slot = slots_[slot_index];
                if (slot.valid &&
                    slot.key == std::pair<int, int>(spec_inflight_[i]->layer,
                                                    spec_inflight_[i]->expert)) {
                    slot.tick        = ++spec_tick_;
                    slot.uses        = 0;
                    slot.lease_until = evictions_ +
                        (i < spec_leases_.size() ? spec_leases_[i] : spec_lease_);
                }
            }
        } catch (const std::exception &) {
            ++spec_errors_;
        }
        spec_pageins_ += n_read;
        release_pins(batch);
        spec_batch_.reset();
        spec_inflight_.clear();
    }

public:
    uint64_t spec_pageins()  const { return spec_pageins_; }
    uint64_t spec_bytes()  const { return spec_bytes_; }
    uint64_t spec_errors() const { return spec_errors_; }

    // BORROWED, NOT OWNED -- the Worker opens WP_HINT_LOG and outlives the pool.
    //
    // The hint log is an ORDERED EVENT STREAM, and the order is the entire point.
    // "Speculatively read, then evicted, then demand-read again" and "speculated
    // on an expert that was never selected" are different failures with different
    // fixes, and the ONLY thing that separates them is whether the demand read
    // came after the speculative one. Two log files have no shared clock, so
    // pool-side page-ins and worker-side hints must go through the SAME handle.
    void set_spec_log(FILE * f) { spec_log_ = f; }

    // The lease a page gets by provenance. Read by the Worker, which owns the
    // hint queue and resolves provenance to a lease at enqueue time.
    uint64_t spec_lease()           const { return spec_lease_; }
    uint64_t spec_lease_predicted() const { return spec_lease_pred_; }

    const ResourcePlan & resources() const {
        return resources_;
    }

    // "pinned" or "pageable" -- whichever path the staging pool actually
    // used, not whichever was requested.
    const char * staging_kind() const {
        return staging_.pinned() ? "pinned" : "pageable";
    }

private:
    struct Slot {
        context_ptr         ctx;
        buffer_ptr          buffer;
        ggml_tensor *       raw       = nullptr;
        std::pair<int, int> key;
        int                 cache_id  = -1;
        uint64_t            capacity  = 0;
        uint64_t            size      = 0;
        uint64_t            tick      = 0;
        // How many times this page has been asked for. Ranking by USE COUNT is
        // the whole policy; tick is only the tie-break. Offline on the reference
        // stream this beats LRU by 3.2-4.0% of page-ins, against +0.2-1.7% for
        // ARC and negative for 2Q -- counting wins, and it is ten lines.
        uint64_t            uses      = 0;
        // Eviction-counter value at which this page stops being protected.
        // A SPECULATIVE PAGE IS USELESS IF IT DIES BEFORE ITS LAYER ARRIVES, and
        // measured 90% of them did. uses=0 made them the first victim by
        // construction -- which is what kept demand pages safe (layers 3+ were
        // identical to the digit across every arm) and also what guaranteed they
        // never survived long enough to pay. The lease is the bounded middle:
        // protected for a fixed number of evictions, then ordinary.
        uint64_t            lease_until = 0;
        int                 pin_count = 0;
        bool                reserved  = false;
        bool                valid     = false;
    };

    // Index of the slot already holding `page`, or slots_.size(). ONE definition:
    // the speculative path must agree with ensure_batch about what "already here" means,
    // or a prefetch re-reads a page that is sitting in a slot and the extra bytes
    // land in the speculative-read counter as if they were a real page-in.
    size_t find_slot(const ExpertPage & page) const {
        const std::pair<int, int> key(page.layer, page.expert);
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].valid && slots_[i].key == key) {
                return i;
            }
        }
        return slots_.size();
    }

    bool demote_slot(const Slot & slot) {
        return host_victim_enabled_ && slot.valid && slot.cache_id >= 0 && slot.size != 0 &&
            host_tier_.store_from_device(slot.cache_id, slot.raw->data, (size_t) slot.size);
    }

    size_t select_victim(const ExpertPage & page) const {
        const uint64_t page_size = page.size;
        const bool wants_reserved = std::binary_search(reserve_blocks_.begin(), reserve_blocks_.end(), page.layer);
        size_t victim = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || slot.valid ||
                (!wants_reserved && slot.reserved) ||
                (wants_reserved && !slot.reserved) ||
                page_size > slot.capacity) {
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity) {
                victim = i;
            }
        }
        if (victim != slots_.size()) {
            return victim;
        }

        if (wants_reserved) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (slot.pin_count != 0 || slot.valid || slot.reserved || page_size > slot.capacity) continue;
                if (victim == slots_.size() || slot.capacity < slots_[victim].capacity) victim = i;
            }
            if (victim != slots_.size()) return victim;
        }

        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || !slot.valid ||
                (!wants_reserved && slot.reserved) ||
                (wants_reserved && !slot.reserved) ||
                page_size > slot.capacity) {
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity ||
                (slot.capacity == slots_[victim].capacity &&
                 rank(slot) < rank(slots_[victim]))) {
                victim = i;
            }
        }
        if (wants_reserved && victim == slots_.size()) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (slot.pin_count != 0 || !slot.valid || page_size > slot.capacity) continue;
                if (victim == slots_.size() || slot.capacity < slots_[victim].capacity ||
                    (slot.capacity == slots_[victim].capacity &&
                     rank(slot) < rank(slots_[victim]))) victim = i;
            }
        }
        return victim;
    }

    void read_worker(const std::shared_ptr<BatchState> & state) {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&]() {
                return state->start || state->cancel;
            });
            if (state->cancel) {
                return;
            }
        }

        while (true) {
            const size_t pagein_indexb =
                state->next.fetch_add(1, std::memory_order_relaxed);
            if (pagein_indexb >= state->pageins.size()) {
                return;
            }
            const PageIn & pagein = state->pageins[pagein_indexb];
            bool read_started = false;
            std::shared_ptr<StagingPool::Lease> staging;
            std::exception_ptr fatal;
            try {
                staging = std::make_shared<StagingPool::Lease>(staging_.borrow());
                if (test_hooks_ != nullptr &&
                    test_hooks_->staging_borrowed) {
                    test_hooks_->staging_borrowed();
                }
                read_started = true;
                if (test_hooks_ != nullptr &&
                    test_hooks_->read_started) {
                    test_hooks_->read_started(
                        pagein.page->layer, pagein.page->expert);
                }
            } catch (...) {
                fatal = std::current_exception();
            }

            const auto plan = fatal ? std::vector<std::pair<size_t, size_t>>{{0, 0}}
                                    : stripe_plan(pagein.page->size,
                                                  state->pageins.size());
            // Publish each stripe as soon as it lands so the dispatch thread can
            // upload it while the next one is still being read. The LAST stripe
            // is what flips entry.ready and advances received_, so a page is
            // never visible to compute half-uploaded.
            for (size_t s = 0; s < plan.size(); ++s) {
                auto result = std::make_unique<ReadResult>();
                result->pagein_indexb = pagein_indexb;
                result->staging       = staging;
                result->offset        = plan[s].first;
                result->len           = plan[s].second;
                result->last          = (s + 1 == plan.size());
                result->error         = fatal;
                if (!fatal) {
                    try {
                        if (state->measure) {
                            result->read_started = std::chrono::steady_clock::now();
                            result->read_timed = true;
                        }
                        read_page_range(
                            *pagein.page, pagein.fd,
                            (char *) staging->get() + result->offset,
                            result->offset, result->len);
                    } catch (...) {
                        result->error = std::current_exception();
                    }
                    if (state->measure && result->read_timed) {
                        result->read_finished = std::chrono::steady_clock::now();
                    }
                }
                // A failed stripe ends the page: mark it last so the drain still
                // accounts for it, and stop reading the rest.
                const bool failed = result->error != nullptr;
                if (failed) {
                    result->last = true;
                }
                if (result->last && read_started && test_hooks_ != nullptr &&
                    test_hooks_->read_finished) {
                    try {
                        test_hooks_->read_finished(
                            pagein.page->layer, pagein.page->expert);
                    } catch (...) {
                        if (result->error == nullptr) {
                            result->error = std::current_exception();
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->ready.push_back(std::move(result));
                }
                state->cv.notify_one();
                if (failed) {
                    break;
                }
            }
        }
    }

    // Process exactly ONE completed read: H2D it into its slot and mark the
    // entry ready. Split out of complete_batch so a drain can stop part-way.
    void drain_one_read(Batch & batch) {
        {
            std::unique_ptr<ReadResult> result;
            {
                std::unique_lock<std::mutex> lock(batch.state_->mutex);
                batch.state_->cv.wait(lock, [&]() {
                    return !batch.state_->ready.empty();
                });
                result = std::move(batch.state_->ready.front());
                batch.state_->ready.pop_front();
            }

            const PageIn & pagein =
                batch.state_->pageins[result->pagein_indexb];
            if (result->read_timed) {
                if (!batch.have_read_time_ || result->read_started < batch.first_read_) {
                    batch.first_read_ = result->read_started;
                }
                if (!batch.have_read_time_ || result->read_finished > batch.last_read_) {
                    batch.last_read_ = result->read_finished;
                }
                batch.have_read_time_ = true;
            }
            if (result->error != nullptr) {
                if (batch.first_error_ == nullptr) {
                    batch.first_error_ = result->error;
                }
            } else {
                Slot & slot = slots_[pagein.slot_index];
                const bool measure_h2d = batch.state_->measure;
                const std::chrono::steady_clock::time_point h2d_started =
                    measure_h2d ? std::chrono::steady_clock::now() :
                                  std::chrono::steady_clock::time_point();
                // Upload ONLY this stripe, at its own offset. With one stripe
                // this is the original whole-page (0, page.size) call.
                ggml_backend_tensor_set(
                    slot.raw, (const char *) result->staging->get() + result->offset,
                    result->offset, result->len);
                if (measure_h2d) {
                    batch.ns_h2d_ +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - h2d_started).count();
                    batch.bytes_h2d_ += result->len;
                }
                // Everything below publishes the page to the compute path and so
                // must happen EXACTLY ONCE, on the final stripe -- otherwise a
                // half-uploaded slot becomes visible, and the page-in log and LRU
                // tick would fire once per stripe.
                if (result->last) {
                    // WP_PAGEIN_LOG=path: append "<layer> <expert>" for every page
                    // actually READ from disk. Intersecting the two 2026 workers'
                    // logs measures whether residency-affinity routing keeps their
                    // caches disjoint, or whether they fetch the same pages twice.
                    // Default off; one fprintf per pagein, negligible against a
                    // 13.37 MB O_DIRECT read.
                    if (pagein_log_ != nullptr) {
                        fprintf(pagein_log_, "%d %d\n", pagein.page->layer, pagein.page->expert);
                        // The harness SIGKILLs workers at teardown, so a buffered
                        // stream is lost entirely -- the first run produced two
                        // 0-byte logs. One fflush per 13.37 MB O_DIRECT read is free.
                        fflush(pagein_log_);
                    }
                    // The same event into the ordered stream, as a DEMAND read.
                    // Deliberately duplicated rather than joined against
                    // WP_PAGEIN_LOG after the fact: what makes this line useful is
                    // its POSITION relative to the S line for the same page, and a
                    // separate file cannot express that.
                    if (spec_log_ != nullptr) {
                        fprintf(spec_log_, "D %d %d\n", pagein.page->layer, pagein.page->expert);
                        fflush(spec_log_);
                    }
                    slot.valid = true;
                    slot.key   = {
                        pagein.page->layer, pagein.page->expert
                    };
                    slot.cache_id = pagein.page->cache_id;
                    slot.size     = pagein.page->size;
                    slot.tick  = ++tick_;
                    slot.uses  = evict_age_ + 1;
                    Batch::Entry & entry =
                        batch.entries_[pagein.entry_index];
                    entry.loaded = {
                        slot.buffer.get(),
                        ggml_backend_buffer_get_base(slot.buffer.get())
                    };
                    entry.ready = true;
                }
            }
            // received_ counts PAGES, not stripes -- complete_batch's loop is
            // bounded by pageins.size().
            const bool page_done = result->last;
            result.reset();
            if (page_done) {
                ++batch.received_;
            }
        }
    }

    // Drain until every entry below entry_end that needs a page-in is ready.
    // Entries at or above entry_end whose reads happen to land first are still
    // processed -- their H2D is done here, so nothing is dropped or re-done.
    // Does NOT join the reader threads or finalise timings; complete_batch does.
    void complete_batch_upto(Batch & batch, size_t entry_end) {
        if (batch.completed_) {
            return;
        }
        const auto range_pending = [&]() {
            for (const PageIn & pagein : batch.state_->pageins) {
                if (pagein.entry_index < entry_end &&
                    !batch.entries_[pagein.entry_index].ready) {
                    return true;
                }
            }
            return false;
        };
        // received_ bounds the loop: a FAILED read never sets ready, so waiting
        // on readiness alone would hang on exactly the error path.
        while (batch.received_ < batch.state_->pageins.size() && range_pending()) {
            drain_one_read(batch);
        }
    }

    void complete_batch(Batch & batch) {
        if (batch.completed_) {
            return;
        }

        while (batch.received_ < batch.state_->pageins.size()) {
            drain_one_read(batch);
        }

        for (std::thread & worker : batch.workers_) {
            worker.join();
        }
        batch.workers_.clear();
        batch.completed_ = true;
        if (batch.have_read_time_) {
            batch.ns_read_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                batch.last_read_ - batch.first_read_).count();
        }
        if (batch.first_error_ != nullptr) {
            std::rethrow_exception(batch.first_error_);
        }
    }

    void cancel_workers(Batch & batch) {
        if (batch.state_ != nullptr && !batch.workers_.empty()) {
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->cancel = true;
            }
            batch.state_->cv.notify_all();
            for (std::thread & worker : batch.workers_) {
                worker.join();
            }
            batch.workers_.clear();
        }
    }

    void release_pins(Batch & batch) noexcept {
        for (const Batch::Entry & entry : batch.entries_) {
            if (entry.slot_index == std::numeric_limits<size_t>::max()) {
                continue;
            }
            Slot & slot = slots_[entry.slot_index];
            if (slot.pin_count > 0) {
                --slot.pin_count;
            }
        }
        batch.entries_.clear();
    }

    void abandon_batch(Batch & batch) noexcept {
        if (!batch.completed_) {
            try {
                complete_batch(batch);
            } catch (...) {
            }
        }
        release_pins(batch);
    }

    FILE * pagein_log_ = [] {
        const char * p = std::getenv("WP_PAGEIN_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    FILE * spec_log_ = nullptr;

    Slot make_slot(uint64_t capacity) {
        if (capacity == 0 ||
            capacity > (uint64_t) std::numeric_limits<size_t>::max() ||
            capacity > (uint64_t) std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("invalid expert slot capacity");
        }

        Slot slot;
        slot.capacity = capacity;
        slot.buffer.reset(ggml_backend_alloc_buffer(backend_, (size_t) capacity));
        if (!slot.buffer) {
            throw std::runtime_error("failed to allocate device expert slot");
        }
        ggml_backend_buffer_set_usage(slot.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 2,
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        slot.ctx.reset(ggml_init(params));
        if (!slot.ctx) {
            throw std::runtime_error("failed to allocate expert slot metadata");
        }
        slot.raw = ggml_new_tensor_1d(
            slot.ctx.get(), GGML_TYPE_I8, (int64_t) capacity);
        slot.raw->buffer = slot.buffer.get();
        slot.raw->data   = ggml_backend_buffer_get_base(slot.buffer.get());
        if (slot.raw->data == nullptr ||
            ggml_backend_buffer_init_tensor(slot.buffer.get(), slot.raw) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize expert slot tensor");
        }
        return slot;
    }

    int fd_for(const fs::path & path) {
#if defined(__linux__)
        const std::string key = path.string();
        auto it = fds_.find(key);
        if (it != fds_.end()) {
            return it->second;
        }
        const int fd = open(key.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "failed to open O_DIRECT shard " + key + ": " + std::strerror(errno));
        }
        fds_.emplace(key, fd);
        return fd;
#else
        (void) path;
        throw std::runtime_error("O_DIRECT expert slots require Linux");
#endif
    }

    // Read [offset, offset+len) of a page. offset is 4096-aligned by
    // construction (see stripe_plan); the final stripe carries whatever
    // remainder the page has, which is exactly the tail a single whole-page
    // read submits today, so O_DIRECT sees no length it did not see before.
    void read_page_range(const ExpertPage & page, int fd, void * dst,
                         size_t offset, size_t len) {
#if defined(__linux__)
        ssize_t n = -1;
        do {
            n = pread(fd, dst, len, (off_t) (page.offset + (uint64_t) offset));
        } while (n < 0 && errno == EINTR);
        if (n < 0 || (size_t) n != len) {
            throw std::runtime_error(
                "short O_DIRECT expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " + std::to_string(len) +
                " at +" + std::to_string(offset));
        }
#else
        (void) page;
        (void) fd;
        (void) dst;
        (void) offset;
        (void) len;
#endif
    }

    // Split a page into aligned stripes. Every stripe but the last is a
    // multiple of 4096 (the alignment the staging pool already guarantees and
    // O_DIRECT requires); the last absorbs the remainder. Returns a single
    // whole-page stripe when striping is off or the page is too small to be
    // worth splitting -- that path is byte-for-byte the old behaviour.
    std::vector<std::pair<size_t, size_t>> stripe_plan(uint64_t page_size,
                                                       size_t   n_pageins) const {
        std::vector<std::pair<size_t, size_t>> out;
        const size_t total = (size_t) page_size;
        constexpr size_t kAlign   = 4096;
        constexpr size_t kMinPart = 1u << 20;   // don't split below 1 MiB/stripe
        size_t n = read_stripes_;
        // *** GATE ON THE BATCH BEING READ-SPARSE. ***
        // Striping only pays when a page has no OTHER page to overlap against.
        // With many page-ins in one batch, drain_one_read is already uploading
        // page N while the reader threads pull N+1, so splitting each page just
        // multiplies preads and tensor_set calls.
        // MEASURED 2026-08-05, identical block counts both arms: decode-side
        // dispatch wait 72.62 -> 66.04 s (-9.1%, and consistent in sign across
        // decode/verify-43/verify-3), but PREFILL wait 16.50 -> 18.26 s
        // (+10.7%). Prefill averages 31.4 page-ins per request; decode averages
        // 0.212 and verify 0.465. So stripe the sparse batches and leave the
        // dense ones whole.
        if (n_pageins > stripe_max_pageins_) {
            n = 1;
        }
        if (n > 1 && total / n < kMinPart) {
            n = total / kMinPart;
        }
        if (n <= 1) {
            out.emplace_back(0, total);
            return out;
        }
        const size_t part = ((total / n) / kAlign) * kAlign;
        if (part == 0) {
            out.emplace_back(0, total);
            return out;
        }
        size_t off = 0;
        while (off + part < total) {
            out.emplace_back(off, part);
            off += part;
        }
        out.emplace_back(off, total - off);   // remainder tail
        return out;
    }

    // WP_EXPERT_READ_STRIPES=<n>: read each expert page in n aligned pieces so
    // the dispatch thread uploads stripe k while the reader thread reads k+1.
    // 1 = the original whole-page read, byte-for-byte.
    //
    // WHAT THIS DOES AND DOES NOT FIX. Across pages, read and h2d ALREADY
    // overlap: drain_one_read uploads page N on the dispatch thread while the
    // reader threads pull page N+1. So this only bites when there are too few
    // pages in flight to overlap each other -- i.e. DECODE, where the RX 480
    // averages 0.212 page-ins per request and a lone page really is read-then-
    // upload serial. At prefill it averages 31.4 page-ins per request and is
    // already deeply pipelined, so expect ~nothing there.
    // Measured per page-in on the 480 at decode: read ~5.4 ms, h2d ~3.63 ms.
    // NOTE ns_read is a SPAN and ns_h2d is a SUM, so those two do not simply add
    // -- do not size this change by adding them.
    static size_t read_stripes_from_env() {
        const char * e = std::getenv("WP_EXPERT_READ_STRIPES");
        if (e == nullptr || e[0] == '\0') {
            return 4;
        }
        const long parsed = std::strtol(e, nullptr, 10);
        return parsed > 0 ? (size_t) parsed : (size_t) 1;
    }

    // WP_EXPERT_STRIPE_MAX_PAGEINS=<n>: only stripe batches with at most n
    // page-ins. Above that the pages already overlap each other. Default 4
    // covers decode (0.212 page-ins/request) and verify (0.465) while leaving
    // prefill (31.4) on the whole-page read, which is where striping measured a
    // +10.7% regression.
    static size_t stripe_max_pageins_from_env() {
        const char * e = std::getenv("WP_EXPERT_STRIPE_MAX_PAGEINS");
        if (e == nullptr || e[0] == '\0') {
            return 4;
        }
        const long parsed = std::strtol(e, nullptr, 10);
        return parsed >= 0 ? (size_t) parsed : (size_t) 4;
    }

    ggml_backend_t             backend_ = nullptr;
    ResourcePlan               resources_;
    StagingPool                staging_;
    size_t                     read_stripes_ = read_stripes_from_env();
    size_t                     stripe_max_pageins_ = stripe_max_pageins_from_env();
    TestHooks *                test_hooks_ = nullptr;
    wp::HostTier               host_tier_;
    bool                       host_victim_enabled_ = false;
    // *** TWO LRU BANDS, NOT ONE COUNTER. ***
    // Demand ticks start at kDemandTickBase and prefetch ticks at 0, so EVERY
    // prefetched page is strictly older than EVERY demand-touched page and is
    // chosen as a victim first. A single counter cannot express this: stamping a
    // speculative page with "the tick at read time" TIES with the demand that produced that
    // tick, and select_victim breaks a tie by slot index -- so a speculative page would
    // evict the very page that was just used, which is the pollution this is
    // here to prevent. Within each band the ordinary LRU order still holds, so
    // an older prefetch is evicted before a newer one.
    // The bands cannot meet: reaching kDemandTickBase needs 2^40 prefetched
    // pages, which at 12.75 MB each is ~14 EB of reads.
    static constexpr uint64_t  kDemandTickBase = 1ull << 40;
    uint64_t                   tick_      = kDemandTickBase;
    uint64_t                   spec_tick_ = 0;
    // Use count of the last page evicted. New pages are admitted here rather
    // than at zero. WP_EXPERT_LFU=0 falls back to pure LRU so the two policies
    // can be A/B'd on the same binary.
    uint64_t                   evict_age_ = 0;
    uint64_t                   evictions_ = 0;
    // WP_EXPERT_SPEC_LEASE -- evictions a speculative page survives before it
    // becomes an ordinary eviction candidate. 0 restores the old first-victim
    // behaviour exactly, so the lease is A/B-able on one binary.
    const uint64_t             spec_lease_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_LEASE");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 64;
        return v > 0 ? (uint64_t) v : (uint64_t) 0;
    }();
    // WP_EXPERT_SPEC_LEASE_PREDICTED -- the lease for a page fetched on a GUESS.
    //
    // A flat lease prices certainty and speculation the same, and measured, that
    // is what broke the predictor: ~1.8 predicted tokens per block displaced ~200
    // ground-truth pages, because a predicted page held its slot exactly as long
    // as a page the target was already committed to. Short by default so a guess
    // can occupy capacity nothing better wants and give it up first.
    const uint64_t             spec_lease_pred_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_LEASE_PREDICTED");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 4;
        return v > 0 ? (uint64_t) v : (uint64_t) 0;
    }();
    // The one speculative batch that may be in flight. Holding the Batch holds
    // its slot pins, which is what stops an in-flight read's slot from being
    // recycled underneath it.
    std::unique_ptr<Batch>          spec_batch_;
    std::vector<const ExpertPage *> spec_inflight_;
    std::vector<uint64_t>           spec_leases_;
    std::thread                     host_thread_;
    std::atomic<size_t>             host_pending_{0};
    std::atomic<uint64_t>           host_landed_{0};
    std::atomic<uint64_t>           host_bytes_{0};
    std::atomic<uint64_t>           host_errors_{0};
    uint64_t                        host_skip_bad_  = 0;
    uint64_t                        host_skip_pin_  = 0;
    uint64_t                        host_skip_vram_ = 0;
    uint64_t                        host_skip_tier_ = 0;
    // retire_spec_batch -> complete_batch -> ... never re-enters ensure_batch,
    // but the guard makes that explicit and cheap rather than assumed.
    bool                            spec_recursion_ = false;
    const bool                 lfu_ = [] {
        const char * e = std::getenv("WP_EXPERT_LFU");
        return e == nullptr || e[0] != '0';   // ON by default
    }();
    // Ranking key. With LFU off this is pure LRU, byte for byte what shipped
    // before, so a control arm needs no separate build.
    std::tuple<int, uint64_t, uint64_t> rank(const Slot & s) const {
        // A live lease is the FIRST key, so a leased page loses to every
        // unleased one and is only taken when nothing else can be. That is the
        // deadlock guard: the lease reorders candidates, it never removes them,
        // so select_victim always has something to return.
        const int leased = (s.lease_until > evictions_) ? 1 : 0;
        return { leased, lfu_ ? s.uses : 0, s.tick };
    }
    // Speculative page-in accounting. spec_pageins_/spec_bytes_ are what
    // speculation SPENT; the request stream's n_pagein is what it SAVED. Both are
    // reported, neither is a verdict -- on this rig the drive is ~78% idle during
    // decode, so extra reads there are spending capacity that would otherwise go
    // unused, and totalling bytes as if bandwidth were scarce answers a question
    // the hardware is not asking.
    uint64_t                   spec_pageins_  = 0;
    uint64_t                   spec_bytes_  = 0;
    uint64_t                   spec_errors_ = 0;
    std::vector<Slot>          slots_;
    std::vector<int>           reserve_blocks_;
    std::map<std::string, int> fds_;
};

ExpertSlotPool::Batch::Batch(Batch && other) noexcept :
    owner_(other.owner_),
    entries_(std::move(other.entries_)),
    state_(std::move(other.state_)),
    workers_(std::move(other.workers_)),
    completed_(other.completed_),
    ns_lookup_(other.ns_lookup_),
    ns_read_(other.ns_read_),
    n_resident_(other.n_resident_),
    n_pagein_(other.n_pagein_),
    n_pagein_reserved_(other.n_pagein_reserved_),
    n_pagein_general_(other.n_pagein_general_),
    bytes_read_(other.bytes_read_),
    n_host_hit_(other.n_host_hit_),
    n_host_demote_(other.n_host_demote_),
    ns_host_get_(other.ns_host_get_),
    host_bytes_(other.host_bytes_),
    ns_h2d_(other.ns_h2d_),
    bytes_h2d_(other.bytes_h2d_) {
    other.owner_ = nullptr;
}

ExpertSlotPool::Batch::~Batch() {
    if (owner_ != nullptr) {
        owner_->abandon_batch(*this);
    }
}

void ExpertSlotPool::Batch::complete() {
    if (owner_ == nullptr) {
        throw std::logic_error("expert batch has no owner");
    }
    owner_->complete_batch(*this);
}

void ExpertSlotPool::Batch::complete_upto(size_t entry_end) {
    if (owner_ == nullptr) {
        throw std::logic_error("expert batch has no owner");
    }
    owner_->complete_batch_upto(*this, entry_end);
}

void attach_weight(
        ggml_tensor * tensor, ggml_backend_buffer_t buffer, void * base,
        uint64_t offset) {
    tensor->buffer = buffer;
    tensor->data   = (uint8_t *) base + offset;
    if (ggml_backend_buffer_init_tensor(buffer, tensor) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("failed to attach expert weight tensor to slot");
    }
}

class Worker {
public:
    Worker(
            Catalog catalog,
            const std::string & device,
            int slots,
            uint64_t host_budget_bytes,
            uint64_t host_victim_bytes,
            TestHooks * test_hooks,
            const std::vector<int> & resident_expert_blocks,
            const std::vector<int> & expert_reserve_blocks,
            uint64_t expert_reserve_bytes) :
        catalog_(std::move(catalog)),
        backend_(init_backend(device)),
        resident_(backend_.get(), catalog_, resident_expert_blocks),
        pool_(
            backend_.get(),
            plan_resources(
                resource_pages(catalog_), slots, host_budget_bytes,
                resident_.pinned_bytes(), expert_reserve_blocks,
                expert_reserve_bytes),
            host_victim_bytes,
            test_hooks, expert_reserve_blocks),
        compute_galloc_(ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_.get()))),
        slots_(pool_.resources().slot_count) {
        if (!compute_galloc_) {
            throw std::runtime_error("failed to create expert graph allocator");
        }
        // The pool logs demand and speculative page-ins into the Worker's handle
        // so every event lands in one ordered stream. Safe here: hint_log_ has a
        // default member initialiser, so it is open before the body runs.
        pool_.set_spec_log(hint_log_);
        stats_.set_staging_kind(pool_.staging_kind());
        std::cerr << "WARN wp expert worker: pinned_pages="
                  << resident_.pinned_pages()
                  << " pinned_bytes=" << pool_.resources().pinned_bytes
                  << " slot_count=" << pool_.resources().slot_count
                  << " slot_budget_bytes=" << pool_.resources().slot_budget_bytes
                  << std::endl;
        if (pool_.resources().requested_reserved_bytes != 0) {
            std::cerr << "WARN wp expert worker: reserved_bytes="
                      << pool_.resources().reserved_bytes
                      << " reserved_slots=" << pool_.resources().reserved_slot_count
                      << " general_slots=" << pool_.resources().general_slot_count
                      << std::endl;
            if (pool_.resources().named_reservable_bytes < pool_.resources().requested_reserved_bytes) {
                std::cerr << "WARN wp expert worker: reservation clamped to named pageable bytes="
                          << pool_.resources().named_reservable_bytes << std::endl;
            }
        }
        // Runs only under WP_SELF_BENCH=1; no-op otherwise. Placed after the
        // slot pool is built so the backend is in the same state it will serve
        // requests in.
        // PRE-ALLOCATE THE IO BUFFER AT STARTUP. The first device-buffer
        // allocation made AFTER the slot pool exists costs 1394 ms on the RX 480
        // (vs 0.3 ms on the 1070) -- one 1 MiB host-visible allocation behind
        // 5.35 GB of device-local slots. Paid lazily on the first request it
        // amortised to 0.87 ms on EVERY request and was 95% of prepare_io.
        // It is a one-time cost, so pay it here with the rest of startup.
        {
            RequestStats warmup;
            // *** THE 1 MiB FLOOR WAS SIZED FOR DECODE AND RE-OPENS THE BUG IT FIXED. ***
            // prepare_io needs n_embd * n_tokens * sizeof(f32) TWICE (input + result).
            // At n_embd=4096 that is 8 KB per buffer for a decode step (n_tokens=1)
            // but 16.8 MB for a prefill ubatch (n_tokens=512) -- so with a 1 MiB floor
            // ANY real prompt forces the io buffer to grow WHILE SERVING, which is
            // precisely the allocation this pre-allocation exists to avoid.
            //
            // MEASURED 2026-08-03 on the RX 480, 739-token prompt, n_ubatch=512:
            //   wp io-buffer grow #2: 1048576 -> 7307264      (during serving)
            //   wp io-buffer grow #3: 7307264 -> 16777216     (during serving)
            //   submit_us_max 219828 -> 1074753 -> 1290431    <- 1.29 SECOND submits
            //   submit_hist >=8ms: 62 on the 480 vs 14 on the 1070
            // The 480 measured 2.35x the 1070's per-request submit during prefill and
            // only 1.26x during decode -- the gap tracks n_tokens because the GROWTHS
            // track n_tokens. The card is not the problem; this floor is.
            // Yesterday's parity result stands: it was taken at a 6-token prompt, where
            // the buffer never outgrows 1 MiB and no in-serving allocation ever happens.
            //
            // WP_IO_PREALLOC_TOKENS overrides the assumed max ubatch. Default 512 =
            // llama.cpp's default n_ubatch. COST IS TRIVIAL: 16.8 MB at 512, 33.6 MB at
            // 1024, against an 8 GB card -- and it must be raised ALONGSIDE n_ubatch,
            // or the n_ubatch lever re-triggers this exact stall at the larger size.
            uint32_t prealloc_tokens = 512;
            if (const char * env = std::getenv("WP_IO_PREALLOC_TOKENS")) {
                if (env[0] != '\0') {
                    prealloc_tokens = (uint32_t) std::strtoul(env, nullptr, 10);
                }
            }
            size_t want = 1u << 20;
            if (prealloc_tokens > 0) {
                // Mirror prepare_io's layout: a padded input plus an equal result,
                // with slack for buffer-type alignment.
                const size_t one =
                    (size_t) catalog_.descriptor.hparams.n_embd * prealloc_tokens * sizeof(float);
                want = std::max(want, 2 * (one + 65536));
            }
            fprintf(stderr,
                    "wp io-buffer prealloc: %zu bytes for n_tokens<=%u (n_embd=%u)\n",
                    want, prealloc_tokens, (unsigned) catalog_.descriptor.hparams.n_embd);
            grow_io_buffer(want, warmup);
        }
        stats_.set_probe_backend(backend_.get());
        run_self_bench(backend_.get(),
                       catalog_.descriptor.hparams.n_embd,
                       catalog_.descriptor.hparams.n_ff_exp);
        build_keepalive();
    }

    // WP_KEEPALIVE_US=N (0 = off): while waiting for the next request, submit a
    // trivial graph every N microseconds instead of leaving the GPU idle.
    //
    // WHY. On the RX 480 the cost of a submit depends on how long the GPU idled
    // beforehand. Measured 2026-08-02 on this exact expert graph, clocks already
    // pinned to max via power_dpm_force_performance_level=high:
    //     idle gap      200us    1ms     3ms
    //     idle          284      512     547   us/expert
    //     keepalive     163      163     165   us/expert
    // The keepalive removes the penalty entirely and is FLAT in gap length. It
    // is not a clock effect -- sclk and mclk are both pinned at maximum while
    // this happens -- so it is gating below the clock level, and occupying the
    // GPU is the only lever available without a kernel parameter.
    //
    // Submitted from THIS thread, between poll() timeouts on the request socket.
    // Do not move it to a background thread: Vulkan command pools have thread
    // affinity, so concurrent submits risk corruption even behind a mutex.
    void keepalive_tick() {
        if (keepalive_graph_ != nullptr) {
            ggml_backend_graph_compute(backend_.get(), keepalive_graph_);
        }
    }

    bool keepalive_enabled() const { return keepalive_us_ > 0 && keepalive_graph_ != nullptr; }
    int  keepalive_us()      const { return keepalive_us_; }

    // Record a prefetch hint. READS NOTHING YET -- see the frame-loop
    // comment. Experts outside this worker's shard are counted separately rather
    // than rejected: the spine routes by the same static hash the dispatch uses,
    // so a nonzero foreign count means the two disagree, which is a spine bug
    // that would otherwise show up only as prefetch mysteriously not helping.
    void note_prefetch_hint(const pipe_expert_prefetch_hint & hint) {
        ++hint_frames_;
        hint_experts_ += hint.expert_ids.size();
        log_hint_ids(hint);
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), hint.layer)) {
            hint_foreign_layer_ += hint.expert_ids.size();
            log_prefetch_hints();
            return;
        }
        for (int32_t expert_id : hint.expert_ids) {
            if (expert_id < catalog_.descriptor.expert_first ||
                expert_id > catalog_.descriptor.expert_last ||
                catalog_.pages.count({ hint.layer, expert_id }) == 0) {
                ++hint_foreign_expert_;
                continue;
            }
            // Resolved here, on the frame thread, so the idle path does no map
            // lookups between reads. Ascending on the wire, so this queue is in
            // ascending page order per layer -- which is the order that lets the
            // drive read something closer to a stream than a random walk.
            if (spec_enabled_) {
                const ExpertPage * page = &catalog_.pages.at({ hint.layer, expert_id });
                // pool_.host_landing_available() is the load-bearing half of
                // this condition. Without it a predicted hint goes onto a queue
                // that has nowhere to drain to -- spec_host_submit returns 0 with
                // no host tier -- and the prediction is silently discarded rather
                // than fetched. That is worse than not predicting: the arm looks
                // like it ran and measured nothing.
                if (hint.provenance == PIPE_HINT_PREDICTED && spec_host_enabled_ &&
                    pool_.host_landing_available()) {
                    // A GUESS DOES NOT GET A VRAM SLOT. It lands in host RAM,
                    // where a wrong guess costs only the bandwidth that fetched
                    // it and a right one is promoted over PCIe instead of being
                    // re-read from NVMe.
                    host_queue_.push_back(page);
                } else {
                    spec_queue_.emplace_back(page,
                                             hint.provenance == PIPE_HINT_PREDICTED
                                                 ? pool_.spec_lease_predicted()
                                                 : pool_.spec_lease());
                }
            }
        }
        log_prefetch_hints();
    }

    void note_prefetch_hint_bad() {
        ++hint_bad_;
        log_prefetch_hints();
    }


    // Drop the queue. Called when the connection has been idle long enough that
    // the hinted layer is certainly behind us -- speculating on a layer already
    // computed is a read with no possible upside.
    void drop_spec_work() {
        spec_dropped_ += spec_queue_.size() + host_queue_.size();
        spec_queue_.clear();
        host_queue_.clear();
    }

    // Keep the speculative pipeline moving. Called from the idle pump.
    //
    // TWO DISTINCT JOBS, and neither of them blocks:
    //   1. harvest a batch that already landed, which frees its slot pins;
    //   2. submit the next chunk if nothing is in flight.
    // Between calls the read runs on the reader threads, so it proceeds WHILE
    // the dispatch thread polls or computes. That overlap is the entire reason
    // this path exists; the previous version completed the read inline here and
    // therefore never overlapped anything.
    //
    // H2D still happens on this thread inside drain_one_read -- Vulkan command
    // pools have thread affinity, the same constraint keepalive_tick documents.
    // Only the DISK read moved off it, which is the part worth moving.
    //
    // Returns true if it did any work worth staying awake for.
    bool spec_pagein_step() {
        // Host landings run on their own reader thread and never touch the GPU,
        // so they are reaped and refilled independently of the VRAM path.
        pool_.spec_host_reap();
        if (!pool_.spec_host_in_flight() && !host_queue_.empty()) {
            const size_t take = std::min(spec_chunk_, host_queue_.size());
            std::vector<const ExpertPage *> chunk(
                host_queue_.begin(), host_queue_.begin() + (ptrdiff_t) take);
            host_queue_.erase(host_queue_.begin(), host_queue_.begin() + (ptrdiff_t) take);
            if (pool_.spec_host_submit(chunk) != 0) {
                return true;
            }
        }
        if (pool_.spec_in_flight()) {
            return pool_.spec_pagein_poll(false);
        }
        if (spec_queue_.empty()) {
            return false;
        }
        const size_t take = std::min(spec_chunk_, spec_queue_.size());
        std::vector<const ExpertPage *> chunk;
        std::vector<uint64_t>           leases;
        chunk.reserve(take);
        leases.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            chunk.push_back(spec_queue_[i].first);
            leases.push_back(spec_queue_[i].second);
        }
        spec_queue_.erase(spec_queue_.begin(), spec_queue_.begin() + (ptrdiff_t) take);
        return pool_.spec_pagein_submit(chunk, leases) != 0;
    }

    // A speculative read in flight is work in progress, not idleness -- the pump
    // must keep spinning to harvest it even when the queue is empty.
    bool has_spec_work() const {
        return !spec_queue_.empty() || !host_queue_.empty() ||
               pool_.spec_in_flight() || pool_.spec_host_in_flight();
    }

    // Something to SUBMIT right now, as opposed to something in flight. Only the
    // former justifies a zero-timeout poll: when a read is already running on a
    // reader thread we are waiting on the disk, not on ourselves, and spinning
    // would burn a core for the 3-5 ms of the read while doing nothing.
    bool has_spec_submit_work() const {
        return (!spec_queue_.empty() && !pool_.spec_in_flight()) ||
               (!host_queue_.empty() && !pool_.spec_host_in_flight());
    }

    // The counter line, built in ONE place so stderr and WP_HINT_LOG cannot
    // drift apart. spec_pageins/spec_bytes sit next to the request stream's own
    // n_pagein and bytes_read because spend and saving are only interpretable
    // together.
    std::string prefetch_hint_line() const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "frames=%llu experts=%llu "
                      "foreign_layer=%llu foreign_expert=%llu malformed=%llu "
                      "spec_pageins=%llu spec_bytes=%llu spec_errors=%llu "
                      "spec_dropped=%llu spec_queue_left=%zu "
                      "host_landed=%llu host_bytes=%llu host_errors=%llu "
                      "host_promoted=%llu host_wasted=%llu "
                      "host_skip[bad/pin/vram/tier]=%llu/%llu/%llu/%llu",
                      (unsigned long long) hint_frames_,
                      (unsigned long long) hint_experts_,
                      (unsigned long long) hint_foreign_layer_,
                      (unsigned long long) hint_foreign_expert_,
                      (unsigned long long) hint_bad_,
                      (unsigned long long) pool_.spec_pageins(),
                      (unsigned long long) pool_.spec_bytes(),
                      (unsigned long long) pool_.spec_errors(),
                      (unsigned long long) spec_dropped_,
                      spec_queue_.size(),
                      (unsigned long long) pool_.host_landed(),
                      (unsigned long long) pool_.host_spec_bytes(),
                      (unsigned long long) pool_.host_spec_errors(),
                      (unsigned long long) pool_.host_spec_promotions(),
                      (unsigned long long) pool_.host_spec_wasted(),
                      (unsigned long long) pool_.host_skip_bad(),
                      (unsigned long long) pool_.host_skip_pin(),
                      (unsigned long long) pool_.host_skip_vram(),
                      (unsigned long long) pool_.host_skip_tier());
        return buf;
    }

    // R -- GROUND TRUTH: the experts a dispatch actually asked for. Without this
    // in the same stream, a speculative page-in that was never selected cannot be
    // told apart from one that was selected but arrived too late, and the whole
    // log answers neither question. Duplicates WP_REF_LOG on purpose -- see
    // set_spec_log for why a second file will not do.
    void log_reference(int32_t layer, const std::vector<pipe_expert_assignment> & assignments) {
        if (hint_log_ == nullptr) {
            return;
        }
        std::fprintf(hint_log_, "R %d", layer);
        for (const pipe_expert_assignment & a : assignments) {
            std::fprintf(hint_log_, " %d", a.expert_id);
        }
        std::fputc('\n', hint_log_);
        std::fflush(hint_log_);
    }

    void report_prefetch_hints() const {
        if (hint_frames_ == 0 && hint_bad_ == 0) {
            return;
        }
        std::fprintf(stderr, "wp-expert-worker prefetch hints: %s\n",
                     prefetch_hint_line().c_str());
    }

    pipe_expert_hello hello() const {
        pipe_expert_hello hello;
        hello.role          = PIPE_EXPERT_ROLE_WORKER;
        hello.hidden_type   = PIPE_HIDDEN_F16;
        hello.n_embd        = catalog_.descriptor.hparams.n_embd;
        hello.n_ff_exp      = catalog_.descriptor.hparams.n_ff_exp;
        hello.n_expert      = catalog_.descriptor.hparams.n_expert;
        hello.n_expert_used = catalog_.descriptor.hparams.n_expert_used;
        hello.expert_first  = catalog_.descriptor.expert_first;
        hello.expert_last   = catalog_.descriptor.expert_last;
        hello.n_slots       = (uint32_t) slots_;
        hello.layers        = catalog_.layers;
        hello.model_identity = source_model_identity(
            catalog_.descriptor.input_model, catalog_.descriptor.model_files);
        hello.shard_identity =
            catalog_.descriptor.identity_algorithm + ":" +
            catalog_.descriptor.identity_value;
        return hello;
    }

    pipe_expert_partial dispatch(
            const pipe_expert_dispatch_req & request,
            RequestStats & request_stats) {
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), request.layer)) {
            throw pipe_protocol_error(
                PIPE_ERR_EXPERT_LAYER,
                "worker does not serve layer " + std::to_string(request.layer));
        }
        if (request.assignments.size() >
            (size_t) catalog_.descriptor.hparams.n_expert) {
            throw pipe_protocol_error(
                PIPE_ERR_BAD_FRAME,
                "expert dispatch has more assignments than model experts");
        }
        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : request.assignments) {
            if (!seen_experts.insert(assignment.expert_id).second) {
                throw pipe_protocol_error(
                    PIPE_ERR_BAD_FRAME,
                    "expert dispatch repeats expert " + std::to_string(assignment.expert_id));
            }
            if (assignment.expert_id < catalog_.descriptor.expert_first ||
                assignment.expert_id > catalog_.descriptor.expert_last ||
                catalog_.pages.count({ request.layer, assignment.expert_id }) == 0) {
                throw pipe_protocol_error(
                    PIPE_ERR_EXPERT_RANGE,
                    "worker does not serve expert " + std::to_string(assignment.expert_id));
            }
        }

        // Already f32 on the wire as of PIPE_VERSION 4; this used to widen a
        // f16 value back to f32, which recovered the storage type but NOT the
        // ~3e-4 of precision the spine had already thrown away.
        const std::vector<float> & activation = request.activations;
        const bool measure = stats_.enabled();
        const std::chrono::steady_clock::time_point lookup_started =
            measure ? std::chrono::steady_clock::now() :
                      std::chrono::steady_clock::time_point();
        std::vector<const ExpertPage *> pages;
        pages.reserve(request.assignments.size());
        for (const pipe_expert_assignment & assignment : request.assignments) {
            pages.push_back(&catalog_.pages.at({
                request.layer, assignment.expert_id
            }));
        }
        ExpertSlotPool::Batch batch = pool_.ensure_batch(
            pages, measure, lookup_started);
        if (measure) {
            request_stats.ns_lookup  = batch.lookup_ns();
            request_stats.n_resident      = batch.n_resident();
            request_stats.n_pagein     = batch.n_pagein();
            request_stats.n_pagein_reserved = batch.n_pagein_reserved();
            request_stats.n_pagein_general = batch.n_pagein_general();
            request_stats.n_host_hit = batch.n_host_hit();
            request_stats.n_host_demote = batch.n_host_demote();
            request_stats.bytes_read = batch.bytes_read();
            request_stats.ns_host_get = batch.ns_host_get();
            request_stats.host_bytes = batch.host_bytes();
        }

        const std::chrono::steady_clock::time_point compute_started =
            measure ? std::chrono::steady_clock::now() :
                      std::chrono::steady_clock::time_point();
        const size_t result_size =
            (size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd;
        std::vector<float> sum(result_size, 0.0f);
        bool have_hits = false;
        bool have_pageins = false;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            have_hits |= batch.is_resident(i);
            have_pageins |= !batch.is_resident(i);
        }
        // PHASE TIMERS. ns_compute is the wall span of this whole section, but
        // ns_read/h2d/submit/readback only summed to 78% of it on the RX 480
        // (8.95 s of 11.53 s) -- 1.34 ms per request attributed to nothing.
        // These close the accounting: prepare_io (activation upload + io buffer
        // growth), the blocking wait in batch.complete() (thread join + I/O),
        // and the fp32->fp16 encode of the reply.
        auto phase = std::chrono::steady_clock::now();
        auto lap = [&measure, &phase]() -> uint64_t {
            if (!measure) return 0;
            const auto now = std::chrono::steady_clock::now();
            const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - phase).count();
            phase = now;
            return ns;
        };
        // *** DETERMINISM: ONE PASS, INDEX ORDER, AFTER complete(). ***
        // This used to compute the RESIDENT experts first (overlapping the
        // page-in I/O), then the newly-paged-in ones, chaining the two partial
        // sums. That made the floating-point ASSOCIATION depend on which pages
        // happened to be resident when the graph was built:
        //     all resident   : total = h1 + h2 + h3 + h4
        //     expert 2 a miss: total = (h1 + h3 + h4) + m2
        // Same experts, same weights, different rounding -- and membership turns
        // on whether the PREVIOUS request's page-ins had landed yet, i.e. on I/O
        // timing. FP addition is not associative, so the worker returned a
        // slightly different sum run to run. f16 KV absorbed it; turbo4 amplified
        // a last-bit delta into a 4-bit centroid flip and thence a router top-k
        // change, which is how it surfaced (2026-08-03: 3 of 3 pairs divergent
        // with the CPU DSpark worker, 0 of 12 without it).
        // Measured cost of dropping the overlap: see WP_EXPERT_OVERLAP below.
        static const bool overlap = [] {
            const char * e = std::getenv("WP_EXPERT_OVERLAP");
            return e != nullptr && e[0] == '1';   // default OFF = deterministic
        }();
        // WP_EXPERT_COMPUTE_CHUNKS=<n>: split the expert compute into n fixed
        // index chunks so all but the last can run while the tail of the page-in
        // reads is still in flight. 1 = the original strictly-serial path.
        // Unlike WP_EXPERT_OVERLAP this does NOT trade determinism -- see the
        // note at the compute loop below.
        // NOTE: this is clamped to the assignment count AND gated on n_pagein > 0
        // at the loop. Clamping alone was NOT enough and the earlier comment here
        // claiming "decode is unaffected" was wrong -- min(4, 2) is 2, so 2-expert
        // decode/verify requests were splitting into two submits for nothing.
        static const size_t s_compute_chunks = [] {
            const char * e = std::getenv("WP_EXPERT_COMPUTE_CHUNKS");
            if (e == nullptr || e[0] == '\0') {
                return (size_t) 4;
            }
            const long parsed = std::strtol(e, nullptr, 10);
            return parsed > 0 ? (size_t) parsed : (size_t) 1;
        }();
        if (!request.assignments.empty()) {
            prepare_io(activation, request.n_tokens, request_stats);
            request_stats.ns_prep = lap();
            if (overlap && have_hits) {
                compute_batch(
                    request, pages, batch, /* hits = */ true,
                    /* add_previous = */ false, request_stats);
            }
            request_stats.ns_hits = lap();
        }
        // *** READ / COMPUTE OVERLAP (WP_EXPERT_COMPUTE_CHUNKS, default 4). ***
        //
        // THE COST THIS ATTACKS. complete() blocks until EVERY expert page has
        // landed and only then runs one compute pass, so read and compute are
        // strictly serial. Measured 2026-08-05 on the RX 480 at 659-token
        // prefill: read 198.3 ms/request against compute 43.8 ms/request. An
        // expert only needs ITS OWN page, so all but the last chunk's compute
        // can hide under the reads still in flight.
        //
        // WHY CHUNK BY INDEX AND NOT BY ARRIVAL. Computing whatever happens to
        // be ready is what WP_EXPERT_OVERLAP does, and it is off by default
        // precisely because it makes the FP summation order depend on I/O
        // timing -- same experts, different association, a last-bit delta that
        // turbo4 amplified into a different token. Fixed index chunks keep the
        // accumulation order identical to the serial path no matter when reads
        // land, so this is deterministic BY CONSTRUCTION rather than by luck.
        //
        // Chunk count is a tuning knob, not a correctness one: the summation
        // order is the same at every value, so arms differ only in speed. =1
        // restores the exact serial path.
        if (!overlap && !request.assignments.empty()) {
            const size_t n_assign = request.assignments.size();
            // *** GATE ON THERE BEING READS TO HIDE UNDER. ***
            // Clamping to the assignment count is NOT enough: min(4, 2) == 2, so a
            // 2-expert request still split into TWO graph submits. Decode averages
            // ~1.5 experts and verify ~2.0, and at 85% residency those requests
            // usually have NOTHING in flight to overlap -- so the split bought a
            // second submit (~0.45 ms on the RX 480, plus its idle-recovery gap
            // penalty) for zero benefit, on the decode critical path. verify is
            // 40.8 s of the 74.0 s decode dispatch wait, so this is not a rounding
            // error. n_pagein == 0 means every expert is already resident and the
            // serial path is strictly better.
            const size_t chunks   = batch.n_pagein() == 0
                ? 1
                : std::max<size_t>(1, std::min(s_compute_chunks, n_assign));
            for (size_t c = 0; c < chunks; ++c) {
                const size_t beg = n_assign * c / chunks;
                const size_t end = n_assign * (c + 1) / chunks;
                if (beg == end) {
                    continue;
                }
                batch.complete_upto(end);
                request_stats.ns_wait += lap();
                compute_batch(
                    request, pages, batch, /* hits = */ true,
                    /* add_previous = */ c > 0, request_stats,
                    /* all_experts = */ true, /* force_dense = */ false,
                    beg, end);
                request_stats.ns_pagein_compute += lap();
            }
        }
        // Always: joins the reader threads, finalises ns_read and rethrows the
        // first read error. Cheap and already drained after the loop above.
        batch.complete();
        request_stats.ns_wait += lap();
        if (measure) {
            request_stats.ns_read = batch.read_ns();
            request_stats.ns_h2d    = batch.ns_h2d();
            request_stats.bytes_h2d = batch.bytes_h2d();
        }
        if (overlap && have_pageins) {
            compute_batch(
                request, pages, batch, /* hits = */ false,
                /* add_previous = */ have_hits, request_stats);
        }
        request_stats.ns_pagein_compute += lap();
        if (!request.assignments.empty()) {
            read_result(sum, request_stats);
        }
        request_stats.ns_result = lap();

        // *** WP_SELFCHECK=1: EQUIVALENCE PROBE (default OFF, diagnostic only). ***
        // THE INVARIANT UNDER TEST: an expert's contribution to a token must not
        // depend on how many OTHER tokens shared its batch. gather changes the graph
        // shape with batch width (get_rows -> matmul at compacted width ->
        // get_rows_back) while dense does not, so if the two disagree the expert path
        // is batch-width-dependent and THAT is the foundational bug.
        // WHY THIS MATTERS (2026-08-04): conf_min changes ONLY n_draft, i.e. the verify
        // batch width -- yet it changed the generated text at temperature 0. Under
        // correct speculative decoding the target's logits cannot depend on the draft,
        // so something downstream of batch width is not width-invariant. This probe
        // answers that directly instead of inferring it from throughput.
        // PRECEDENT: this exact code already had one width-dependent defect -- the
        // routing-weight orientation, "correct only at n_tokens == 1, so decode looked
        // fine while PREFILL was corrupted and poisoned the KV cache."
        static const bool s_selfcheck = [] {
            const char * e = std::getenv("WP_SELFCHECK");
            return e != nullptr && e[0] == '1';
        }();
        if (s_selfcheck && !request.assignments.empty() && !sum.empty()) {
            std::vector<float> dense(sum.size(), 0.0f);
            compute_batch(
                request, pages, batch, /* hits = */ true,
                /* add_previous = */ false, request_stats,
                /* all_experts = */ true, /* force_dense = */ true);
            read_result(dense, request_stats);
            double max_abs = 0.0, max_rel = 0.0, sum_abs = 0.0;
            size_t worst = 0;
            for (size_t i = 0; i < sum.size(); ++i) {
                const double a = (double) sum[i], b = (double) dense[i];
                const double d = std::fabs(a - b);
                sum_abs += d;
                if (d > max_abs) { max_abs = d; worst = i; }
                const double den = std::max(std::fabs(a), std::fabs(b));
                if (den > 1e-6) { max_rel = std::max(max_rel, d / den); }
            }
            std::fprintf(stderr,
                "WP_SELFCHECK layer=%d n_tokens=%u n_exp=%zu "
                "max_abs=%.6g max_rel=%.6g mean_abs=%.6g worst_i=%zu "
                "gather=%.6g dense=%.6g %s\n",
                request.layer, request.n_tokens, request.assignments.size(),
                max_abs, max_rel, sum_abs / (double) sum.size(), worst,
                (double) sum[worst], (double) dense[worst],
                max_rel > 1e-3 ? "*** MISMATCH ***" : "ok");
            // NOTE: `sum` already holds the GATHER result (read above) and is what the
            // caller ships to the spine. The dense recompute overwrote only the io
            // buffer's result region, which is not read again. Do NOT re-read into
            // `sum` here -- that would ship the dense result and silently change what
            // the probe is supposed to be observing.
        }
        if (measure) {
            request_stats.ns_compute =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - compute_started).count();
        }

        pipe_expert_partial response;
        response.layer    = request.layer;
        response.n_tokens = request.n_tokens;
        // f32, not f16: this subtotal is only PART of the layer's expert sum -- the
        // spine adds the other workers' subtotals to it. Rounding a partial sum to f16
        // put a ~5e-4 relative error at the expert->worker partition boundary, so the
        // layer output depended on which worker happened to get which expert.
        response.partial.assign(sum.begin(), sum.end());
        request_stats.ns_encode = lap();
        return response;
    }

    const ResourcePlan & resources() const {
        return pool_.resources();
    }

    int pinned_pages() const {
        return resident_.pinned_pages();
    }

    bool stats_enabled() const {
        return stats_.enabled();
    }

    void record_stats(const RequestStats & request, size_t n_experts) {
        stats_.record(request, n_experts);
    }

private:
    // ---- prefetch hints (see note_prefetch_hint) ----
    //
    // WP_EXPERT_SPEC_PAGEIN=1 arms speculative page-ins. DEFAULT OFF and separate from the
    // spine.s WP_PREFETCH_HINT on purpose: with hints on and speculation off, a run
    // reads exactly what the config of record reads while still reporting what
    // was offered, so "the hint is wrong" and "the speculation is wrong" are two
    // separate experiments instead of one confounded one.
    const bool         spec_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PAGEIN");
        return e != nullptr && e[0] == '1';
    }();
    // WP_EXPERT_SPEC_CHUNK -- pages read per idle step. 1 by default: this is
    // the worst-case delay a real request can inherit from a speculative read already in
    // progress, about one 12.75 MB O_DIRECT read. Raising it trades that latency
    // for fewer round trips through poll().
    const size_t       spec_chunk_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_CHUNK");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 1;
        return v > 0 ? (size_t) v : (size_t) 1;
    }();
    // (page, lease) -- provenance is resolved to a lease at enqueue, so nothing
    // downstream has to know where a page came from.
    std::deque<std::pair<const ExpertPage *, uint64_t>> spec_queue_;
    // Predicted pages bound for host RAM. Separate queue, not a flag on the
    // other one: they take a different read path to a different destination.
    std::deque<const ExpertPage *> host_queue_;
    // WP_EXPERT_SPEC_HOST=0 sends predictions back to VRAM on their short lease,
    // which is the arm this is meant to beat.
    const bool spec_host_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_HOST");
        return e == nullptr || e[0] != '0';
    }();
    uint64_t           spec_dropped_        = 0;
    uint64_t           hint_frames_         = 0;
    uint64_t           hint_experts_        = 0;
    uint64_t           hint_foreign_layer_  = 0;
    uint64_t           hint_foreign_expert_ = 0;
    uint64_t           hint_bad_            = 0;

    // WP_HINT_LOG=path -- the counter line, appended after every hint frame and
    // fflushed, so `tail -1` is the final answer.
    //
    // WHY THIS EXISTS: report_prefetch_hints() writes to stderr only on a CLEAN
    // CONNECTION CLOSE, and the harness SIGKILLs workers at teardown. Arm 1
    // therefore produced NO foreign_expert number at all -- and foreign_expert
    // is the routing-agreement check, the one counter that proves spine and
    // worker resolve (layer, expert) through the same static hash. A disagreement
    // there would otherwise surface only much later, disguised as "prefetch
    // mysteriously does not help". Same failure and same fix as WP_PAGEIN_LOG's
    // per-line fflush, which was added after that flag's first run produced two
    // 0-byte logs.
    //
    // One line per hint frame rather than one at exit ON PURPOSE: it survives any
    // death, and it dates the FIRST frame at which a foreign count appears, which
    // is the next question if one ever does. ~150 bytes per frame against a frame
    // that already crossed WireGuard.
    FILE * const       hint_log_ = [] {
        const char * p = std::getenv("WP_HINT_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    void log_prefetch_hints() {
        if (hint_log_ == nullptr) {
            return;
        }
        std::fprintf(hint_log_, "C %s\n", prefetch_hint_line().c_str());
        std::fflush(hint_log_);
    }

    // H -- what was PREDICTED, as received on the wire, before any shard filter.
    // Logged even for foreign ids: if spine and worker ever disagree about who
    // owns an expert, the ids are the evidence and the counters are only the
    // alarm.
    void log_hint_ids(const pipe_expert_prefetch_hint & hint) {
        if (hint_log_ == nullptr) {
            return;
        }
        std::fprintf(hint_log_, "H %d", hint.layer);
        for (int32_t expert_id : hint.expert_ids) {
            std::fprintf(hint_log_, " %d", expert_id);
        }
        std::fputc('\n', hint_log_);
        std::fflush(hint_log_);
    }


    // ---- keepalive (see keepalive_tick) ----
    int                keepalive_us_    = 0;
    ggml_context     * keepalive_ctx_   = nullptr;
    ggml_gallocr_t     keepalive_alloc_ = nullptr;
    ggml_cgraph      * keepalive_graph_ = nullptr;

    void build_keepalive() {
        const char * e = std::getenv("WP_KEEPALIVE_US");
        keepalive_us_ = (e != nullptr && e[0] != '\0') ? atoi(e) : 0;
        if (keepalive_us_ <= 0) {
            return;
        }
        // Deliberately the smallest graph that still reaches the GPU: one add on
        // a 1-element tensor. It exists to occupy the device, not to compute.
        ggml_init_params p = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        keepalive_ctx_ = ggml_init(p);
        if (keepalive_ctx_ == nullptr) {
            keepalive_us_ = 0;
            return;
        }
        ggml_tensor * a   = ggml_new_tensor_1d(keepalive_ctx_, GGML_TYPE_F32, 1);
        ggml_tensor * sum = ggml_add(keepalive_ctx_, a, a);
        keepalive_graph_  = ggml_new_graph(keepalive_ctx_);
        ggml_build_forward_expand(keepalive_graph_, sum);
        keepalive_alloc_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_.get()));
        if (keepalive_alloc_ == nullptr ||
            !ggml_gallocr_alloc_graph(keepalive_alloc_, keepalive_graph_)) {
            if (keepalive_alloc_) { ggml_gallocr_free(keepalive_alloc_); keepalive_alloc_ = nullptr; }
            ggml_free(keepalive_ctx_); keepalive_ctx_ = nullptr;
            keepalive_graph_ = nullptr;
            keepalive_us_ = 0;
            return;
        }
        // Warm it once so the first real gap does not pay pipeline compilation.
        ggml_backend_graph_compute(backend_.get(), keepalive_graph_);
        fprintf(stderr, "wp keepalive enabled: every %d us while idle\n", keepalive_us_);
    }

    void grow_io_buffer(size_t size, RequestStats & request_stats) {
        if (io_buffer_ && io_buffer_size_ >= size) {
            return;
        }
        // GEOMETRIC GROWTH. This used to allocate EXACTLY `size`, so a run whose
        // requests arrive at increasing n_tokens reallocated once per new high-
        // water mark (observed: 65536 -> 81936 -> 81952 -> 131072 -> 163920).
        // Only 5 allocations in a 64-token run -- but on Vulkan each one frees
        // the previous device buffer, and that stalls: measured 1.59 s total,
        // ~318 ms per growth, which amortised to 1.04 ms on EVERY request and
        // was 95% of prepare_io on the RX 480 (vs 0.001 ms on CUDA, where frees
        // are cheap). Doubling makes growth O(log n) instead of O(distinct
        // sizes), and the floor means the common decode case allocates once.
        static constexpr size_t IO_FLOOR = 1u << 20;   // 1 MiB: ~16 tokens of f32 activations
        size_t want = std::max(size, IO_FLOOR);
        if (io_buffer_ && want < io_buffer_size_ * 2) {
            want = io_buffer_size_ * 2;
        }
        // n_device_allocs is shared with the gallocr growth in compute_batch, so
        // it cannot say how many of these were io-buffer allocations. Time and
        // count them explicitly: ns_prep_grow is a FIXED ~1.4 s total regardless
        // of request count, so it is a few very expensive calls, not per-request.
        const auto grow_t0 = std::chrono::steady_clock::now();
        buffer_ptr buffer(ggml_backend_alloc_buffer(backend_.get(), want));
        fprintf(stderr, "wp io-buffer grow #%llu: %zu -> %zu bytes, alloc took %.1f ms\n",
                (unsigned long long) ++io_grow_count_, io_buffer_size_, want,
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - grow_t0).count() / 1000.0);
        if (!buffer) {
            throw std::runtime_error("failed to allocate persistent expert IO buffer");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        io_buffer_ = std::move(buffer);
        io_buffer_size_ = want;
        ++request_stats.n_device_allocs;
    }

    ggml_tensor * make_io_tensor(
            ggml_context * ctx, uint32_t n_tokens, size_t offset) const {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, n_tokens);
        attach_weight(
            tensor, io_buffer_.get(), ggml_backend_buffer_get_base(io_buffer_.get()), offset);
        return tensor;
    }

    void prepare_io(
            const std::vector<float> & activation,
            uint32_t n_tokens,
            RequestStats & request_stats) {
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert IO metadata");
        }
        // Sub-timers: prepare_io measured 1.53 ms/req on the RX 480 vs 0.07 on
        // the 1070, and the io buffer is CONFIRMED host-visible (memcpy writes)
        // under the size-split policy -- so the cost is not the upload. Split
        // the function to find what it actually is.
        auto sub = std::chrono::steady_clock::now();
        auto sublap = [&request_stats, &sub](uint64_t & dst) {
            const auto now = std::chrono::steady_clock::now();
            dst += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(now - sub).count();
            sub = now;
        };
        ggml_tensor * input = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, n_tokens);
        const ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(backend_.get());
        const size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        io_result_offset_ = GGML_PAD(input_size, alignment);
        const size_t result_size = ggml_backend_buft_get_alloc_size(buft, input);
        sublap(request_stats.ns_prep_setup);
        grow_io_buffer(io_result_offset_ + result_size, request_stats);
        sublap(request_stats.ns_prep_grow);
        attach_weight(
            input, io_buffer_.get(), ggml_backend_buffer_get_base(io_buffer_.get()), 0);
        sublap(request_stats.ns_prep_attach);
        ggml_backend_tensor_set(
            input, activation.data(), 0, activation.size() * sizeof(float));
        sublap(request_stats.ns_prep_set);
    }

    // all_experts=true builds ONE graph over every assignment in index order,
    // ignoring residency. See the determinism note on handle_request: splitting
    // by residency makes the floating-point ASSOCIATION depend on I/O timing.
    void compute_batch(
            const pipe_expert_dispatch_req & request,
            const std::vector<const ExpertPage *> & pages,
            const ExpertSlotPool::Batch & batch,
            bool hits,
            bool add_previous,
            RequestStats & request_stats,
            bool all_experts = false,
            // force_dense: ignore the gather path for this call only. Used by
            // WP_SELFCHECK to compute the SAME request both ways and diff them.
            bool force_dense = false,
            // Half-open ASSIGNMENT-INDEX range this call computes. Used by the
            // read/compute overlap to run a leading chunk of experts while the
            // tail is still being read. entries_ is indexed by assignment index
            // (is_resident(i) reads entries_.at(i)), and pageins carry the same
            // index, so this range selects a matching set of page-ins.
            // Default [0, SIZE_MAX) = every expert, i.e. unchanged behaviour.
            size_t sel_begin = 0,
            size_t sel_end = std::numeric_limits<size_t>::max()) {
        const auto selected = [&](size_t i) {
            return i >= sel_begin && i < sel_end &&
                   (all_experts || (batch.is_resident(i) == hits));
        };
        size_t n_selected = 0;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            n_selected += selected(i) ? 1 : 0;
        }
        if (n_selected == 0) {
            return;
        }

        // *** ROUTING-DENSITY GATHER/SCATTER (WP_EXPERT_GATHER=1, default OFF). ***
        // MEASURED 2026-08-04: at n_tokens=512 an assigned expert receives only
        // ~3.5% of the tokens, so the dense path below multiplies 96.5% of its
        // expert FLOPs by a ZERO router weight (28.7x waste in PREFILL; VERIFY is
        // 2.3x; decode is exactly 1.0x and thus unaffected). This path instead
        // gathers each expert's routed tokens with ggml_get_rows, runs the FFN at
        // the compacted width, and scatters back with ggml_get_rows_back (whose
        // semantics are precisely scatter-ADD: it sums every source row mapping to
        // the same destination row).
        // RISK, WHICH IS WHY THIS IS A FLAG AND NOT A REWRITE: get_rows_back is the
        // gradient of a gather and its kernel is O(ncols * nrows_dst * nrows_grad) --
        // it rescans every source row for each destination row. That is ~41G scan
        // ops per prefill run here, which could plausibly eat the ~13s of expert
        // compute it saves. MEASURE BOTH ARMS; do not assume.
        // *** CONFIG OF RECORD 2026-08-04: DEFAULT ON. *** Measured +25.8% prefill on
        // its own at n_ubatch=512, and SUPER-ADDITIVE with n_ubatch: the pair
        // (ub1024 + gather) is +67.3% against +45.3% predicted multiplicatively,
        // because a wider ubatch manufactures more zero-weight work for gather to
        // strip.
        // Set WP_EXPERT_GATHER=0 to fall back to the dense masked path.
        //
        // *** CORRECTION 2026-08-04 EVENING: "decode is unaffected" WAS WRONG. ***
        // That claim (previously on this line) rested on decode TOK/S, which sits
        // inside this harness's ~11.5% same-config noise floor and so could not have
        // resolved the effect. The spine's per-token decode dispatch wait CAN: eight
        // paired dense-vs-gather arms, alternating within one sweep, put gather at
        // +100 ms/token on that number, 8 for 8 (dense 177-253 ms -> gather 290-366 ms).
        // MECHANISM, and it is not subtle: decode routing density is EXACTLY 100%
        // (measured pre-gather; at n_tokens=1 an assigned expert has that token routed
        // by definition). So at decode ggml_get_rows compacts NOTHING and
        // ggml_get_rows_back scatters NOTHING back -- gather is pure added graph nodes
        // per expert per layer buying zero saved FLOPs. It is a prefill optimisation
        // that was billed to decode.
        // Hence WP_EXPERT_GATHER_MIN_TOKENS: gather only once a request is wide enough
        // for the compaction to pay for its own nodes. Default 2 = bypass at decode.
        // Set it to 1 to restore the always-gather behaviour (the A/B control).
        static const bool s_gather = [] {
            const char * e = std::getenv("WP_EXPERT_GATHER");
            return e == nullptr || e[0] != '0';   // default ON
        }();
        // *** DEFAULT 1 = GATHER ALWAYS, i.e. EXACTLY THE MEASURED CONFIG OF RECORD. ***
        // This briefly defaulted to 2 on 2026-08-04, which silently put the untested
        // decode bypass into every run -- including the arm I was calling the control.
        // A change ships as a DEFAULT only after it is measured; until then it is an
        // opt-in flag. Set 2 to test the bypass (#25): decode routing density is exactly
        // 100%, so at n_tokens==1 gather compacts nothing and only adds graph nodes,
        // measured at +100 ms/token across 8 of 8 paired arms. Still UNMEASURED as a
        // fix, because its A/B was confounded by WP_EXPERT_OVERLAP (see #23) and killed.
        static const int s_gather_min_tokens = [] {
            const char * e = std::getenv("WP_EXPERT_GATHER_MIN_TOKENS");
            const int v = e ? atoi(e) : 1;        // default 1 -> gather always (config of record)
            return v < 1 ? 1 : v;
        }();
        // PER-REQUEST, not static: prefill and decode requests interleave in one
        // worker, so this must be decided per request and never cached.
        const bool use_gather = s_gather && !force_dense &&
                                (int64_t) request.n_tokens >= (int64_t) s_gather_min_tokens;
        const auto build_started = std::chrono::steady_clock::now();
        // gather adds per expert: idx, sub_input, scattered (+ the add) -- budget
        // generously, ggml_init only reserves metadata.
        // +2 tensors / +2 nodes per expert for the SwiGLU clamp pair (up, gate)
        // added 2026-08-05. These are budgeted unconditionally even when the
        // clamp is off: under-budgeting the graph is a hard allocation failure,
        // and ggml_init only reserves metadata, so the slack is free.
        const size_t tensor_count = (s_gather ? 20 : 14) * n_selected + 8;
        const size_t graph_nodes  = (s_gather ? 12 :  8) * n_selected + 3;
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                              ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate batched expert graph metadata");
        }

        ggml_tensor * input = make_io_tensor(ctx.get(), request.n_tokens, 0);
        ggml_set_input(input);
        ggml_tensor * result = make_io_tensor(
            ctx.get(), request.n_tokens, io_result_offset_);
        // *** SEED THE FOLD, DO NOT ADD AT THE END. ***
        // The accumulator below is a LEFT-FOLD in assignment-index order:
        //     sum = ((((e0 + e1) + e2) + ...))
        // A chunked caller (WP_EXPERT_COMPUTE_CHUNKS) must continue that exact
        // fold, so the previous chunks' running total has to be the SEED. Adding
        // it at the end instead computes
        //     (e8 + ... + e15) + (e0 + ... + e7)
        // which is the same experts in the same order but a DIFFERENT
        // ASSOCIATION, and FP addition is not associative. Measured 2026-08-05:
        // that re-association alone moved draft acceptance 0.84286 -> 0.77966,
        // because the router's top-k is discontinuous and amplifies a last-bit
        // delta into a different token. Seeding makes chunked output bit-identical
        // to the unchunked path.
        ggml_tensor * sum = add_previous ? result : nullptr;
        std::vector<std::pair<ggml_tensor *, const pipe_expert_assignment *>> routing_weights;
        routing_weights.reserve(n_selected);
        // Parallel to routing_weights: the gathered token indices per expert, kept
        // alive until after ggml_gallocr_alloc_graph so they can be uploaded.
        std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> gather_idx;
        gather_idx.reserve(n_selected);

        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (!selected(i)) {
                continue;
            }
            const ExpertPage & page = *pages[i];
            const ExpertSlotPool::Loaded loaded = batch.loaded(i);
            const auto & specs = catalog_.descriptor.layers.at(page.layer);
            const auto make_weight = [&](const std::string & role) {
                const RoleSpec & spec = specs.at(role);
                ggml_tensor * tensor =
                    ggml_new_tensor_2d(ctx.get(), spec.type, spec.ne0, spec.ne1);
                attach_weight(
                    tensor, loaded.buffer, loaded.base, page.roles.at(role).offset);
                return tensor;
            };

            // GATHER: restrict this expert to the tokens actually routed to it.
            ggml_tensor * ffn_in = input;
            ggml_tensor * idx_t  = nullptr;
            std::vector<int32_t> idx;
            if (use_gather) {
                const auto & wv = request.assignments[i].weights;
                idx.reserve(wv.size());
                for (size_t t = 0; t < wv.size(); ++t) {
                    if (wv[t] != 0.0f) { idx.push_back((int32_t) t); }
                }
                if (idx.empty()) {
                    // No token routed to this expert in this ubatch. Skipping it
                    // outright is WRONG: if EVERY selected expert is empty, `sum`
                    // stays nullptr and ggml_cpy(sum, result) segfaults in
                    // ggml_nelements -- which is exactly how the first build of
                    // this path killed the RX 480 worker mid-prefill.
                    // Instead keep ONE row whose router weight is zero. Its
                    // contribution is exactly 0.0, identical to what the dense
                    // path computed, and it costs a single N=1 FFN.
                    idx.push_back(0);
                }
                idx_t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t) idx.size());
                ggml_set_input(idx_t);
                ffn_in = ggml_get_rows(ctx.get(), input, idx_t);
            }
            ggml_tensor * gate = ggml_mul_mat(ctx.get(), make_weight("gate"), ffn_in);
            ggml_tensor * up   = ggml_mul_mat(ctx.get(), make_weight("up"), ffn_in);
            // *** SwiGLU CLAMP. ADDED 2026-08-05 -- ITS ABSENCE WAS A CORRECTNESS BUG. ***
            // Mirrors the LLM_ARCH_DEEPSEEK4 branch of build_moe_ffn() in
            // src/llama-graph.cpp EXACTLY: up is clamped symmetrically, the GATE is
            // clamped ABOVE ONLY (-INFINITY lower bound) and fed to swiglu_split,
            // which applies silu to it. Do not "simplify" this to a symmetric gate
            // clamp or to silu-then-clamp -- those are the OTHER architectures'
            // branch and give different numbers.
            // The spine cannot do this for us: build_moe_ffn() returns at the
            // `expert_dispatch != nullptr` branch before ever reaching the clamp.
            const float swiglu_limit = request.swiglu_clamp;
            if (swiglu_limit > 1e-6f) {
                up   = ggml_clamp(ctx.get(), up,   -swiglu_limit, swiglu_limit);
                gate = ggml_clamp(ctx.get(), gate, -INFINITY,     swiglu_limit);
            }
            ggml_tensor * hidden = ggml_swiglu_split(ctx.get(), gate, up);
            ggml_tensor * output = ggml_mul_mat(ctx.get(), make_weight("down"), hidden);
            // SHAPE MATTERS: [1, n_tokens], NOT [n_tokens]. output is
            // [n_embd, n_tokens]; ggml_mul broadcasts src1 into src0 via
            // ggml_can_repeat, which only checks ne[i] % src1->ne[i] == 0. A
            // 1-D [n_tokens] tensor PASSES that check (6144 % 8 == 0) and then
            // scales along the EMBEDDING axis instead of the token axis --
            // silently wrong, no assert. It is correct only at n_tokens == 1,
            // where both readings coincide, so decode looked fine while PREFILL
            // was corrupted and poisoned the KV cache.
            const int64_t n_rows = use_gather ? (int64_t) idx.size() : request.n_tokens;
            ggml_tensor * weights =
                ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_rows);
            ggml_set_input(weights);
            ggml_tensor * weighted = ggml_mul(ctx.get(), output, weights);
            if (use_gather) {
                // SCATTER-ADD back to full token width. get_rows_back sums every
                // source row that maps to the same destination row, and rows this
                // expert did not touch stay zero -- which is exactly the dense
                // path's zero-weight contribution, so the sum below is unchanged.
                weighted = ggml_get_rows_back(ctx.get(), weighted, idx_t, input);
            }
            sum = sum ? ggml_add(ctx.get(), sum, weighted) : weighted;
            routing_weights.emplace_back(weights, &request.assignments[i]);
            gather_idx.emplace_back(idx_t, std::move(idx));
        }
        // (add_previous is folded in as the SEED above, not appended here.)
        if (sum == nullptr) {
            // Defensive: nothing contributed. Cannot happen now that empty experts
            // keep a zero-weight row, but a null here segfaults inside ggml_cpy
            // with no diagnostic, so refuse loudly rather than crash the worker.
            throw std::runtime_error("expert compute produced no contribution");
        }
        ggml_tensor * copy = ggml_cpy(ctx.get(), sum, result);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(graph, copy);

        const size_t old_compute_size =
            ggml_gallocr_get_buffer_size(compute_galloc_.get(), 0);
        if (!ggml_gallocr_alloc_graph(compute_galloc_.get(), graph)) {
            throw std::runtime_error("failed to allocate batched expert compute graph");
        }
        if (ggml_gallocr_get_buffer_size(compute_galloc_.get(), 0) > old_compute_size) {
            ++request_stats.n_device_allocs;
        }
        request_stats.ns_graph_build +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - build_started).count();

        for (size_t k = 0; k < routing_weights.size(); ++k) {
            const auto & item = routing_weights[k];
            const auto & wv = item.second->weights;
            uint64_t nz = 0;
            for (float f : wv) { nz += (f != 0.0f); }
            request_stats.n_weight_nonzero += nz;
            // n_weight_total is what was actually COMPUTED, so it must follow the
            // path taken: the compacted row count when gathering, the full ubatch
            // when not. Otherwise the density counter would report the dense
            // figure even after the fix and hide whether it worked.
            if (use_gather) {
                const auto & idx = gather_idx[k].second;
                std::vector<float> compact;
                compact.reserve(idx.size());
                for (int32_t t : idx) { compact.push_back(wv[(size_t) t]); }
                request_stats.n_weight_total += compact.size();
                ggml_backend_tensor_set(
                    item.first, compact.data(), 0, compact.size() * sizeof(float));
                ggml_backend_tensor_set(
                    gather_idx[k].first, idx.data(), 0, idx.size() * sizeof(int32_t));
            } else {
                request_stats.n_weight_total += wv.size();
                ggml_backend_tensor_set(
                    item.first, wv.data(), 0, wv.size() * sizeof(float));
            }
        }
        const auto submit_started = std::chrono::steady_clock::now();
        const enum ggml_status status =
            ggml_backend_graph_compute(backend_.get(), graph);
        request_stats.ns_submit +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - submit_started).count();
        ++request_stats.n_graph_submits;
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("batched expert backend graph compute failed");
        }
    }

    void read_result(std::vector<float> & result, RequestStats & request_stats) {
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert result metadata");
        }
        const uint32_t n_tokens = (uint32_t) (
            result.size() / (size_t) catalog_.descriptor.hparams.n_embd);
        ggml_tensor * output = make_io_tensor(ctx.get(), n_tokens, io_result_offset_);
        const auto readback_started = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(
            output, result.data(), 0, result.size() * sizeof(float));
        request_stats.ns_readback +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - readback_started).count();
    }

    Catalog        catalog_;
    backend_ptr    backend_;
    ResidentExpertPool resident_;
    ExpertSlotPool pool_;
    galloc_ptr     compute_galloc_;
    buffer_ptr     io_buffer_;
    size_t         io_buffer_size_ = 0;
    uint64_t            io_grow_count_ = 0;
    size_t         io_result_offset_ = 0;
    WorkerStats    stats_;
    int            slots_ = 0;
};

bool validate_client_hello(
        const pipe_expert_hello & client, const pipe_expert_hello & worker,
        std::string & error) {
    if (client.role != PIPE_EXPERT_ROLE_CLIENT) {
        error = "peer did not identify as an expert client";
    } else if (client.hidden_type != worker.hidden_type ||
               client.n_embd != worker.n_embd ||
               client.n_ff_exp != worker.n_ff_exp ||
               client.n_expert != worker.n_expert ||
               client.n_expert_used != worker.n_expert_used) {
        error = "expert HELLO hparams mismatch";
    } else if (client.model_identity != worker.model_identity) {
        error = "expert HELLO model identity mismatch";
    } else if (client.shard_identity != worker.shard_identity) {
        error = "expert HELLO shard identity mismatch";
    }
    return error.empty();
}

bool send_hello_ack(
        pipe_socket_t & socket, bool accepted, const std::string & reason) {
    const std::vector<uint8_t> payload =
        pipe_encode_expert_hello_ack({ accepted, reason });
    return pipe_send_frame(
        socket, PIPE_EXPERT_HELLO_ACK, 0, payload.data(), payload.size());
}

int serve_connection(pipe_socket_t & socket, Worker & worker) {
    const pipe_expert_hello mine = worker.hello();
    const std::vector<uint8_t> hello_payload = pipe_encode_expert_hello(mine);
    if (!pipe_send_frame(
            socket, PIPE_HELLO, 0, hello_payload.data(), hello_payload.size())) {
        return 1;
    }

    pipe_frame_type type;
    uint64_t seq_id = 0;
    std::vector<uint8_t> payload;
    if (!pipe_recv_frame(socket, type, seq_id, payload)) {
        return 1;
    }
    if (type != PIPE_HELLO || seq_id != 0) {
        send_hello_ack(socket, false, "expected expert HELLO");
        return 1;
    }
    try {
        const pipe_expert_hello client =
            pipe_decode_expert_hello(payload.data(), payload.size());
        std::string error;
        if (!validate_client_hello(client, mine, error)) {
            send_hello_ack(socket, false, error);
            return 1;
        }
    } catch (const pipe_protocol_error & error) {
        send_hello_ack(socket, false, error.what());
        return 1;
    }
    if (!send_hello_ack(socket, true, "")) {
        return 1;
    }

    // WP_REQ_LOG=path -- one line per dispatch request. Columns:
    //   layer n_tokens n_exp n_resident n_pagein bytes_read ns_wall ns_lookup ns_prep
    //   ns_hits ns_wait ns_pagein_compute ns_result ns_read ns_h2d ns_submit
    //   ns_readback ns_encode ns_send
    // Segment into tokens by watching request.layer wrap back to its minimum.
    //
    // n_tokens (added 2026-08-03) IS THE PREFILL/DECODE LABEL, and it is the whole
    // reason this log can now answer "where does prefill go". n_tokens > 1 is a
    // prefill ubatch, n_tokens == 1 is a decode step -- the dispatcher already
    // relies on exactly this test to disable deferral (pipe-expert-dispatcher.cpp
    // :969), so the semantics are not invented here, only recorded.
    //
    // Until now EVERY phase timer in this file was un-attributable: prefill and
    // decode requests interleave in one stream and the only way to tell them apart
    // was to guess from n_exp. That made the 18 columns below useless for the one
    // question that matters (prefill is 33.9 ms/token and decode is 6.5 tok/s --
    // WHICH of these phases owns which?). One integer fixes it.
    FILE * const req_log = [] {
        const char * p = std::getenv("WP_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    // WP_REF_LOG=path -- the full REFERENCE stream: "<layer> <expert> <expert> ..."
    // one line per request, every expert asked for whether it was resident or paged in.
    // WP_PAGEIN_LOG cannot substitute: which pages pagein is a function of the
    // replacement policy, so a pagein trace can only ever describe the policy that
    // produced it. The reference stream is policy-independent, which makes LRU /
    // LFU / ARC / Belady all simulatable OFFLINE from a single run, at zero GPU
    // cost per candidate -- and Belady gives the true ceiling, so we learn
    // whether a policy change is worth implementing BEFORE implementing one.
    FILE * const ref_log = [] {
        const char * p = std::getenv("WP_REF_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    // Keepalive pump: while no request is pending, occupy the GPU rather than
    // let it idle (see Worker::keepalive_tick for the measurements). Gated on
    // KEEPALIVE_IDLE_MS of recent activity so a genuinely idle worker still lets
    // the card reach runtime suspend (D3), which is where its idle power saving
    // comes from -- an unconditional pump would keep it awake forever.
    const int keepalive_fd = socket.poll_fd();
    const auto KEEPALIVE_IDLE_MS = std::chrono::milliseconds(2000);
    auto last_request_at = std::chrono::steady_clock::now();
    auto await_request = [&]() {
        // The speculative path needs this loop even when the keepalive pump is off:
        // they share the idle window but are independent features, and gating
        // speculation on WP_KEEPALIVE_US would silently couple two experiments.
        if ((!worker.keepalive_enabled() && !worker.has_spec_work()) || keepalive_fd < 0) {
            return;
        }
        for (;;) {
            // Re-checked every iteration, not just on entry. Without this, a
            // worker with the pump OFF that finishes its spec queue would sit in
            // a zero-timeout ppoll spinning a core: nothing left to read, and a
            // keepalive_tick that is a no-op when the pump is disabled.
            if (!worker.keepalive_enabled() && !worker.has_spec_work()) {
                return;
            }
            struct pollfd pfd { keepalive_fd, POLLIN, 0 };
            // ppoll, NOT poll: poll's timeout is in whole milliseconds, so a
            // 200 us period silently became 1 ms and left the GPU idle for most
            // of every interval. That cost us half the available win -- measured
            // 0.319 ms/expert with the 1 ms pump against 0.163 in an isolated
            // bench that occupied the device continuously.
            // Zero timeout ONLY when there is something to submit -- then the
            // pump period is dead time before issuing a read. With a read already
            // in flight we are waiting on the reader thread, so poll on the
            // ordinary period and harvest when it lands. Spinning there would
            // burn a core for the whole 3-5 ms read and buy nothing.
            const long ns = worker.has_spec_submit_work() ? 0L
                                                          : (long) worker.keepalive_us() * 1000L;
            struct timespec ts { ns / 1000000000L, ns % 1000000000L };
            const int r = ::ppoll(&pfd, 1, &ts, nullptr);
            if (r != 0) {
                // A REAL REQUEST BEATS A PREFETCH, ALWAYS. Whatever is still
                // queued stays queued -- if it is still worth reading, the next
                // idle window will take it, and if the layer has moved on the
                // idle timeout below discards it.
                return;   // data ready, or an error recv_data will surface
            }
            if (std::chrono::steady_clock::now() - last_request_at > KEEPALIVE_IDLE_MS) {
                // Idle long enough that any hinted layer is far behind us.
                // Reading for a layer already computed is a read with no
                // possible upside, so drop it rather than carry it forward.
                worker.drop_spec_work();
                return;   // let the card sleep
            }
            // Warm before pumping: the pump exists to stop the card dropping
            // clocks while idle, and a page-in plus its H2D is not idle.
            if (worker.spec_pagein_step()) {
                continue;
            }
            worker.keepalive_tick();
        }
    };

    // Comma operator so the pump runs before EVERY recv, including after the
    // PING branch's `continue` -- appending it to the loop body would skip that.
    while ((await_request(), pipe_recv_frame(socket, type, seq_id, payload))) {
        last_request_at = std::chrono::steady_clock::now();
        if (type == PIPE_PING) {
            if (!pipe_send_frame(socket, PIPE_PONG, seq_id, nullptr, 0)) {
                return 1;
            }
            continue;
        }
        // Prefetch hint: accept, validate, count, DO NOTHING ELSE (yet).
        //
        // This half exists on its own so the wire can be proven before the pool
        // is touched. With only this, a run is observably identical to the
        // config of record -- no extra read, no eviction, no slot pinned -- and
        // the hint counters still say exactly what the spine offered and when.
        // If those counters come out wrong, the fault is on the spine side and
        // is found without a single changed page-in.
        //
        // A hint gets NO response frame, so it must not fall through to the
        // dispatch path's reply, and a malformed one must not kill the session:
        // it is advisory, and the request stream is untouched by dropping it.
        if (type == PIPE_EXPERT_PREFETCH_HINT) {
            try {
                const pipe_expert_prefetch_hint hint =
                    pipe_decode_expert_prefetch_hint(payload.data(), payload.size());
                worker.note_prefetch_hint(hint);
            } catch (const pipe_protocol_error & error) {
                worker.note_prefetch_hint_bad();
                std::fprintf(stderr, "wp-expert-worker: ignoring malformed prefetch hint: %s\n",
                             error.what());
            }
            continue;
        }
        if (type != PIPE_EXPERT_DISPATCH_REQ) {
            pipe_send_error(socket, seq_id, PIPE_ERR_BAD_FRAME, "expected expert dispatch request");
            return 1;
        }
        try {
            const pipe_expert_dispatch_req request =
                pipe_decode_expert_dispatch_req(
                    payload.data(), payload.size(), mine.n_embd);
            RequestStats request_stats;
            // WP_REQ_LOG: per-request phase dump. The cumulative WorkerStats
            // totals cannot separate "every request is uniformly slow" from
            // "most are fast and a few are enormous", and on the RX 480 the
            // per-token cost distribution is p25 140 / median 170 / max 1653 ms
            // -- so the shape is the whole question. request.layer lets the
            // reader segment this stream into tokens by layer wrap WITHOUT a
            // cross-machine clock join (the spine runs on the other box); that
            // is what made WP_PAGEIN_LOG unusable for the same purpose.
            if (ref_log != nullptr) {
                fprintf(ref_log, "%d", request.layer);
                for (const pipe_expert_assignment & a : request.assignments) {
                    fprintf(ref_log, " %d", a.expert_id);
                }
                fputc('\n', ref_log);
                fflush(ref_log);
            }
            // Ground truth into the ordered stream. Emitted BEFORE dispatch, so
            // an R always precedes the D lines that dispatch provokes -- which is
            // what makes "was this page already speculated for THIS reference"
            // answerable by a single forward pass over the file.
            worker.log_reference(request.layer, request.assignments);
            const std::chrono::steady_clock::time_point req_started =
                req_log != nullptr ? std::chrono::steady_clock::now() :
                                     std::chrono::steady_clock::time_point();
            const pipe_expert_partial response = worker.dispatch(
                request, request_stats);
            const bool measure = worker.stats_enabled();
            const std::chrono::steady_clock::time_point send_started =
                measure ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
            const std::vector<uint8_t> encoded =
                pipe_encode_expert_partial(response);
            if (!pipe_send_frame(
                    socket, PIPE_EXPERT_PARTIAL, seq_id,
                    encoded.data(), encoded.size())) {
                return 1;
            }
            if (measure) {
                request_stats.ns_send =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - send_started).count();
                worker.record_stats(
                    request_stats, request.assignments.size());
            }
            // Every phase field above is populated only when WP_WORKER_STATS=1,
            // so this log is meaningless without it -- hence `measure &&`.
            // fflush per line: the harness SIGKILLs workers at teardown, and an
            // unflushed stdio buffer produced 0-byte files the first time.
            if (measure && req_log != nullptr) {
                const uint64_t ns_wall =
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - req_started).count();
                const RequestStats & s = request_stats;
                fprintf(req_log,
                        "%d %u %zu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                        "%llu %llu %llu %llu %llu %llu %llu %llu\n",
                        request.layer, request.n_tokens, request.assignments.size(),
                        (unsigned long long) s.n_resident,
                        (unsigned long long) s.n_pagein,
                        (unsigned long long) s.bytes_read,
                        (unsigned long long) ns_wall,
                        (unsigned long long) s.ns_lookup,
                        (unsigned long long) s.ns_prep,
                        (unsigned long long) s.ns_hits,
                        (unsigned long long) s.ns_wait,
                        (unsigned long long) s.ns_pagein_compute,
                        (unsigned long long) s.ns_result,
                        (unsigned long long) s.ns_read,
                        (unsigned long long) s.ns_h2d,
                        (unsigned long long) s.ns_submit,
                        (unsigned long long) s.ns_readback,
                        (unsigned long long) s.ns_encode,
                        (unsigned long long) s.ns_send,
                        // ROUTING DENSITY, per request, so it can be split by
                        // n_tokens (col 2). Appended at the END: the format is
                        // POSITIONAL and older parsers index from the left, so
                        // adding columns here keeps every existing parser working.
                        (unsigned long long) s.n_weight_nonzero,
                        (unsigned long long) s.n_weight_total);
                fflush(req_log);
            }
        } catch (const pipe_protocol_error & error) {
            if (!pipe_send_error(socket, seq_id, error.code, error.what())) {
                return 1;
            }
        } catch (const std::exception & error) {
            pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
            return 1;
        }
    }
    // Only on a clean close, and the harness SIGKILLs workers at teardown, so
    // this line is best effort and arm 1 never saw it. WP_HINT_LOG is the
    // durable record -- NOT WP_REQ_LOG, which carries no hint fields and was
    // wrongly named here before.
    worker.report_prefetch_hints();
    return 0;
}

} // namespace

bool self_bench_stats(uint64_t & n, uint64_t & min_us, uint64_t & mean_us) {
    if (g_probe.n == 0) return false;
    n = g_probe.n;
    min_us = g_probe.min_ns / 1000;
    mean_us = g_probe.total_ns / g_probe.n / 1000;
    return true;
}

void self_bench_tick(ggml_backend_t backend) {
    if (!g_probe.ready) return;
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, g_probe.graph);
    const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (ns < g_probe.min_ns) g_probe.min_ns = ns;
    g_probe.total_ns += ns;
    ++g_probe.n;
}

ResourcePlan inspect_resources(const Options & options) {
    if (options.slots <= 0 || options.device.empty()) {
        throw std::invalid_argument("invalid expert worker resource options");
    }
    const fs::path manifest   = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        options.device,
        options.slots,
        options.host_budget_bytes,
        options.host_victim_bytes,
        options.test_hooks,
        options.resident_expert_blocks, options.expert_reserve_blocks,
        options.expert_reserve_bytes);
    return worker.resources();
}

int run(const Options & options) {
    if (options.slots <= 0 || options.listen_host.empty() ||
        options.listen_port <= 0 || options.listen_port > 65535 ||
        options.device.empty()) {
        throw std::invalid_argument("invalid expert worker options");
    }
    const fs::path manifest = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        options.device,
        options.slots,
        options.host_budget_bytes,
        options.host_victim_bytes,
        options.test_hooks,
        options.resident_expert_blocks, options.expert_reserve_blocks,
        options.expert_reserve_bytes);
    const pipe_expert_hello advertised = worker.hello();
    const ResourcePlan & resources = worker.resources();

    // *** WP_WARMUP: pay the cold-start stalls BEFORE serving. Default OFF. ***
    //
    // MEASURED 2026-08-05 (fp3-r1, both workers). ALL of the slow submits and ALL
    // device allocations happen in the first ~140 requests, then stop dead:
    //     requests:      1   74   82   91   99  107  115  140  362  646 5794
    //     >=8ms submits: 1    2   10   19   27   35   43   44   44   44   44
    //     device allocs: 1    5    7    8    9   11   11   11   11   11   11
    // ns_submit is 1.833 s at request 140 and 4.677 s at request 5794 -- 39% of all
    // submit time is spent in 2.4% of the requests. The RX 480's first n_tokens=2
    // verify block cost 6044 ms, of which 5975 ms was that one worker.
    // This is NOT a Vulkan pathology: the CUDA worker shows 14 device allocations
    // and an 83.8 ms worst submit against the RX 480's 11 and 74.7 ms.
    //
    // run_self_bench does NOT cover this: it builds a ONE-EXPERT, ONE-TOKEN graph,
    // so it never touches verify widths, the prefill width, multi-expert graphs,
    // the gather path, or the SwiGLU clamp.
    //
    // Spec: WP_WARMUP="<tokens>x<experts>[,<tokens>x<experts>...]", e.g.
    //   WP_WARMUP="1x8,2x8,5x8,659x35"
    // Each entry replays a synthetic request through the REAL dispatch path so the
    // graph build, gallocr allocation, backend pipeline specialisation and clamp
    // ops are all compiled for that shape before the first real request arrives.
    // Results are discarded. Kept default-OFF so it can be A/B'd against the
    // measured baseline rather than silently changing the config of record.
    if (const char * warm = std::getenv("WP_WARMUP")) {
        if (warm[0] != '\0' && warm[0] != '0') {
            // ALL LAYERS, not just the first (2026-08-05, second iteration).
            // Warming one layer removed all 11 device allocations but left the
            // >=8 ms submit count at 43-44 -- and the worker serves 43 layers.
            // One stall per layer says the cost is per-LAYER tensor-set/descriptor
            // setup, not per-shape, so the shape list is swept on one layer while
            // every layer is touched at the cheapest shape.
            const int32_t n_embd = advertised.n_embd;
            const int32_t layer  = advertised.layers.empty() ? -1 : advertised.layers.front();
            const int32_t e_first = advertised.expert_first;
            const int32_t e_last  = advertised.expert_last;
            const auto t0 = std::chrono::steady_clock::now();
            size_t done = 0;
            std::string spec(warm);
            size_t pos = 0;
            while (pos <= spec.size() && layer >= 0 && e_first >= 0) {
                const size_t comma = spec.find(',', pos);
                const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
                const size_t x = item.find('x');
                if (x == std::string::npos) { continue; }
                const long toks = strtol(item.substr(0, x).c_str(), nullptr, 10);
                const long nexp = strtol(item.substr(x + 1).c_str(), nullptr, 10);
                if (toks <= 0 || nexp <= 0) { continue; }
                pipe_expert_dispatch_req req;
                req.layer        = layer;
                req.n_tokens     = (uint32_t) toks;
                req.swiglu_clamp = 10.0f;   // non-zero so the clamp ops get built
                const long avail = (long) (e_last - e_first + 1);
                for (long i = 0; i < nexp && i < avail; ++i) {
                    pipe_expert_assignment a;
                    a.expert_id = (int32_t) (e_first + i);
                    a.weights.assign((size_t) toks, 0.5f);
                    req.assignments.push_back(std::move(a));
                }
                req.activations.assign((size_t) toks * (size_t) n_embd, 0.01f);
                try {
                    RequestStats throwaway;
                    (void) worker.dispatch(req, throwaway);
                    ++done;
                } catch (const std::exception & e) {
                    // Warmup must never prevent serving. A shape we cannot warm is
                    // simply a shape that pays its stall on the first real request,
                    // which is exactly the status quo.
                    std::cout << "wp warmup: skipped " << item << ": " << e.what() << std::endl;
                }
            }
            // Second pass: touch EVERY served layer at the cheapest shape.
            size_t layers_done = 0;
            for (int32_t ly : advertised.layers) {
                if (ly == layer || e_first < 0) { continue; }   // first layer already covered
                pipe_expert_dispatch_req req;
                req.layer        = ly;
                req.n_tokens     = 1;
                req.swiglu_clamp = 10.0f;
                pipe_expert_assignment a;
                a.expert_id = e_first;
                a.weights.assign(1, 0.5f);
                req.assignments.push_back(std::move(a));
                req.activations.assign((size_t) n_embd, 0.01f);
                try {
                    RequestStats throwaway;
                    (void) worker.dispatch(req, throwaway);
                    ++layers_done;
                } catch (const std::exception &) {
                    // non-fatal, same reasoning as above
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "wp warmup: " << done << " shape(s) from \"" << warm
                      << "\" + " << layers_done << " extra layer(s) in "
                      << ms << " ms (results discarded)" << std::endl;
        }
    }

    pipe_socket_ptr server =
        pipe_socket_t::create_server(options.listen_host.c_str(), options.listen_port);
    if (!server) {
        throw std::runtime_error(
            "failed to listen on " + options.listen_host + ":" +
            std::to_string(options.listen_port));
    }
    std::cout << "expert worker listening on " << options.listen_host << ":"
              << options.listen_port << " device=" << options.device
              << " experts=" << advertised.expert_first << "-"
              << advertised.expert_last << " layers=" << advertised.layers.size()
              << " slots=" << advertised.n_slots
              << " requested_slots=" << resources.requested_slots
              << " pinned_pages=" << worker.pinned_pages()
              << " pinned_bytes=" << resources.pinned_bytes
              << " slot_budget_bytes=" << resources.slot_budget_bytes
              << " device_bytes=" << resources.device_bytes
              << " size_classes=" << (resources.size_classes ? 1 : 0)
              << " staging=" << resources.staging_buffers << "x"
              << resources.staging_buffer_bytes
              << " host_budget=" << resources.host_budget_bytes
              << " host_victim_budget=" << options.host_victim_bytes << '\n';
    for (const SlotClass & slot_class : resources.slot_classes) {
        std::cout << "expert slot class bytes=" << slot_class.size
                  << " slots=" << slot_class.slots
                  << " pin_floor=" << slot_class.pin_floor
                  << " pages=" << slot_class.pages << '\n';
    }

    int result = 0;
    do {
        pipe_socket_ptr client = server->accept();
        if (!client) {
            return 1;
        }
        result = serve_connection(*client, worker);
    } while (!options.once);
    return result;
}

} // namespace wp_expert_worker
