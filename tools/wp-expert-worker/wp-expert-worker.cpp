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
#include <thread>
#include <utility>
#include <vector>

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
    uint64_t n_hit      = 0;
    uint64_t n_miss     = 0;
    uint64_t n_host_hit = 0;
    uint64_t n_host_demote = 0;
    uint64_t bytes_read = 0;
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
    uint64_t ns_miss_compute = 0;  // compute_batch(misses) end to end
    uint64_t ns_result = 0;        // read_result
    uint64_t ns_encode = 0;        // fp32 -> fp16 of the reply
    uint64_t ns_h2d = 0;
    uint64_t bytes_h2d = 0;
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
        n_hit_ += request.n_hit;
        n_miss_ += request.n_miss;
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
        ns_miss_compute_ += request.ns_miss_compute;
        ns_result_ += request.ns_result;
        ns_encode_ += request.ns_encode;
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
                  << " n_hit=" << n_hit_
                  << " n_miss=" << n_miss_
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
                  << " ns_misscompute=" << ns_miss_compute_
                  << " ns_result=" << ns_result_
                  << " ns_encode=" << ns_encode_
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
    uint64_t          n_hit_      = 0;
    uint64_t          n_miss_     = 0;
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
    uint64_t          ns_miss_compute_ = 0;
    uint64_t          ns_result_ = 0;
    uint64_t          ns_encode_ = 0;
    uint64_t          ns_h2d_     = 0;
    uint64_t          bytes_h2d_  = 0;
    std::string       staging_kind_ = "unknown";
    uint64_t          n_requests_ = 0;
    uint64_t          n_experts_  = 0;
};

ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes) {
    if (pages.empty() || requested_slots <= 0) {
        throw std::invalid_argument("invalid expert resource plan dimensions");
    }

    std::map<uint64_t, int> histogram;
    std::map<uint64_t, std::map<int, int>> layer_counts;
    uint64_t max_page_size = 0;
    for (const ResourcePage & page : pages) {
        if (page.layer < 0 || page.size == 0 ||
            histogram[page.size] == std::numeric_limits<int>::max() ||
            layer_counts[page.size][page.layer] == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("invalid expert resource page");
        }
        ++histogram[page.size];
        ++layer_counts[page.size][page.layer];
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

    bool use_size_classes = floor_bytes <= result.device_budget_bytes;
    if (use_size_classes) {
        const long double average =
            weighted_bytes / (long double) total_pages;
        const long double remaining =
            (long double) (result.device_budget_bytes - floor_bytes);
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
        while (planned_bytes() > result.device_budget_bytes) {
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
            max_layer_pages =
                std::max(max_layer_pages, ++pages_by_layer[page.layer]);
        }
        if (requested_slots < max_layer_pages) {
            throw std::invalid_argument(
                "expert slot budget is smaller than the largest layer request");
        }
        classes.clear();
        classes.push_back({
            max_page_size, requested_slots, max_layer_pages, (int) pages.size()
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
        result.push_back({ item.second.layer, item.second.size });
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

class ExpertSlotPool {
private:
    struct Miss {
        size_t             entry_index = 0;
        size_t             slot_index  = 0;
        const ExpertPage * page        = nullptr;
        int                fd          = -1;
    };

    struct ReadResult {
        size_t                              miss_index = 0;
        std::unique_ptr<StagingPool::Lease> staging;
        std::exception_ptr                  error;
        std::chrono::steady_clock::time_point read_started;
        std::chrono::steady_clock::time_point read_finished;
        bool                                read_timed = false;
    };

    struct BatchState {
        std::vector<Miss>                       misses;
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
            uint64_t host_victim_bytes, TestHooks * test_hooks) :
        backend_(backend),
        resources_(std::move(resources)),
        staging_(resources_, backend),
        test_hooks_(test_hooks) {
        if (resources_.slot_count <= 0 ||
            resources_.staging_buffer_bytes == 0 ||
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
        for (const SlotClass & slot_class : resources_.slot_classes) {
            for (int i = 0; i < slot_class.slots; ++i) {
                slots_.push_back(make_slot(slot_class.size));
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
        }
    }

    ~ExpertSlotPool() {
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

        bool is_hit(size_t index) const {
            return entries_.at(index).hit;
        }

        const Loaded & loaded(size_t index) const {
            const Entry & entry = entries_.at(index);
            if (!entry.ready) {
                throw std::logic_error("expert batch entry is not ready");
            }
            return entry.loaded;
        }

        void complete();

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

        uint64_t n_hit() const {
            return n_hit_;
        }

        uint64_t n_miss() const {
            return n_miss_;
        }

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
        uint64_t                   n_hit_     = 0;
        uint64_t                   n_miss_    = 0;
        uint64_t                   bytes_read_ = 0;
        uint64_t                   n_host_hit_ = 0;
        uint64_t                   n_host_demote_ = 0;
        uint64_t                   ns_host_get_ = 0;
        uint64_t                   host_bytes_ = 0;
        uint64_t                   ns_h2d_    = 0;
        uint64_t                   bytes_h2d_ = 0;
    };

    Batch ensure_batch(
            const std::vector<const ExpertPage *> & pages,
            bool measure,
            std::chrono::steady_clock::time_point lookup_started) {
        Batch batch(this, pages.size());
        try {
            std::vector<size_t> misses;
            misses.reserve(pages.size());

            // Resolve and pin every hit before selecting a victim. A hit is
            // immediately usable while sibling misses are read.
            for (size_t i = 0; i < pages.size(); ++i) {
                const ExpertPage & page = *pages[i];
                const std::pair<int, int> key(page.layer, page.expert);
                size_t slot_index = slots_.size();
                for (size_t j = 0; j < slots_.size(); ++j) {
                    if (slots_[j].valid && slots_[j].key == key) {
                        slot_index = j;
                        break;
                    }
                }
                if (slot_index == slots_.size()) {
                    misses.push_back(i);
                    continue;
                }
                Slot & slot = slots_[slot_index];
                ++slot.pin_count;
                slot.tick = ++tick_;
                batch.entries_[i] = {
                    { slot.buffer.get(),
                      ggml_backend_buffer_get_base(slot.buffer.get()) },
                    slot_index,
                    true,
                    true,
                };
                ++batch.n_hit_;
            }

            // Reserve and pin all miss slots before starting any read. Later
            // allocations in this request cannot select an earlier miss.
            for (size_t entry_index : misses) {
                const ExpertPage & page = *pages[entry_index];
                const size_t slot_index = select_victim(page.size);
                if (slot_index == slots_.size()) {
                    throw std::runtime_error(
                        "no expert slot can hold requested page");
                }
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
                for (size_t entry_index : misses) {
                    const ExpertPage & page = *pages[entry_index];
                    if (host_tier_.borrow(
                            page.cache_id, &host_hits[entry_index].src,
                            (size_t) page.size, &host_hits[entry_index].borrow)) {
                        continue;
                    }
                }
            }

            auto release_host_hits = [&]() {
                for (size_t entry_index : misses) {
                    HostHit & host_hit = host_hits[entry_index];
                    if (host_hit.borrow != wp::HostTier::kInvalidBorrowHandle) {
                        host_tier_.release(
                            pages[entry_index]->cache_id, host_hit.borrow);
                        host_hit.borrow = wp::HostTier::kInvalidBorrowHandle;
                    }
                }
            };

            batch.state_ = std::make_shared<BatchState>();
            batch.state_->misses.reserve(misses.size());
            try {
                for (size_t entry_index : misses) {
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
                        batch.state_->misses.push_back({
                            entry_index, slot_index, &page, fd_for(page.blob)
                        });
                        ++batch.n_miss_;
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

            if (batch.state_->misses.empty()) {
                batch.completed_ = true;
                return batch;
            }

            batch.state_->measure = measure;

            const size_t worker_count = std::min<size_t>(
                batch.state_->misses.size(), (size_t) staging_.buffer_count());
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
        int                 pin_count = 0;
        bool                valid     = false;
    };

    bool demote_slot(const Slot & slot) {
        return host_victim_enabled_ && slot.valid && slot.cache_id >= 0 && slot.size != 0 &&
            host_tier_.store_from_device(slot.cache_id, slot.raw->data, (size_t) slot.size);
    }

    size_t select_victim(uint64_t page_size) const {
        size_t victim = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || slot.valid ||
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

        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || !slot.valid ||
                page_size > slot.capacity) {
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity ||
                (slot.capacity == slots_[victim].capacity &&
                 slot.tick < slots_[victim].tick)) {
                victim = i;
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
            const size_t miss_index =
                state->next.fetch_add(1, std::memory_order_relaxed);
            if (miss_index >= state->misses.size()) {
                return;
            }
            const Miss & miss = state->misses[miss_index];
            auto result = std::make_unique<ReadResult>();
            result->miss_index = miss_index;
            bool read_started = false;
            try {
                result->staging = std::make_unique<StagingPool::Lease>(
                    staging_.borrow());
                if (test_hooks_ != nullptr &&
                    test_hooks_->staging_borrowed) {
                    test_hooks_->staging_borrowed();
                }
                read_started = true;
                if (test_hooks_ != nullptr &&
                    test_hooks_->read_started) {
                    test_hooks_->read_started(
                        miss.page->layer, miss.page->expert);
                }
                if (state->measure) {
                    result->read_started = std::chrono::steady_clock::now();
                    result->read_timed = true;
                }
                read_page(*miss.page, miss.fd, result->staging->get());
            } catch (...) {
                result->error = std::current_exception();
            }
            if (state->measure && result->read_timed) {
                result->read_finished = std::chrono::steady_clock::now();
            }
            if (read_started && test_hooks_ != nullptr &&
                test_hooks_->read_finished) {
                try {
                    test_hooks_->read_finished(
                        miss.page->layer, miss.page->expert);
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
        }
    }

    void complete_batch(Batch & batch) {
        if (batch.completed_) {
            return;
        }

        std::exception_ptr first_error;
        size_t received = 0;
        std::chrono::steady_clock::time_point first_read;
        std::chrono::steady_clock::time_point last_read;
        bool have_read_time = false;
        while (received < batch.state_->misses.size()) {
            std::unique_ptr<ReadResult> result;
            {
                std::unique_lock<std::mutex> lock(batch.state_->mutex);
                batch.state_->cv.wait(lock, [&]() {
                    return !batch.state_->ready.empty();
                });
                result = std::move(batch.state_->ready.front());
                batch.state_->ready.pop_front();
            }

            const Miss & miss =
                batch.state_->misses[result->miss_index];
            if (result->read_timed) {
                if (!have_read_time || result->read_started < first_read) {
                    first_read = result->read_started;
                }
                if (!have_read_time || result->read_finished > last_read) {
                    last_read = result->read_finished;
                }
                have_read_time = true;
            }
            if (result->error != nullptr) {
                if (first_error == nullptr) {
                    first_error = result->error;
                }
            } else {
                // WP_MISS_LOG=path: append "<layer> <expert>" for every page
                // actually READ from disk. Intersecting the two 2026 workers'
                // logs measures whether residency-affinity routing keeps their
                // caches disjoint, or whether they fetch the same pages twice.
                // Default off; one fprintf per miss, negligible against a
                // 13.37 MB O_DIRECT read.
                if (miss_log_ != nullptr) {
                    fprintf(miss_log_, "%d %d\n", miss.page->layer, miss.page->expert);
                    // The harness SIGKILLs workers at teardown, so a buffered
                    // stream is lost entirely -- the first run produced two
                    // 0-byte logs. One fflush per 13.37 MB O_DIRECT read is free.
                    fflush(miss_log_);
                }
                Slot & slot = slots_[miss.slot_index];
                const bool measure_h2d = batch.state_->measure;
                const std::chrono::steady_clock::time_point h2d_started =
                    measure_h2d ? std::chrono::steady_clock::now() :
                                  std::chrono::steady_clock::time_point();
                ggml_backend_tensor_set(
                    slot.raw, result->staging->get(), 0,
                    (size_t) miss.page->size);
                if (measure_h2d) {
                    batch.ns_h2d_ +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - h2d_started).count();
                    batch.bytes_h2d_ += miss.page->size;
                }
                slot.valid = true;
                slot.key   = {
                    miss.page->layer, miss.page->expert
                };
                slot.cache_id = miss.page->cache_id;
                slot.size     = miss.page->size;
                slot.tick  = ++tick_;
                Batch::Entry & entry =
                    batch.entries_[miss.entry_index];
                entry.loaded = {
                    slot.buffer.get(),
                    ggml_backend_buffer_get_base(slot.buffer.get())
                };
                entry.ready = true;
            }
            result.reset();
            ++received;
        }

        for (std::thread & worker : batch.workers_) {
            worker.join();
        }
        batch.workers_.clear();
        batch.completed_ = true;
        if (have_read_time) {
            batch.ns_read_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                last_read - first_read).count();
        }
        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
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

    FILE * miss_log_ = [] {
        const char * p = std::getenv("WP_MISS_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

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

    void read_page(const ExpertPage & page, int fd, void * dst) {
#if defined(__linux__)
        ssize_t n = -1;
        do {
            n = pread(fd, dst, (size_t) page.size, (off_t) page.offset);
        } while (n < 0 && errno == EINTR);
        if (n < 0 || (uint64_t) n != page.size) {
            throw std::runtime_error(
                "short O_DIRECT expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " + std::to_string(page.size));
        }
#else
        (void) page;
        (void) fd;
        (void) dst;
#endif
    }

    ggml_backend_t             backend_ = nullptr;
    ResourcePlan               resources_;
    StagingPool                staging_;
    TestHooks *                test_hooks_ = nullptr;
    wp::HostTier               host_tier_;
    bool                       host_victim_enabled_ = false;
    uint64_t                   tick_ = 0;
    std::vector<Slot>          slots_;
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
    n_hit_(other.n_hit_),
    n_miss_(other.n_miss_),
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
            TestHooks * test_hooks) :
        catalog_(std::move(catalog)),
        backend_(init_backend(device)),
        pool_(
            backend_.get(),
            plan_resources(
                resource_pages(catalog_), slots, host_budget_bytes),
            host_victim_bytes,
            test_hooks),
        compute_galloc_(ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_.get()))),
        slots_(pool_.resources().slot_count) {
        if (!compute_galloc_) {
            throw std::runtime_error("failed to create expert graph allocator");
        }
        stats_.set_staging_kind(pool_.staging_kind());
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
            grow_io_buffer(1u << 20, warmup);
        }
        stats_.set_probe_backend(backend_.get());
        run_self_bench(backend_.get(),
                       catalog_.descriptor.hparams.n_embd,
                       catalog_.descriptor.hparams.n_ff_exp);
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

        std::vector<float> activation(request.activations.size());
        for (size_t i = 0; i < request.activations.size(); ++i) {
            activation[i] = ggml_fp16_to_fp32((ggml_fp16_t) request.activations[i]);
        }
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
            request_stats.n_hit      = batch.n_hit();
            request_stats.n_miss     = batch.n_miss();
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
        bool have_misses = false;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            have_hits |= batch.is_hit(i);
            have_misses |= !batch.is_hit(i);
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
        if (!request.assignments.empty()) {
            prepare_io(activation, request.n_tokens, request_stats);
            request_stats.ns_prep = lap();
            if (have_hits) {
                compute_batch(
                    request, pages, batch, /* hits = */ true,
                    /* add_previous = */ false, request_stats);
            }
            request_stats.ns_hits = lap();
        }
        batch.complete();
        request_stats.ns_wait = lap();
        if (measure) {
            request_stats.ns_read = batch.read_ns();
            request_stats.ns_h2d    = batch.ns_h2d();
            request_stats.bytes_h2d = batch.bytes_h2d();
        }
        if (have_misses) {
            compute_batch(
                request, pages, batch, /* hits = */ false,
                /* add_previous = */ have_hits, request_stats);
        }
        request_stats.ns_miss_compute = lap();
        if (!request.assignments.empty()) {
            read_result(sum, request_stats);
        }
        request_stats.ns_result = lap();
        if (measure) {
            request_stats.ns_compute =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - compute_started).count();
        }

        pipe_expert_partial response;
        response.layer    = request.layer;
        response.n_tokens = request.n_tokens;
        response.partial.resize(sum.size());
        for (size_t i = 0; i < sum.size(); ++i) {
            response.partial[i] = (uint16_t) ggml_fp32_to_fp16(sum[i]);
        }
        request_stats.ns_encode = lap();
        return response;
    }

    const ResourcePlan & resources() const {
        return pool_.resources();
    }

    bool stats_enabled() const {
        return stats_.enabled();
    }

    void record_stats(const RequestStats & request, size_t n_experts) {
        stats_.record(request, n_experts);
    }

private:
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

    void compute_batch(
            const pipe_expert_dispatch_req & request,
            const std::vector<const ExpertPage *> & pages,
            const ExpertSlotPool::Batch & batch,
            bool hits,
            bool add_previous,
            RequestStats & request_stats) {
        size_t n_selected = 0;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            n_selected += batch.is_hit(i) == hits;
        }
        if (n_selected == 0) {
            return;
        }

        const auto build_started = std::chrono::steady_clock::now();
        const size_t tensor_count = 12 * n_selected + 8;
        const size_t graph_nodes = 6 * n_selected + 3;
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
        ggml_tensor * sum = nullptr;
        std::vector<std::pair<ggml_tensor *, const pipe_expert_assignment *>> routing_weights;
        routing_weights.reserve(n_selected);

        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (batch.is_hit(i) != hits) {
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

            ggml_tensor * gate = ggml_mul_mat(ctx.get(), make_weight("gate"), input);
            ggml_tensor * up   = ggml_mul_mat(ctx.get(), make_weight("up"), input);
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
            ggml_tensor * weights =
                ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, request.n_tokens);
            ggml_set_input(weights);
            ggml_tensor * weighted = ggml_mul(ctx.get(), output, weights);
            sum = sum ? ggml_add(ctx.get(), sum, weighted) : weighted;
            routing_weights.emplace_back(weights, &request.assignments[i]);
        }
        if (add_previous) {
            sum = ggml_add(ctx.get(), sum, result);
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

        for (const auto & item : routing_weights) {
            ggml_backend_tensor_set(
                item.first, item.second->weights.data(), 0,
                item.second->weights.size() * sizeof(float));
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
    //   layer n_exp n_hit n_miss bytes_read ns_wall ns_lookup ns_prep ns_hits
    //   ns_wait ns_misscompute ns_result ns_read ns_h2d ns_submit ns_readback
    //   ns_encode ns_send
    // Segment into tokens by watching request.layer wrap back to its minimum.
    FILE * const req_log = [] {
        const char * p = std::getenv("WP_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    // WP_REF_LOG=path -- the full REFERENCE stream: "<layer> <expert> <expert> ..."
    // one line per request, every expert asked for whether it hit or missed.
    // WP_MISS_LOG cannot substitute: which pages miss is a function of the
    // replacement policy, so a miss trace can only ever describe the policy that
    // produced it. The reference stream is policy-independent, which makes LRU /
    // LFU / ARC / Belady all simulatable OFFLINE from a single run, at zero GPU
    // cost per candidate -- and Belady gives the true ceiling, so we learn
    // whether a policy change is worth implementing BEFORE implementing one.
    FILE * const ref_log = [] {
        const char * p = std::getenv("WP_REF_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    while (pipe_recv_frame(socket, type, seq_id, payload)) {
        if (type == PIPE_PING) {
            if (!pipe_send_frame(socket, PIPE_PONG, seq_id, nullptr, 0)) {
                return 1;
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
            // is what made WP_MISS_LOG unusable for the same purpose.
            if (ref_log != nullptr) {
                fprintf(ref_log, "%d", request.layer);
                for (const pipe_expert_assignment & a : request.assignments) {
                    fprintf(ref_log, " %d", a.expert_id);
                }
                fputc('\n', ref_log);
                fflush(ref_log);
            }
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
                        "%d %zu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                        "%llu %llu %llu %llu %llu %llu\n",
                        request.layer, request.assignments.size(),
                        (unsigned long long) s.n_hit,
                        (unsigned long long) s.n_miss,
                        (unsigned long long) s.bytes_read,
                        (unsigned long long) ns_wall,
                        (unsigned long long) s.ns_lookup,
                        (unsigned long long) s.ns_prep,
                        (unsigned long long) s.ns_hits,
                        (unsigned long long) s.ns_wait,
                        (unsigned long long) s.ns_miss_compute,
                        (unsigned long long) s.ns_result,
                        (unsigned long long) s.ns_read,
                        (unsigned long long) s.ns_h2d,
                        (unsigned long long) s.ns_submit,
                        (unsigned long long) s.ns_readback,
                        (unsigned long long) s.ns_encode,
                        (unsigned long long) s.ns_send);
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
        options.test_hooks);
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
        options.test_hooks);
    const pipe_expert_hello advertised = worker.hello();
    const ResourcePlan & resources = worker.resources();

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
