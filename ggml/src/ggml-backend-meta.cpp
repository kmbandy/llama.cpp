#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-ml8.h" // GGML_FP8_QUANT_ROT_KIND_* (FP8_B128 phase 2, see handle_fp8_quant_rot below)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

// ---------------------------------------------------------------------------------------------
// WP_TP_TRACE=1: COMPUTE-flag bookkeeping trace.
//
// A node is disabled on a device when one of its meta sources has a zero-sized slice there
// (ggml_backend_meta_buffer_init_tensor_impl), and a whole window of nodes is disabled when the
// AllReduce at a subgraph boundary is delayed (ggml_backend_meta_graph_compute). Zero-sized
// slices only exist when some tensor is restricted to fewer than n_world devices, i.e. only in a
// cross-host world - so these two sites behave differently on the leader and on the follower and
// there is no way to see that from outside. Everything here is off unless WP_TP_TRACE is set to
// something other than "0"; the environment is read once. One line per site per graph build, plus
// one line per node for the clearing sweep, capped.
//
// Deliberately GGML_LOG_INFO and not GGML_LOG_DEBUG: DEBUG is log level 5 and does not print at
// the -lv 4 the server is run with. Same spelling of the switch as src/llama-tp-lockstep.cpp.
// ---------------------------------------------------------------------------------------------

static bool ggml_backend_meta_trace_enabled() {
    static const bool enabled = []() {
        const char * e = getenv("WP_TP_TRACE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return enabled;
}

// Graph build counter, incremented once per ggml_backend_meta_graph_compute rebuild. The two
// ranks build the same sequence of graphs, so the counter is the join key between the two logs.
static uint64_t g_ggml_backend_meta_trace_build = 0;

// Width of the ubatch currently being processed, published by the host application
// (llama_context::process_ubatch). Trace-only: nothing reads it except the lines below, and it is
// 0 when nobody publishes it.
static int32_t g_ggml_backend_meta_trace_n_tokens = 0;

void ggml_backend_meta_trace_set_ubatch(int32_t n_tokens) {
    g_ggml_backend_meta_trace_n_tokens = n_tokens;
}

// Accumulator for the init_tensor site. init_tensor_impl runs once per tensor, so it cannot log a
// line of its own without drowning the log; it accumulates here instead and the next graph build
// flushes one summary line. The name list is capped.
struct ggml_backend_meta_trace_init_acc {
    static constexpr size_t max_names = 24;
    size_t                   n_tensors = 0; // tensors with >= 1 device slot disabled
    size_t                   n_slots   = 0; // (tensor, local device) pairs disabled
    std::vector<std::string> names;         // first max_names of them, "name[devmask]"

    void reset() { n_tensors = 0; n_slots = 0; names.clear(); }
};

static ggml_backend_meta_trace_init_acc g_ggml_backend_meta_trace_init;

// WP_TP_TRACE=2: per-subgraph cross-host reduce trace.
//
// WHY THIS EXISTS. Everything the trace prints today is structural (how many tensors were
// disabled, how many butterfly ADDs ran); nothing prints a VALUE. The observed failure -
// large prefill ubatches always right, small ones wrong once the process has run a large one,
// and the first real ubatch right at any size - cannot be told apart by structure alone,
// because both candidate mechanisms (a) a stale, previous-shape node0 driving the reduce width
// and (b) a correct-width reduce over a partly-stale buffer produce identical structural
// counters. What separates them is the DATA: the byte width of each reduce and a hash of this
// rank's partial before the exchange plus the world total after it.
//
// HOW TO USE IT. Run the same request twice - once at a size that works and once at a size that
// fails - with WP_TP_TRACE=2 on BOTH ranks, then join the two logs on (build, subgraph):
//   - nbytes differs between the good and the bad run at the SAME subgraph, or differs between
//     the two ranks  => the reduce is being driven by a stale/foreign node0 (hypothesis a).
//   - nbytes identical everywhere, but the FIRST subgraph whose pre= hash differs from the good
//     run is on exactly one rank  => that rank's partial is already wrong before any exchange,
//     and the subgraph index names the layer and the op (hypothesis b). If both ranks' pre=
//     hashes match the good run up to subgraph N and the post= hash diverges at N, the defect is
//     in the exchange/accumulate itself, not in either rank's compute.
// The first subgraph index at which the good and bad runs diverge is the answer; every later
// line is downstream of it.
//
// Level 2 and not 1: this is 128 lines per ubatch. WP_TP_TRACE=1 keeps the existing summaries.
static bool ggml_backend_meta_trace_values_enabled() {
    static const bool enabled = []() {
        const char * e = getenv("WP_TP_TRACE");
        return e != nullptr && e[0] >= '2' && e[0] <= '9';
    }();
    return enabled;
}

static uint64_t ggml_backend_meta_trace_fnv1a(const void * data, size_t nbytes) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < nbytes; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    // Rank window into a (possibly cross-process) world of n_world devices.
    // n_world == simple_devs.size() && rank_first == 0 is the single-process case and is the default.
    size_t n_world;
    size_t rank_first;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud,
            size_t n_world, size_t rank_first) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud),
            n_world(n_world), rank_first(rank_first) {
        GGML_ASSERT(n_world >= this->simple_devs.size());
        GGML_ASSERT(rank_first + this->simple_devs.size() <= n_world);
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < this->simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (this->simple_devs[i]);
            description += ggml_backend_dev_description(this->simple_devs[i]);
        }
        name        += ")";
        description += ")";
        if (n_world != this->simple_devs.size() || rank_first != 0) {
            const std::string window = "[" + std::to_string(rank_first) + ".." +
                std::to_string(rank_first + this->simple_devs.size()) + ")/" + std::to_string(n_world);
            name        += window;
            description += window;
        }
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud, n_world, rank_first)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud, other.n_world, other.rank_first);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev);
static void ggml_backend_meta_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event);
static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event);

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

// Meta events: one event per simple device, recorded on every device's stream
// at once. Lets ggml_backend_sched run the meta backend with rotating input
// copies (parallel=true) so a slot's next graph only waits for the graph that
// last used its input copy instead of ggml_backend_synchronize() draining
// both devices (the per-sub-batch bubble in the rolling TP loop).
struct ggml_backend_meta_event_context {
    std::vector<ggml_backend_event_t> simple;
};

static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_backend_meta_device_context *) dev->context;
    auto * ctx = new ggml_backend_meta_event_context;
    for (ggml_backend_dev_t simple_dev : dev_ctx->simple_devs) {
        ggml_backend_event_t ev = ggml_backend_event_new(simple_dev);
        if (ev == nullptr) {
            for (ggml_backend_event_t e : ctx->simple) {
                ggml_backend_event_free(e);
            }
            delete ctx;
            return nullptr;
        }
        ctx->simple.push_back(ev);
    }
    return new ggml_backend_event { dev, ctx };
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ctx->simple) {
        ggml_backend_event_free(e);
    }
    delete ctx;
    delete event;
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ctx->simple) {
        ggml_backend_event_synchronize(e);
    }
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

size_t ggml_backend_meta_dev_n_world(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->n_world;
}

size_t ggml_backend_meta_dev_rank_first(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->rank_first;
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    return ggml_backend_meta_device_ranked(devs, n_devs, n_devs, 0, get_split_state, get_split_state_ud);
}

ggml_backend_dev_t ggml_backend_meta_device_ranked(
        ggml_backend_dev_t * devs, size_t n_devs, size_t n_world, size_t rank_first,
        ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    GGML_ASSERT(n_world <= GGML_BACKEND_META_MAX_DEVICES);
    GGML_ASSERT(n_devs > 0);
    GGML_ASSERT(rank_first + n_devs <= n_world);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud, n_world, rank_first);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // Rank window, copied from the owning meta device (see ggml_backend_meta_device_ranked).
    // ALL ggml_backend_meta_split_state::ne arrays in this file are indexed with a stride of
    // n_world: ne[segment*n_world + world_device]. Local simple buffer/tensor `j` corresponds to
    // world device `rank_first + j`. With n_world == bufs.size() && rank_first == 0 this is
    // exactly the pre-existing indexing.
    size_t n_world    = 0;
    size_t rank_first = 0;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    // World index of local simple buffer j.
    size_t world_index(size_t j) const {
        return rank_first + j;
    }

    // True when world device jw is owned by this process.
    bool world_index_is_local(size_t jw) const {
        return jw >= rank_first && jw < rank_first + bufs.size();
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

// Stride of the split-state ne array: the number of devices in the WORLD, which is >= the number
// of local simple buffers. Equal to it in the single-process case.
static size_t ggml_backend_meta_buffer_n_world(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->n_world;
}

// Index of local simple buffer 0 within the world.
static size_t ggml_backend_meta_buffer_rank_first(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->rank_first;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

void ggml_backend_meta_buffer_set_op_param_i32(struct ggml_tensor * tensor, int i, int32_t v) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(i >= 0 && i * (int) sizeof(int32_t) < (int) sizeof(tensor->op_params));

    tensor->op_params[i] = v;

    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return;
    }

    // MAD-378 predequant scratch-bound plumbing: propagate to every per-device
    // "simple" clone already materialized for this meta tensor -- see the
    // declaration comment in ggml-backend.h for why the memcpy done once at
    // tensor-init time (below, in ggml_backend_meta_buffer_init_tensor_impl)
    // isn't enough for an op_param whose host-known value changes call-to-call
    // on a graph that gets reused (not rebuilt) across ubatches.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    for (size_t j = 0; j < n_bufs; j++) {
        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
        if (simple_tensor != nullptr) {
            simple_tensor->op_params[i] = v;
        }
    }
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    //
    // Split states are computed over the WORLD, never over the local device window: every rank must
    // derive the same axis and the same per-world-device ne[] for every node, otherwise the ranks
    // disagree about where the subgraph boundaries are and lockstep deadlocks. n_bufs here is
    // therefore the world size (== the local device count in the single-process case).
    const size_t n_bufs = ggml_backend_meta_buffer_n_world(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // TurboQuant Walsh-Hadamard rotation (ggml_turbo_wht), used by the turbo{2,3,4}_0 KV cache to
    // rotate Q before attention and to un-rotate the attention output.
    //
    // The result has exactly src[0]'s ne[] and the kernels (ggml-cpu/ops.cpp
    // ggml_compute_forward_turbo_wht_f32, ggml-cuda/turbo-wht.cu k_turbo_wht_f32) address the data
    // as flat rows of ne[0]: for every row they transform each consecutive run of `group_size`
    // (32/64/128) elements independently - scale, sign flip, butterfly, normalize, sign flip - and
    // copy the ne[0] % group_size tail through unchanged. No value ever crosses a row boundary, so
    // this is a per-row op and the split state simply follows src[0], exactly like NORM/RMS_NORM.
    //
    // ne[0] must not be split. It is the head dim (128 after the turbo zero-padding in
    // llama-kv-cache.cpp cpy_k/cpy_v) and group_size is the whole head dim in the KV path, so any
    // split of ne[0] would cut a butterfly group in half and each device would compute a transform
    // of the wrong length - silently wrong numerics, not a crash. Tensor parallelism shards heads
    // here instead (ne[1] for q_cur / the flash-attn output, ne[2] once q is permuted for
    // flash_attn_ext), which is safe: a device owns whole heads and therefore whole groups.
    //
    // src[1] is the optional InnerQ scale ("turbo_innerq_scale_inv", 128 floats). It is indexed by
    // the position WITHIN a group and reused for every group of every row, so it is not sharded
    // along with the heads - every device needs all of it, i.e. MIRRORED.
    auto handle_turbo_wht = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0 &&
            "turbo_wht mixes values within groups along ne[0], so ne[0] cannot be split");
        GGML_ASSERT(tensor->src[1] == nullptr || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    // ML8_FP8 rotation sidecar (GGML_OP_ML8_APPLY_ROTATION; "ML8_FP8 + rotation
    // under tensor parallel", 2026-09-17). src[0] is the activation x (F32);
    // src[1] is the Kronecker factor h_a (F32 [a,a]) or nullptr. op_params[0]/
    // [1] carry (a_dim, b_dim) -- see ggml_ml8_apply_rotation in ggml-ml8.c.
    // The result always has src[0]'s exact ne[], so like turbo_wht above the
    // split state is just src[0]'s, propagated as-is -- but which split states
    // of src[0] are even legal depends on which of the two rotation kinds this
    // node is:
    //   - kronecker (h_a != nullptr): Q = H_a (x) H_b mixes across the WHOLE
    //     of ne[0], so it is only well-defined when every device already holds
    //     x in full, i.e. src[0] MIRRORED. h_a is a tiny sidecar and must be
    //     MIRRORED too (see the ml8-sidecar rule in
    //     llama_meta_device_get_split_state).
    //   - block_hadamard (h_a == nullptr): independent, normalized Hadamard on
    //     each contiguous b_dim-wide block of ne[0], so it is purely local to
    //     each block and a K-split (AXIS_0) x can be rotated slice-by-slice --
    //     that is the whole point of this variant, it is what lets a K-split
    //     weight's activation (attn_output/ffn_down/ssm_out) be rotated
    //     without an AllReduce first. Also legal, trivially, when x is
    //     MIRRORED (single-device / no TP).
    //   - a kronecker rotation fed a split x, or any split axis other than
    //     MIRRORED/AXIS_0, is a graph-construction bug (the registry should
    //     never emit that combination) -- abort loudly naming the tensor
    //     rather than silently rotate the wrong slice.
    auto handle_ml8_apply_rotation = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(tensor->src[1] == nullptr || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            if (tensor->src[1] != nullptr) {
                GGML_ABORT("ML8_APPLY_ROTATION %s: activation %s is K-split (AXIS_0) but the "
                    "rotation factor h_a (%s) is non-null -- kronecker rotation mixes across the "
                    "whole K dimension and requires a MIRRORED activation; only block_hadamard "
                    "(h_a == NULL) may run on a K-split activation",
                    tensor->name, tensor->src[0]->name, tensor->src[1]->name);
            }
            return src_ss[0];
        }
        GGML_ABORT("ML8_APPLY_ROTATION %s: unsupported split state for activation %s (axis=%d) -- "
            "expected MIRRORED (kronecker) or AXIS_0 (block_hadamard)",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis);
    };

    // GGML_OP_FP8_QUANT_ROT (FP8_B128 phase 2, "meta backend tensor-parallel rules", 2026-09-17).
    // ggml_fp8_quant_rot (ggml-ml8.c): fused activation rotate + block-128 fp8 quantize. src[0] is
    // the activation x (F32); src[1] is the Kronecker factor h_a (F32 [a,a]) or nullptr;
    // op_params[0]/[1]/[2] carry (a_dim, b_dim, kind) with kind 0 = none (per-element
    // copy+quantize), 1 = kronecker, 2 = block_hadamard.
    //
    // This follows GGML_OP_ML8_APPLY_ROTATION's split rules EXACTLY (same rotation math, same
    // per-kind locality) even though, unlike ML8_APPLY_ROTATION, the dst tensor here is NOT
    // src[0]'s ne[] verbatim -- it is I8 [K + 4*K/G, n1, n2, n3] (K/G fp32 scale bytes appended
    // per row; G = op_params[3], 0 meaning PER-ROW, not "128" -- see the GGML_OP_FP8_QUANT_ROT doc
    // comment in ggml.h; G must be passed explicitly, the historical 0-aliases-128 behaviour is
    // gone). Returning src_ss[0] as-is (same axis, same per-device ne) is still correct for the
    // grouped case (G == 32 or 128): the generic epilogue below (the "take over ratio from src"
    // block) recomputes split_state.ne[j] by scaling src[0]'s per-device ne (K_local) by
    // tensor->ne[axis]/src[0]->ne[axis] == (K + 4*K/G)/K, i.e. split_state.ne[j] becomes
    // K_local * (K + 4*K/G) / K. Because K_local % G == 0 for G in {32, 128} (K % 128 == 0 is a
    // hard constraint on this op, and any K-split boundary is itself a multiple of 128 -- see
    // get_split_granularity and the 128-alignment enforced below for FP8_B128/ML8_FP8 weights),
    // that arithmetic is exact and yields precisely K_local + 4*K_local/G: the per-device dst
    // ne0 is derived from the per-device SRC ne0, not from a naive proportional split of
    // dst->ne[0] against the (K + 4*K/G)-wide WORLD dst, which would not equal
    // K_local + 4*K_local/G in general (it does here only because the epilogue's scaling
    // factor is applied to src's ne, not dst's).
    //   - PER-ROW (G == 0) on a K-split (AXIS_0) activation: the same ratio arithmetic gives
    //     K_local * (K + 4) / K, which is NOT an integer in general (unlike the grouped case,
    //     K + 4 need not divide evenly against a K-split boundary). Each device instead rescales
    //     its OWN K slice independently (one local per-row scale over just that slice), which is
    //     mathematically exact because that scale multiplies the corresponding PARTIAL dot
    //     product before the K-split FP8_MUL_MAT results are allreduced. So this case is handled
    //     explicitly, bypassing the generic ratio epilogue entirely: this device's dst ne0 is
    //     simply src0_local_ne0 + 4 (see the special case right after the main switch below,
    //     keyed on tensor->op == GGML_OP_FP8_QUANT_ROT && axis == AXIS_0 && G == 0).
    //   - kronecker (kind 1): mixes across the WHOLE of ne[0], so it is only well-defined when
    //     every device already holds x in full (src[0] MIRRORED); h_a is MIRRORED too.
    //   - block_hadamard (kind 2): independent per b_dim-wide (128) block, so it is purely local
    //     to each block and a K-split (AXIS_0) x is fine -- that is the whole point of this
    //     variant. op_params[0] (a_dim) is rederived per device from the LOCAL K
    //     (K_local / b_dim) in the per-device op_params fixup in
    //     ggml_backend_meta_buffer_init_tensor_impl, exactly like ML8_APPLY_ROTATION's.
    //   - kind 0 (no rotation) is likewise purely per-element/per-group, so K-split is fine too
    //     and needs no a_dim fixup (a_dim/b_dim are unused by the kind-0 compute path).
    //   - a kronecker rotation fed a split x, or any split axis other than MIRRORED/AXIS_0, is a
    //     graph-construction bug (the registry should never emit that combination) -- abort
    //     loudly naming the tensor rather than silently rotate/quantize the wrong slice.
    auto handle_fp8_quant_rot = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const int32_t kind = ggml_get_op_params_i32(tensor, 2);
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(tensor->src[1] == nullptr || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
                GGML_ABORT("FP8_QUANT_ROT %s: activation %s is K-split (AXIS_0) but kind is KRONECKER -- "
                    "kronecker rotation mixes across the whole K dimension and requires a MIRRORED "
                    "activation; only kind NONE/BLOCK_HADAMARD may run on a K-split activation",
                    tensor->name, tensor->src[0]->name);
            }
            GGML_ASSERT(tensor->src[1] == nullptr &&
                "FP8_QUANT_ROT: h_a must be NULL for kind NONE/BLOCK_HADAMARD");
            return src_ss[0];
        }
        GGML_ABORT("FP8_QUANT_ROT %s: unsupported split state for activation %s (axis=%d) -- "
            "expected MIRRORED (kronecker) or AXIS_0 (none/block_hadamard)",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis);
    };

    // GGML_OP_FP8_MUL_MAT (FP8_B128 phase 2). ggml_fp8_mul_mat (ggml-ml8.c): src[0] = w
    // (GGML_TYPE_FP8_B128 or GGML_TYPE_ML8_FP8, [K, N]), src[1] = a (GGML_TYPE_I8,
    // [K + 4*K/G, n1, n2, n3] grouped, or [K + 4, n1, n2, n3] per-row -- the packed output of
    // GGML_OP_FP8_QUANT_ROT). dst = F32 [N, n1, n2, n3]. G is the activation scale-group width:
    // 128 for FP8_B128 weights, 32 for ML8_FP8 weights (used for the weight-side dequant and for
    // the grouped-layout width check) -- but the activation itself may instead be PER-ROW
    // (src[1]'s own op_params[3] == 0, when src[1] is a GGML_OP_FP8_QUANT_ROT node; else inferred
    // from src[1]->ne[0] == src[0]->ne[0] + 4), in which case its row width is K + 4 regardless of
    // G. Same slot roles as GGML_OP_MUL_MAT (src[0] weight, src[1] activation) and the same split
    // rules:
    //   - N-split weight (AXIS_1, e.g. attn_qkv/attn_gate/ffn_gate/ffn_up/output) with a
    //     MIRRORED (or replicated) activation -> dst split on AXIS_0 (each device produces its
    //     own N-slice of the output; no reduce needed).
    //   - K-split weight (AXIS_0, e.g. attn_output/ssm_out/ffn_down) -> the activation MUST be
    //     the K-split QUANT_ROT output, i.e. also AXIS_0 with per-device ne0 == this device's
    //     w-slice ne0 + 4*w-slice ne0/G (grouped) or w-slice ne0 + 4 (per-row), asserted below;
    //     dst is PARTIAL (accumulate via AllReduce), matching plain MUL_MAT's K-split rule
    //     exactly -- per-row is exact here too since each device's own per-slice scale multiplies
    //     that device's own partial dot product before the AllReduce sums them.
    //   - Token-split activations (weight MIRRORED, activation split on a token/batch axis) and
    //     the fully-MIRRORED (single-device / no TP) case follow MUL_MAT's existing behaviour
    //     unchanged.
    auto handle_fp8_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        int32_t G;
        bool a_per_row;
        if (tensor->src[1]->op == GGML_OP_FP8_QUANT_ROT) {
            const int32_t a_G = ggml_get_op_params_i32(tensor->src[1], 3);
            a_per_row = (a_G == 0);
            G = a_per_row ? ((tensor->src[0]->type == GGML_TYPE_ML8_FP8) ? 32 : 128) : a_G;
        } else {
            G = (tensor->src[0]->type == GGML_TYPE_ML8_FP8) ? 32 : 128;
            a_per_row = (tensor->src[1]->ne[0] == tensor->src[0]->ne[0] + 4);
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        // N-split weight, replicated activation -> dst split on axis 0.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        // Replicated weight, token-split activation (MUL_MAT's existing behaviour).
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        // K-split weight x K-split (packed) activation -> PARTIAL, reduced via AllReduce.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            const size_t n_bufs_local = ggml_backend_meta_buffer_n_world(tensor->buffer);
            // per-device element counts: sum over segments of (slice units x repeats)
            auto local_elems = [&](const ggml_backend_meta_split_state & ss, size_t j) {
                int64_t n = 0;
                for (int is = 0; is < ss.n_segments; is++) {
                    n += ss.ne[is*n_bufs_local + j] * (int64_t) ss.nr[is];
                }
                return n;
            };
            for (size_t j = 0; j < n_bufs_local; j++) {
                const int64_t w_local = local_elems(src_ss[0], j);
                const int64_t a_local = local_elems(src_ss[1], j);
                const int64_t a_expected = a_per_row ? (w_local + 4) : (w_local + 4 * w_local / G);
                if (a_local != a_expected) {
                    GGML_LOG_ERROR("%s: FP8_MUL_MAT %s: device %zu G=%d per_row=%d w=%s ne0=%" PRId64 " w_local=%" PRId64 " nr=%u nseg=%d | a=%s ne0=%" PRId64 " a_local=%" PRId64 " nr=%u nseg=%d\n",
                        __func__, tensor->name, j, G, (int) a_per_row, tensor->src[0]->name, tensor->src[0]->ne[0], w_local, src_ss[0].nr[0], src_ss[0].n_segments,
                        tensor->src[1]->name, tensor->src[1]->ne[0], a_local, src_ss[1].nr[0], src_ss[1].n_segments);
                    GGML_ABORT("FP8_MUL_MAT: K-split activation's per-device packed width must be "
                        "this device's w-slice K_local + 4*K_local/G (grouped) or K_local + 4 "
                        "(per-row) -- the QUANT_ROT packing on the SAME K-split as the weight");
                }
            }
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        // Batched matmul: batches split identically on both sides (rare for a 2D FP8_B128
        // weight, kept for symmetry with handle_mul_mat).
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        GGML_ABORT("unsupported fp8_mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
    };

    // GGML_OP_ML8_MUL_MAT (MAD-223 G.4.b, "ml8-4 quantized matmul with separate fp8 centroid
    // LUT"). ggml_ml8_mul_mat (ggml-ml8.c): src[0] = w (GGML_TYPE_ML8_4, [K, N], QK_ML8=64-wide
    // blocks along K), src[1] = centroids (GGML_TYPE_F8_E4M3, [16, K/QK_ML8] LUT sidecar), src[2]
    // = x (GGML_TYPE_F32, [K, M...]). dst = F32 [N, M...]. The only shape difference from
    // MUL_MAT/FP8_MUL_MAT is that the activation is src[2] instead of src[1] -- src[1] here is the
    // LUT sidecar, which under TP is MIRRORED IN FULL on every device (pattern_ml8_sidecar,
    // src/llama-model.cpp) regardless of how the weight is split, so it is asserted MIRRORED
    // unconditionally rather than branched on like MUL_MAT's activation. A K-split weight's
    // per-device offset into that mirrored LUT (op_params[0] = lut_group_off) is derived and
    // written by the per-device op_params fixup in ggml_backend_meta_buffer_init_tensor_impl, not
    // here. Otherwise the same split rules as MUL_MAT/FP8_MUL_MAT, with x taking src[1]'s role:
    //   - w MIRRORED & x MIRRORED -> dst MIRRORED.
    //   - w N-split (AXIS_1) & x MIRRORED -> dst split on AXIS_0 (each device produces only its
    //     own N-slice of the output; no reduce needed).
    //   - w MIRRORED & x token-split (AXIS_1) -> dst follows x's split.
    //   - w K-split (AXIS_0) & x K-split (AXIS_0) -> dst PARTIAL (accumulate via AllReduce), same
    //     as MUL_MAT's K-split rule. Unlike FP8_MUL_MAT's packed/scaled activation, x here is plain
    //     F32 with no extra sidecar bytes, so the per-device element counts of w and x must match
    //     EXACTLY (not just proportionally); each device's w-slice must also be a whole number of
    //     QK_ML8=64 blocks so the dequant never straddles a device boundary.
    //   - anything else -> abort naming the tensor and both axes.
    auto handle_ml8_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            "ML8_MUL_MAT: centroids sidecar must be MIRRORED (mirrored in full on every device)");
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        // N-split weight, replicated activation -> dst split on axis 0.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        // Replicated weight, token-split activation.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            return src_ss[2];
        }
        // K-split weight x K-split activation -> PARTIAL, reduced via AllReduce.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            const size_t n_bufs_local = ggml_backend_meta_buffer_n_world(tensor->buffer);
            const int64_t blck = ggml_blck_size(tensor->src[0]->type);
            // per-device element counts: sum over segments of (slice units x repeats)
            auto local_elems = [&](const ggml_backend_meta_split_state & ss, size_t j) {
                int64_t n = 0;
                for (int is = 0; is < ss.n_segments; is++) {
                    n += ss.ne[is*n_bufs_local + j] * (int64_t) ss.nr[is];
                }
                return n;
            };
            for (size_t j = 0; j < n_bufs_local; j++) {
                const int64_t w_local = local_elems(src_ss[0], j);
                const int64_t x_local = local_elems(src_ss[2], j);
                if (x_local != w_local) {
                    GGML_LOG_ERROR("%s: ML8_MUL_MAT %s: device %zu w=%s w_local=%" PRId64 " nr=%u nseg=%d | x=%s x_local=%" PRId64 " nr=%u nseg=%d\n",
                        __func__, tensor->name, j, tensor->src[0]->name, w_local, src_ss[0].nr[0], src_ss[0].n_segments,
                        tensor->src[2]->name, x_local, src_ss[2].nr[0], src_ss[2].n_segments);
                    GGML_ABORT("ML8_MUL_MAT: K-split activation's per-device element count must equal "
                        "this device's w-slice K_local exactly (no packed sidecar bytes in the "
                        "activation, unlike FP8_MUL_MAT)");
                }
                GGML_ASSERT(w_local % blck == 0 &&
                    "ML8_MUL_MAT: K-split weight slice is not a whole number of QK_ML8=64 blocks");
            }
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        GGML_ABORT("unsupported ml8_mul_mat split states: node=%s src0=%s axis=%d src2=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[2]->name, (int) src_ss[2].axis);
    };

    // GGML_OP_ML8_GET_ROWS (MAD-223 sibling of ML8_MUL_MAT): src[0] = w (GGML_TYPE_ML8_4,
    // [K, N] embedding table), src[1] = centroids (GGML_TYPE_F8_E4M3 LUT sidecar), src[2] = ids
    // (GGML_TYPE_I32). Slot roles are shifted one place from plain GET_ROWS (table/ids ->
    // table/centroids/ids). Unlike plain GET_ROWS, there is no K-split (row-content-split) case
    // here: gathering a row requires the WHOLE row (and its LUT) on one device, so both the table
    // and its sidecar are required MIRRORED, and the result simply follows the ids split -- either
    // replicated (every device gathers the same rows) or split along a token axis (each device
    // gathers only the ids it owns, e.g. after a token-parallel split upstream).
    auto handle_ml8_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            "ML8_GET_ROWS: embedding table must be MIRRORED");
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            "ML8_GET_ROWS: centroids sidecar must be MIRRORED");
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            return src_ss[2];
        }
        GGML_ABORT("ML8_GET_ROWS %s: unsupported ids split state (axis=%d) -- expected MIRRORED or "
            "a token-axis split (AXIS_1)", tensor->name, (int) src_ss[2].axis);
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // a view node can carry only view_src and no src[0] (e.g. views emitted
        // by graph passes that rewire sources); the data parent is equivalent
        const ggml_tensor * vsrc = tensor->src[0] != nullptr ? tensor->src[0] : tensor->view_src;
        GGML_ASSERT(vsrc != nullptr);

        // llm_build_delta_net_base::build_recurrent_attn (src/models/delta-net-base.cpp:400-421)
        // packs GATED_DELTA_NET's per-token output and its updated state into ONE flat result
        // tensor ([S_v*H, n_tokens*n_seqs + K*S_v*n_seqs]) and views the trailing K*S_v*n_seqs
        // rows back out into the state's own [S_v, S_v, H, n_seqs] shape ("new_state"). That
        // view's own nb[] is a textbook contiguous layout for ITS shape, which makes it pass
        // the "both sides contiguous" fast path below and get treated as a plain reshape of
        // vsrc - but vsrc's split axis (0, the flat S_v*H row) describes the S_v/H layout of a
        // DIFFERENT sub-region (the tokens prefix, offset 0); this view starts at a nonzero
        // offset and inserts an extra S_v axis the generic reshape logic cannot relate back to
        // vsrc's split, so it fell through to a bogus raw fraction-of-S_v split on axis 1
        // instead of the axis-2 (head) split the state actually has. The state that comes back
        // out of GATED_DELTA_NET MUST be split exactly like the state that went in - reuse that
        // split directly rather than re-deriving it from the flattened result tensor.
        if (vsrc->op == GGML_OP_GATED_DELTA_NET && tensor->view_offs != 0) {
            const ggml_tensor * state_in = vsrc->src[5];
            if (state_in != nullptr &&
                    tensor->ne[0] == state_in->ne[0] && tensor->ne[1] == state_in->ne[1] &&
                    tensor->ne[2] == state_in->ne[2] && tensor->ne[3] == state_in->ne[3]) {
                return ggml_backend_meta_get_split_state(stc, state_in, /*assume_sync =*/ true);
            }
        }

        const ggml_backend_meta_split_state ss0 = tensor->src[0] != nullptr
            ? src_ss[0]
            : ggml_backend_meta_get_split_state(stc, vsrc, /*assume_sync =*/ true);
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(vsrc)) {
            std::vector<ggml_backend_meta_split_state> vsrc_ss(src_ss);
            vsrc_ss[0] = ss0;
            return handle_reshape(vsrc_ss);
        }
        const int axis = ss0.axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && vsrc->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != vsrc->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return ss0;
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(vsrc) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == vsrc->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (ss0.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || ss0.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return ss0;
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_paged_attn_mt = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // src: 0 q, 1 k_cache, 2 v_cache, 3 block_tables, 4 context_lens,
        //      5 q_lens, 6 k_cur, 7 v_cur, 8 slot_mapping.
        //
        // q / k_cur / v_cur are [head_dim, heads, n_tokens] -> heads on axis 1.
        // The block tables and the per-seq metadata describe the SEQUENCE, not the
        // heads, so every device needs all of them: MIRRORED.
        //
        // k_cache / v_cache are the flat paged arenas. Their AXIS_0 "split" is a
        // sizing device, not a data partition -- see the pattern_paged_cache
        // comment in llama_meta_device_get_split_state. Each device gets an arena
        // for its own KV heads and addresses it with its LOCAL n_kv_heads, which
        // ggml_backend_meta_fixup_op_params writes into op_params[3] per device.
        for (size_t i = 3; i <= 5; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        GGML_ASSERT(src_ss[8].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            // Single-device / fully replicated attention: everything stays whole.
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[6].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[7].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[6].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[7].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                // FP8_B128 (block_fp8_b128, blck_size 128, type_size 130 -- see ggml.h) packs a
                // 128x128 tile scale shared across 128 CONSECUTIVE rows (the N dimension, axis 1)
                // as well as 128 CONSECUTIVE columns (the K dimension, axis 0, which is already
                // what ggml_blck_size enforces below for any quantized type). A loader that split
                // N at a non-128 boundary would hand two devices halves of the same tile, and the
                // in-place CUDA packer (which reads the shared scale from row
                // 128*(n/128) of the WORLD tensor) would read across a device boundary or read a
                // scale that does not belong to the rows this device actually owns. Pin BOTH axes
                // to 128 for this type so a bad N-split (get_split_granularity's job, src/llama-*)
                // aborts here instead of silently corrupting tile scales.
                //
                // ML8_4 (block_ml8_4, blck_size 64) has no shared cross-row tile scale like
                // FP8_B128 -- each block's scale is self-contained -- so a K-split only needs the
                // usual whole-block (64) boundary, already covered by ggml_blck_size below. An
                // N-split, however, feeds the HIP GEMM's native fp8 WMMA path (mt_ml8_gemm), whose
                // N tile is 16 rows wide; an N-split not aligned to 16 would hand a device a
                // partial tile. Pin the N-split granularity to 16 for this type instead of the
                // generic "1" so a bad N-split aborts here too.
                const int64_t granularity = tensor->type == GGML_TYPE_FP8_B128 ? 128
                    : tensor->type == GGML_TYPE_ML8_4 ? (ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 16)
                    : (ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1);
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_SINKHORN_NORM: {
                // NOT handle_per_row: Sinkhorn couples elements across BOTH
                // ne0 and ne1 within a token, so neither of those axes may be
                // split. handle_per_row only forbids axis 0 and would silently
                // permit an axis-1 split.
                GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
                GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
                split_state = src_ss[0];
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_PAGED_ATTN_MT: {
                split_state = handle_paged_attn_mt(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_TURBO_WHT: {
                split_state = handle_turbo_wht(src_ss);
            } break;
            case GGML_OP_ML8_APPLY_ROTATION: {
                split_state = handle_ml8_apply_rotation(src_ss);
            } break;
            case GGML_OP_FP8_QUANT_ROT: {
                split_state = handle_fp8_quant_rot(src_ss);
            } break;
            case GGML_OP_FP8_MUL_MAT: {
                split_state = handle_fp8_mul_mat(src_ss);
            } break;
            case GGML_OP_ML8_MUL_MAT: {
                split_state = handle_ml8_mul_mat(src_ss);
            } break;
            case GGML_OP_ML8_GET_ROWS: {
                split_state = handle_ml8_get_rows(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }

        // FP8_QUANT_ROT per-row (G==0) on a K-split (AXIS_0) activation: see the long comment on
        // handle_fp8_quant_rot above. The generic ratio epilogue below would compute
        // split_state.ne[j] as src0_local_ne0 * tensor->ne[axis]/src0->ne[axis], i.e.
        // K_local*(K+4)/K, which is not an integer in general (unlike the grouped case's
        // K_local*(K+4*K/G)/K, which always is). Bypass the epilogue entirely for this one case
        // and set the per-device dst ne0 directly: this device's own K slice plus 4 (one fp32
        // per-row scale, computed only over that device's local K slice -- see
        // ggml_compute_forward_fp8_quant_rot / the ggml.h doc comment for the exact byte layout).
        if (tensor->op == GGML_OP_FP8_QUANT_ROT && split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 &&
                ggml_get_op_params_i32(tensor, 3) == 0) {
            const size_t n_bufs_pr = ggml_backend_meta_buffer_n_world(tensor->buffer);
            for (uint32_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs_pr; j++) {
                    split_state.ne[s*n_bufs_pr + j] = src_ss[0].ne[s*n_bufs_pr + j] + 4;
                }
            }
            return split_state;
        }

        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_world(tensor->buffer); // world stride, see above

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        if (split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 != sum * tensor->ne[split_state.axis]) {
                            fprintf(stderr, "META SPLIT MISMATCH: node '%s' [%s] buf=%zu; src[%zu]='%s' [%s] axis=%d; "
                                            "node.ne[%d]=%" PRId64 " src.ne[%d]=%" PRId64 " node_share=%" PRId64 " src_share=%" PRId64 "\n",
                                    tensor->name, ggml_op_name(tensor->op), j,
                                    i, tensor->src[i]->name, ggml_op_name(tensor->src[i]->op), (int) src_ss[i].axis,
                                    (int) split_state.axis, tensor->ne[split_state.axis],
                                    (int) src_ss[i].axis, tensor->src[i]->ne[src_ss[i].axis],
                                    split_state.ne[j]*split_state.nr[0], sum);
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }
    const bool trace_cache_was_hit = (it != buf_ctx->split_state_cache.end());

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state =
                        ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);

    // WP_TP_TRACE=3: split_state_cache hit/miss and resolved split, for the tensors on the
    // suspect path between a reproducible cache_s_l0 write-back and a non-reproducible read of
    // it next build (state_predelta-0, its GET_ROWS source, cache_s_l0 itself, conv_states-0,
    // and every other layer's cache_s_l*/cache_r_l* gather). The cache is keyed by tensor
    // POINTER and only invalidated by a raw struct memcmp (just above); a fresh ggml_context is
    // built every llama_decode() call, so a tensor at a given address in THIS build's arena can
    // be a completely different node than whatever last lived there - if the memcmp ever misses
    // that (or if two DIFFERENT tensors this same build alias the same key some other way), a
    // stale ne/axis/offset would be silently reused. Logging hit/miss plus the resolved ne[]
    // per call, across repeated fresh-process runs of the identical request sequence, is what
    // would show that: a "hit" that should have been a "miss" (or the reverse) at build 4.
    if (ggml_backend_meta_trace_values_enabled() && getenv("WP_TP_TRACE")[0] >= '3') {
        bool selected =
            strcmp(tensor->name, "node_23") == 0 ||
            strcmp(tensor->name, "state_predelta-0") == 0 ||
            strcmp(tensor->name, "cache_s_l0") == 0 ||
            strcmp(tensor->name, "conv_states-0") == 0;
        if (!selected && tensor->op == GGML_OP_GET_ROWS && tensor->src[0] != nullptr) {
            selected = strncmp(tensor->src[0]->name, "cache_s_l", 9) == 0 ||
                       strncmp(tensor->src[0]->name, "cache_r_l", 9) == 0;
        }
        if (selected) {
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ",";
                }
                int64_t sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                }
                ne_info += std::to_string(sum);
            }
            GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu site=split_state_cache "
                          "tensor=%p name=%s op=%s cache=%s axis=%s ne=[%s]\n",
                    buf_ctx->rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                    (void *) tensor, tensor->name, ggml_op_name(tensor->op),
                    trace_cache_was_hit ? "hit" : "miss",
                    ggml_backend_meta_split_axis_name(ret.axis), ne_info.c_str());

            // For the GET_ROWS itself: the s_copy index tensor's own resolved split (should be
            // MIRRORED - the row/cache-slot axis is not split, only the channel/head axis is,
            // see handle_get_rows) and, since s_copy is set_input on a host buffer, its actual
            // values - the first 4 indices actually used to gather this build.
            if (tensor->op == GGML_OP_GET_ROWS && tensor->src[1] != nullptr) {
                const ggml_tensor * idx = tensor->src[1];
                std::string idx_vals = "n/a";
                if (ggml_backend_buffer_is_host(idx->buffer) && idx->data != nullptr) {
                    const int32_t * d = (const int32_t *) idx->data;
                    const int64_t n = std::min<int64_t>(4, ggml_nelements(idx));
                    idx_vals.clear();
                    for (int64_t k = 0; k < n; k++) {
                        if (k) idx_vals += ",";
                        idx_vals += std::to_string(d[k]);
                    }
                }
                const ggml_backend_meta_split_state idx_ss =
                    ggml_backend_meta_get_split_state(stc, idx, assume_sync);
                GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu site=split_state_cache_scopy "
                              "node=%s idx=%s idx_axis=%s idx_first4=[%s]\n",
                        buf_ctx->rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                        tensor->name, idx->name,
                        ggml_backend_meta_split_axis_name(idx_ss.axis), idx_vals.c_str());
            }
        }
    }

#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    const size_t n_world       = ggml_backend_meta_buffer_n_world(tensor->buffer);
    const size_t rank_first    = ggml_backend_meta_buffer_rank_first(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_world + rank_first + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (simple_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(simple_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        // Per-device op_params fixup.
        //
        // op_params are memcpy'd verbatim above, which is right for every
        // parameter that describes the MODEL. It is wrong for a parameter that
        // describes the SHAPE this device actually computes, because that shape
        // was just narrowed by the split. Fix those up here, from the per-device
        // sources, after src[] has been rewired.
        switch (t_ij->op) {
            case GGML_OP_PAGED_ATTN_MT: {
                // op_params[3] is n_kv_heads. Under tensor parallelism this device
                // owns only its slice of the KV heads: its paged arena is sized for
                // them and k_cur/v_cur carry only them, so the addressing inside
                // mt_pagedattn must use the LOCAL count. Left at the global value
                // the kernel would stride as if the arena held every head and read
                // the wrong keys -- ggml_cuda_op_paged_attn_mt's
                // k_cur->ne[1] == n_kv_heads assert catches that, loudly.
                GGML_ASSERT(t_ij->src[6] != nullptr);
                const int32_t n_kv_heads_local = (int32_t) t_ij->src[6]->ne[1];
                memcpy((char *) t_ij->op_params + 3*sizeof(int32_t), &n_kv_heads_local, sizeof(int32_t));
            } break;
            case GGML_OP_ML8_APPLY_ROTATION: {
                // op_params[0] is a_dim, op_params[1] is b_dim (ggml_ml8_apply_rotation,
                // ggml-ml8.c); a_dim*b_dim must equal src[0]'s ne[0]. That holds for the
                // WORLD tensor by construction, but when src[0] (the activation) is
                // K-split (AXIS_0, block_hadamard -- see handle_ml8_apply_rotation
                // above) this device only sees its local slice, so a_dim must be
                // rederived from the LOCAL ne[0]. b_dim never changes: it is the fixed
                // Hadamard block size (128) and the K-split boundary is guaranteed to
                // land on a whole multiple of it (get_split_granularity pins K-split
                // weights to a 128-or-whole-head boundary), so the division below is
                // always exact. This also leaves a_dim unchanged in the MIRRORED case,
                // where the local ne[0] equals the world ne[0].
                const int32_t b_dim = ((const int32_t *) t_ij->op_params)[1];
                GGML_ASSERT(t_ij->src[0]->ne[0] % b_dim == 0 &&
                    "ML8_APPLY_ROTATION: this device's local K slice is not a whole multiple of b_dim");
                const int32_t a_dim_local = (int32_t) (t_ij->src[0]->ne[0] / b_dim);
                memcpy((char *) t_ij->op_params, &a_dim_local, sizeof(int32_t));
            } break;
            case GGML_OP_FP8_QUANT_ROT: {
                // op_params[0] is a_dim, op_params[1] is b_dim, op_params[2] is kind (ggml_fp8_quant_rot,
                // ggml-ml8.c). Same rederivation as ML8_APPLY_ROTATION above, but only for kind
                // BLOCK_HADAMARD: that is the only kind handle_fp8_quant_rot allows on a K-split (AXIS_0)
                // activation (kind KRONECKER is asserted MIRRORED-only there, and kind NONE does not read
                // a_dim/b_dim at all -- see ggml_compute_forward_fp8_quant_rot). For kind BLOCK_HADAMARD,
                // a_dim*b_dim must equal src[0]'s LOCAL ne[0]; b_dim (the fixed 128-wide Hadamard block)
                // never changes, and the K-split boundary is guaranteed 128-aligned (enforced on FP8_B128
                // weight splits below), so the division is always exact. Kind MIRRORED (whole-K) case needs
                // no fixup since local ne[0] == world ne[0] already.
                const int32_t kind = ((const int32_t *) t_ij->op_params)[2];
                if (kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD) {
                    const int32_t b_dim = ((const int32_t *) t_ij->op_params)[1];
                    GGML_ASSERT(t_ij->src[0]->ne[0] % b_dim == 0 &&
                        "FP8_QUANT_ROT: this device's local K slice is not a whole multiple of b_dim");
                    const int32_t a_dim_local = (int32_t) (t_ij->src[0]->ne[0] / b_dim);
                    memcpy((char *) t_ij->op_params, &a_dim_local, sizeof(int32_t));
                }
            } break;
            case GGML_OP_ML8_MUL_MAT: {
                // op_params[0] is lut_group_off (GGML_OP_ML8_MUL_MAT, ggml.h): the index of the
                // first QK_ML8=64-wide K-group of the centroid LUT this device's weight slice
                // starts at. The LUT sidecar (t_ij->src[1]) is mirrored in full on every device
                // (pattern_ml8_sidecar, src/llama-model.cpp), so only a K-split (AXIS_0) weight
                // needs a nonzero offset -- an N-split or MIRRORED weight slice starts at K offset
                // 0 and keeps whatever op_params[0] the memcpy above already carried (must be 0:
                // this op has no notion of a K-split WORLD tensor with a nonzero base offset).
                const ggml_backend_meta_split_state w_ss =
                    ggml_backend_meta_get_split_state(tensor->src[0], /*assume_sync =*/ true);
                if (w_ss.axis == GGML_BACKEND_SPLIT_AXIS_0) {
                    // Sum of the LOCAL K element counts of every device before this one, using the
                    // exact same s*n_world + rank_first + j' indexing the split_dim ne[] loop above
                    // uses to compute THIS device's own local ne[split_dim].
                    int64_t k_off = 0;
                    for (size_t jj = 0; jj < j; jj++) {
                        for (size_t s = 0; s < w_ss.n_segments; s++) {
                            k_off += w_ss.ne[s*n_world + rank_first + jj] * (int64_t) w_ss.nr[s];
                        }
                    }
                    const int64_t blck = ggml_blck_size(t_ij->src[0]->type);
                    GGML_ASSERT(k_off % blck == 0 &&
                        "ML8_MUL_MAT: this device's K offset into the centroid LUT is not a whole QK_ML8=64 group");
                    GGML_ASSERT(t_ij->src[0]->ne[0] % blck == 0 &&
                        "ML8_MUL_MAT: this device's local K slice is not a whole number of QK_ML8=64 blocks");
                    const int32_t lut_group_off = (int32_t) (k_off / blck);
                    GGML_ASSERT(t_ij->src[1]->ne[1] >= lut_group_off + t_ij->src[0]->ne[0] / blck &&
                        "ML8_MUL_MAT: centroid LUT sidecar does not cover this device's K-group range");
                    memcpy((char *) t_ij->op_params, &lut_group_off, sizeof(int32_t));
                } else {
                    GGML_ASSERT(((const int32_t *) t_ij->op_params)[0] == 0 &&
                        "ML8_MUL_MAT: lut_group_off must be 0 for a non-K-split weight");
                }
            } break;
            default: break;
        }

        simple_tensors.push_back(t_ij);
    }

    // WP_TP_TRACE=3: per-device split dump for the recurrent chain.
    //
    // The rs trace proves both ranks INTEND the same thing (same rs_z, same s_copy, same head).
    // What it cannot show is what each rank's devices actually got: the state cache, the gather
    // that reads it, the GDN output, the new_state view and the write-back destination each get
    // their split derived by a DIFFERENT rule (cache_s_l by llama_meta_device_get_split_state's
    // pattern_s_cache segments, the GDN output by handle_gated_delta_net, the views off it by
    // handle_reshape's flatten-until-the-products-line-up arithmetic). If the rows a device owns
    // in the state cache do not correspond to the heads it owns in the GDN, the read gathers one
    // set of rows and the write-back stores a different one, and no counter anywhere says so.
    //
    // This dumps, for every tensor in that chain, the WORLD ne[] (what every rank must agree on)
    // and this rank's LOCAL ne/nb3/view_offs/COMPUTE per device, plus the same for each source.
    // Compare, for layer 0, on both ranks:
    //   - cache_s_l0's world ne[] against the GDN output's world ne[]: the per-device row counts
    //     must be in the same proportion (state rows per device == S_v*S_v*H_local),
    //   - the write-back cpy's src local element count against its dst local element count,
    //   - the get_rows result's local ne against what the GDN's state src expects.
    // A device where those disagree is the bug.
    //
    // Selection: anything that stores into a persistent buffer, any GATED_DELTA_NET node, and any
    // tensor whose name contains one of the WP_TP_SPLIT_NAMES substrings (default: the layer-0
    // recurrent chain). Set WP_TP_SPLIT_NAMES to a comma-separated list to widen it.
    // Note: init_tensor runs BEFORE graph_compute bumps the build counter, so these lines belong
    // to the build whose summary line follows them.
    auto trace_split_dump = [&](bool writes_persistent_in) {
        if (!ggml_backend_meta_trace_values_enabled() || getenv("WP_TP_TRACE")[0] < '3') {
            return;
        }
        static const std::vector<std::string> filters = []() {
            const char * e = getenv("WP_TP_SPLIT_NAMES");
            const std::string spec = e && e[0] ? e :
                "cache_s_l0,cache_r_l0,state_predelta-0,attn_output-0,linear_attn_out-0,"
                "conv_states-0,new_state-0,q_conv_predelta-0,k_conv_predelta-0,v_conv_predelta-0,"
                "conv_states_reshaped-0,conv_output_raw-0,attn_norm-0";
            std::vector<std::string> out;
            size_t p = 0;
            while (p <= spec.size()) {
                const size_t q = std::min(spec.find(',', p), spec.size());
                if (q > p) {
                    out.emplace_back(spec.substr(p, q - p));
                }
                p = q + 1;
            }
            return out;
        }();

        bool selected = writes_persistent_in || tensor->op == GGML_OP_GATED_DELTA_NET;
        for (size_t f = 0; !selected && f < filters.size(); f++) {
            selected = std::string(tensor->name).find(filters[f]) != std::string::npos;
        }
        if (!selected) {
            return;
        }

        auto world_ne = [&](const ggml_backend_meta_split_state & ss) {
            std::string s = "axis=";
            s += ggml_backend_meta_split_axis_name(ss.axis);
            s += " nr=" + std::to_string(ss.nr[0]) + " nseg=" + std::to_string(ss.n_segments) + " w=[";
            if (ss.axis >= 0 && ss.axis < GGML_MAX_DIMS) {
                for (size_t jw = 0; jw < n_world; jw++) {
                    int64_t sum = 0;
                    for (size_t s2 = 0; s2 < ss.n_segments; s2++) {
                        sum += ss.ne[s2*n_world + jw] * ss.nr[s2];
                    }
                    s += (jw ? "," : "") + std::to_string(sum);
                }
            }
            return s + "]";
        };

        std::string local;
        for (size_t j = 0; j < n_simple_bufs; j++) {
            const ggml_tensor * t = simple_tensors[j];
            local += (j ? " " : "");
            local += "dev" + std::to_string(rank_first + j) + ":ne=[" +
                std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," +
                std::to_string(t->ne[2]) + "," + std::to_string(t->ne[3]) + "]" +
                " nel=" + std::to_string((long long) ggml_nelements(t)) +
                " nb3=" + std::to_string((long long) t->nb[3]) +
                " offs=" + std::to_string((long long) t->view_offs) +
                " C=" + ((t->flags & GGML_TENSOR_FLAG_COMPUTE) ? "1" : "0");
        }

        std::string srcs;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr) {
                continue;
            }
            srcs += " src" + std::to_string(i) + "=" + tensor->src[i]->name + "[" +
                ggml_op_name(tensor->src[i]->op) + " ne0=" +
                std::to_string((long long) tensor->src[i]->ne[0]);
            if (ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
                srcs += " " + world_ne(ggml_backend_meta_get_split_state(stc, tensor->src[i], true));
            } else {
                srcs += " host";
            }
            srcs += "]";
        }
        std::string vsrc = "none";
        if (tensor->view_src) {
            vsrc = std::string(tensor->view_src->name) + "[usage=" +
                std::to_string((int) ggml_backend_buffer_get_usage(tensor->view_src->buffer)) + "]";
        }

        GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu+1 site=split_dump name=%s op=%s "
                      "ne=[%lld,%lld,%lld,%lld] persist=%d vsrc=%s %s | %s |%s\n",
                rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                tensor->name, ggml_op_name(tensor->op),
                (long long) tensor->ne[0], (long long) tensor->ne[1],
                (long long) tensor->ne[2], (long long) tensor->ne[3],
                writes_persistent_in ? 1 : 0, vsrc.c_str(),
                world_ne(split_state).c_str(), local.c_str(), srcs.c_str());
    };

    // Does this node STORE into a persistent buffer (the KV / recurrent-state cache) rather than
    // produce a value into the graph's own compute buffer? ggml_cpy() and every *_inplace op
    // return a view of their destination, so the view_src chain of such a node bottoms out in a
    // tensor whose buffer usage is not COMPUTE. build_rs() emits exactly two of these per
    // recurrent layer: the ggml_scale_inplace that zeroes the reused state row
    // (src/llama-graph.cpp:4217) and the ggml_cpy that stores the new state back
    // (llm_build_delta_net_base::build_recurrent_attn, src/models/delta-net-base.cpp:551-556).
    bool writes_persistent = false;
    for (const ggml_tensor * v = tensor->view_src; v != nullptr; v = v->view_src) {
        if (v->buffer != nullptr &&
                ggml_backend_buffer_get_usage(v->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            writes_persistent = true;
            break;
        }
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    //
    // WHY A PERSISTENT STORE IS EXEMPT FROM THE *SOURCE* TEST. The rule below exists so that a
    // device holding no rows of a split source does not compute a garbage partial that the next
    // AllReduce would then add in. That reasoning is about a node whose VALUE is consumed. It is
    // wrong for a side-effecting store into the recurrent-state cache: there the source and the
    // destination are different tensors with independently computed splits, so a source that
    // rounds to a zero-sized slice on this device does NOT imply the destination row is empty -
    // and skipping the store leaves that device's persistent row at whatever it held before,
    // forever. The recurrent-state row is zero at construction time
    // (ggml_backend_buffer_clear), so a dropped store is invisible on the first request of a
    // process and wrong on every request after it: the state accumulates across requests on the
    // rank whose devices lost the slice, while the other rank resets normally. Zero-sized slices
    // only exist when a tensor's rows are spread unevenly enough for a device's share to round to
    // nothing (a skewed --tensor-split, or n_head_devices, src/llama.cpp:667), which is a
    // cross-host-only configuration - which is why a single-host -sm tensor run never shows it.
    // This is the same defect the delayed-AllReduce sweep below documents; that sweep was
    // narrowed to transitive consumers, but this site clears the very same nodes and was not.
    //
    // The safe test for a persistent store is the DESTINATION's own slice: when the node's own
    // slice is zero-sized the store writes zero bytes and skipping it is free; when it is not,
    // the store must run.
    uint32_t trace_devmask = 0; // WP_TP_TRACE only: local devices disabled for this tensor
    if (writes_persistent) {
        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            for (size_t j = 0; j < n_simple_bufs; j++) {
                int64_t ne_sum = 0;
                for (size_t s = 0; s < split_state.n_segments; s++) {
                    ne_sum += split_state.ne[s*n_world + rank_first + j] * split_state.nr[s];
                }
                if (ne_sum == 0) {
                    if (ggml_backend_meta_trace_enabled() &&
                            (simple_tensors[j]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
                        trace_devmask |= 1u << j;
                        g_ggml_backend_meta_trace_init.n_slots++;
                    }
                    simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                }
            }
        }
        stc.simple_tensors[tensor] = simple_tensors;
        if (trace_devmask != 0) {
            auto & acc = g_ggml_backend_meta_trace_init;
            acc.n_tensors++;
            if (acc.names.size() < ggml_backend_meta_trace_init_acc::max_names) {
                acc.names.emplace_back(std::string(tensor->name) + "[" + ggml_op_name(tensor->op)
                    + ",persist,devmask=0x" + std::to_string(trace_devmask) + "]");
            }
        }
        trace_split_dump(true);
        return GGML_STATUS_SUCCESS;
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_world + rank_first + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                // WP_TP_TRACE: count it only on the transition, so a tensor whose flag is cleared
                // twice (two zero-sized sources) is not double counted.
                if (ggml_backend_meta_trace_enabled() &&
                        (simple_tensors[j]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
                    trace_devmask |= 1u << j;
                    g_ggml_backend_meta_trace_init.n_slots++;
                }
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    if (trace_devmask != 0) {
        auto & acc = g_ggml_backend_meta_trace_init;
        acc.n_tensors++;
        if (acc.names.size() < ggml_backend_meta_trace_init_acc::max_names) {
            acc.names.emplace_back(std::string(tensor->name) + "[" + ggml_op_name(tensor->op)
                + ",devmask=0x" + std::to_string(trace_devmask) + "]");
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    trace_split_dump(false);

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

// ---------------------------------------------------------------------------------------------
// Load-path profiling, GGML_META_PROFILE=1.
//
// Weight upload through the meta backend is silent and can take minutes on a slow interconnect,
// with no way to tell from outside whether the time is going into the split-state callback, the
// host-side source walk, or the device transfer itself. These counters separate the three. They
// are off by default and cost one steady_clock read per set_tensor call when on.
// ---------------------------------------------------------------------------------------------

struct ggml_backend_meta_profile {
    bool     enabled     = false;
    uint64_t n_set       = 0;   // set_tensor calls
    uint64_t n_transfers = 0;   // ggml_backend_tensor_set_* calls actually issued (per device)
    uint64_t n_rows      = 0;   // rows those transfers cover, i.e. DMA descriptors for a 2D copy
    uint64_t bytes_local = 0;   // bytes written to local devices
    uint64_t bytes_world = 0;   // bytes walked over, including slices owned by other ranks
    uint64_t ns_split    = 0;   // time in the split-state callback
    uint64_t ns_transfer = 0;   // time in the transfer loop

    ggml_backend_meta_profile() {
        const char * env = getenv("GGML_META_PROFILE");
        enabled = env != nullptr && atoi(env) != 0;
    }
    ~ggml_backend_meta_profile() {
        if (!enabled || n_set == 0) {
            return;
        }
        GGML_LOG_INFO("meta profile: set_tensor=%llu transfers=%llu rows=%llu "
                      "local=%.2f GiB world_walked=%.2f GiB split=%.2f s transfer=%.2f s\n",
            (unsigned long long) n_set, (unsigned long long) n_transfers, (unsigned long long) n_rows,
            bytes_local / 1073741824.0, bytes_world / 1073741824.0,
            ns_split / 1e9, ns_transfer / 1e9);
    }
};

static ggml_backend_meta_profile g_meta_profile;

static uint64_t ggml_backend_meta_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Byte size of WORLD device jw's chunk along the split axis, for a single-segment (nr == 1) split.
// For a local device this is exactly the simple tensor's nb[axis+1]; deriving it from the world
// split state instead lets a rank advance the source/destination pointer past chunks it does not
// own without needing a tensor for them.
static size_t ggml_backend_meta_world_chunk_size(
        const ggml_tensor * tensor, const ggml_backend_meta_split_state & split_state, size_t jw, size_t chunk_size_full) {
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);
    const int64_t ne_full = tensor->ne[int(split_state.axis)];
    if (ne_full == 0) {
        return 0;
    }
    const size_t num = chunk_size_full * (size_t) split_state.ne[jw];
    GGML_ASSERT(num % (size_t) ne_full == 0);
    return num / (size_t) ne_full;
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs     = ggml_backend_meta_buffer_n_bufs(buffer);
    const size_t n_world    = ggml_backend_meta_buffer_n_world(buffer);
    const size_t rank_first = ggml_backend_meta_buffer_rank_first(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        const int64_t ne_j = split_state.ne[s*n_world + rank_first + j];
                        GGML_ASSERT(ne_j % blck_size == 0);
                        const size_t nbytes = ne_j/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_world + rank_first + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

// Under a rank window the source buffer handed in here is the FULL tensor: every rank reads the
// same bytes and keeps only its own slices. The running source offset must therefore advance over
// every world device, while ggml_backend_tensor_set_* is called only for the local ones.
static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs     = ggml_backend_meta_buffer_n_bufs(buffer);
    const size_t n_world    = ggml_backend_meta_buffer_n_world(buffer);
    const size_t rank_first = ggml_backend_meta_buffer_rank_first(buffer);
    const uint64_t t_split0 = ggml_backend_meta_now_ns();
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (g_meta_profile.enabled) {
        g_meta_profile.ns_split += ggml_backend_meta_now_ns() - t_split0;
        g_meta_profile.n_set++;
        g_meta_profile.bytes_world += size;
    }
    struct meta_set_timer {
        uint64_t t0;
        meta_set_timer() : t0(ggml_backend_meta_now_ns()) {}
        ~meta_set_timer() {
            if (g_meta_profile.enabled) {
                g_meta_profile.ns_transfer += ggml_backend_meta_now_ns() - t0;
            }
        }
    } meta_set_timer_instance;
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t jw = 0; jw < n_world; jw++) {
                        const int64_t ne_jw = split_state.ne[s*n_world + jw];
                        GGML_ASSERT(ne_jw % blck_size == 0);
                        const size_t nbytes = ne_jw/blck_size * tensor->nb[0];
                        if (jw >= rank_first && jw < rank_first + n_bufs) {
                            const size_t j = jw - rank_first;
                            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                            ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                                simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                                row_count, simple_tensor->nb[1], tensor->nb[1]);
                            simple_offsets[j] += nbytes;
                        }
                        offset_data += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t jw = 0; jw < n_world; jw++) {
                    const size_t nbytes = split_state.ne[s*n_world + jw] * tensor->nb[1];
                    if (jw >= rank_first && jw < rank_first + n_bufs) {
                        const size_t j = jw - rank_first;
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                            row_count, simple_tensor->nb[2], tensor->nb[2]);
                        simple_offsets[j] += nbytes;
                    }
                    offset_data += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t jw = 0; jw < n_world; jw++) {
                const size_t chunk_size_j = ggml_backend_meta_world_chunk_size(tensor, split_state, jw, chunk_size_full);
                if (chunk_size_j == 0) {
                    continue;
                }
                if (jw >= rank_first && jw < rank_first + n_bufs) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, jw - rank_first);
                    GGML_ASSERT(simple_tensor->nb[split_state.axis + 1] == chunk_size_j);
                    const size_t simple_offset = i_start * chunk_size_j;
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                    if (g_meta_profile.enabled) {
                        g_meta_profile.n_transfers++;
                        g_meta_profile.n_rows      += (uint64_t) (i_stop - i_start);
                        g_meta_profile.bytes_local += chunk_size_j * (uint64_t) (i_stop - i_start);
                    }
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            // Count contributors over the WORLD: the value must sum back to `data` only after the
            // local reduce AND the cross-host reduce have both run.
            size_t n_contributors = 0;
            for (size_t jw = 0; jw < n_world; jw++) {
                n_contributors += split_state.ne[jw] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_world;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[rank_first + j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

// Reading a split tensor back gathers every world device's slice into `data`. A rank can only
// supply its own slices, so under a rank window every REMOTE slice of a tensor that is read back
// must be zero-sized. That is a design constraint, not a limitation to work around: the only split
// tensor llama.cpp reads back through this path is the LM head's logits, and the cross-host TP
// configuration deliberately places output.weight entirely on rank 0 precisely so that no
// per-token cross-host vocab gather is needed. A non-zero remote slice here means that placement
// was not applied, which would otherwise show up as silently stale logits.
static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs     = ggml_backend_meta_buffer_n_bufs(buffer);
    const size_t n_world    = ggml_backend_meta_buffer_n_world(buffer);
    const size_t rank_first = ggml_backend_meta_buffer_rank_first(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t jw = 0; jw < n_world; jw++) {
                        const int64_t ne_jw = split_state.ne[s*n_world + jw];
                        GGML_ASSERT(ne_jw % blck_size == 0);
                        const size_t nbytes = ne_jw/blck_size * tensor->nb[0];
                        if (jw >= rank_first && jw < rank_first + n_bufs) {
                            const size_t j = jw - rank_first;
                            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                            ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                                simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                                row_count, simple_tensor->nb[1], tensor->nb[1]);
                            simple_offsets[j] += nbytes;
                        } else {
                            GGML_ASSERT(ne_jw == 0 && "cannot read back a remote slice of a split tensor");
                        }
                        offset_data += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t jw = 0; jw < n_world; jw++) {
                    const int64_t ne_jw = split_state.ne[s*n_world + jw];
                    const size_t nbytes = ne_jw * tensor->nb[1];
                    if (jw >= rank_first && jw < rank_first + n_bufs) {
                        const size_t j = jw - rank_first;
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                            row_count, simple_tensor->nb[2], tensor->nb[2]);
                        simple_offsets[j] += nbytes;
                    } else {
                        GGML_ASSERT(ne_jw == 0 && "cannot read back a remote slice of a split tensor");
                    }
                    offset_data += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t jw = 0; jw < n_world; jw++){
                const size_t chunk_size_j = ggml_backend_meta_world_chunk_size(tensor, split_state, jw, chunk_size_full);
                if (chunk_size_j == 0) {
                    continue;
                }
                GGML_ASSERT((jw >= rank_first && jw < rank_first + n_bufs) &&
                    "cannot read back a remote slice of a split tensor");
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, jw - rank_first);
                GGML_ASSERT(simple_tensor->nb[split_state.axis + 1] == chunk_size_j);
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

// Shared implementation for ggml_backend_meta_tensor_{get,set}_block. See the
// declarations in ggml-backend-impl.h for the contract.
static void ggml_backend_meta_tensor_block_io(
        const struct ggml_tensor * tensor, void * data, size_t block_index, size_t n_blocks, bool write) {
    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(block_index < n_blocks);

    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_meta(tensor->buffer)) {
        const size_t block_size = ggml_nbytes(tensor) / n_blocks;
        GGML_ASSERT(ggml_nbytes(tensor) % n_blocks == 0);
        if (write) {
            ggml_backend_tensor_set((struct ggml_tensor *) tensor, data, block_index*block_size, block_size);
        } else {
            ggml_backend_tensor_get(tensor, data, block_index*block_size, block_size);
        }
        return;
    }

    const struct ggml_backend_meta_split_state split_state =
        ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT((split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ||
                 split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) &&
            "meta block IO requires an axis-0 or mirrored tensor");

    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    // MIRRORED: every device holds the same bytes. Read from device 0; write to
    // all of them so the copies stay in agreement.
    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        for (size_t j = 0; j < n_bufs; j++) {
            struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            GGML_ASSERT(st != nullptr);
            const size_t block_size = ggml_nbytes(st) / n_blocks;
            GGML_ASSERT(ggml_nbytes(st) % n_blocks == 0);
            if (write) {
                ggml_backend_tensor_set(st, data, block_index*block_size, block_size);
            } else {
                ggml_backend_tensor_get(st, data, block_index*block_size, block_size);
                return;
            }
        }
        return;
    }

    char * p = (char *) data;
    for (size_t j = 0; j < n_bufs; j++) {
        struct ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(tensor, j);
        GGML_ASSERT(st != nullptr);
        const size_t nbytes_j = ggml_nbytes(st);
        if (nbytes_j == 0) {
            continue;
        }
        GGML_ASSERT(nbytes_j % n_blocks == 0 &&
                "meta block IO: device slice must divide evenly into n_blocks");
        const size_t block_size_j = nbytes_j / n_blocks;
        if (write) {
            ggml_backend_tensor_set(st, p, block_index*block_size_j, block_size_j);
        } else {
            ggml_backend_tensor_get(st, p, block_index*block_size_j, block_size_j);
        }
        p += block_size_j;
    }
}

void ggml_backend_meta_tensor_get_block(const struct ggml_tensor * tensor, void * data, size_t block_index, size_t n_blocks) {
    ggml_backend_meta_tensor_block_io(tensor, data, block_index, n_blocks, /*write =*/ false);
}

void ggml_backend_meta_tensor_set_block(const struct ggml_tensor * tensor, const void * data, size_t block_index, size_t n_blocks) {
    ggml_backend_meta_tensor_block_io(tensor, (void *) data, block_index, n_blocks, /*write =*/ true);
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

// The rank window lives on the meta DEVICE; every meta buffer type carries that device.
static void ggml_backend_meta_buft_rank_window(ggml_backend_buffer_type_t buft, size_t * n_world, size_t * rank_first) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *n_world    = dev_ctx->n_world;
    *rank_first = dev_ctx->rank_first;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);
    ggml_backend_meta_buft_rank_window(buft, &buf_ctx->n_world, &buf_ctx->rank_first);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    // Maximum number of views per statically allocated tensor that can be
    // created between evals. These contexts are no_alloc, so this bounds tensor
    // HEADERS only -- raising it costs metadata, never VRAM.
    // Raised 16 -> 48: with MTP speculative decoding the draft path adds views
    // on top of the target graph and 16 overflowed by 368 bytes (needed 754032,
    // available 753664), aborting mid-request in
    // ggml_backend_meta_buffer_init_tensor_impl. Sized with margin rather than
    // to the observed need, because the view count scales with the speculative
    // draft depth (--spec-draft-n-max) and would silently re-break on a deeper draft.
    constexpr size_t compute_headroom = 48;
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);
    ggml_backend_meta_buft_rank_window(buft, &meta_buf_ctx->n_world, &meta_buf_ctx->rank_first);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    static constexpr size_t n_graph_slots = 2;

    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs[n_graph_slots];
        std::vector<ggml_tensor *>           nodes[n_graph_slots];
        std::vector<ggml_backend_buffer_ptr> bufs[n_graph_slots];

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            for (size_t i = 0; i < n_graph_slots; i++) {
                bufs[i].resize(n_reduce_steps);
            }
        }
    };
    struct graph_state {
        ggml_context_ptr           ctx;
        std::vector<ggml_cgraph *> cgraphs_aux;
        std::vector<ggml_tensor *> nodes_aux;
        int                        max_nnodes    = 0;
        size_t                     max_tmp_size  = 0;
        size_t                     max_subgraphs = 0;
        size_t                     n_subgraphs   = 0;
        size_t                     next_subgraph = 0;
        uint64_t                   uid           = 0;
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    size_t                      n_reduce_steps;
    graph_state                 graph_states[n_graph_slots];

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_comm_allreduce_begin_t  comm_allreduce_begin = nullptr;
    ggml_backend_comm_allreduce_end_t    comm_allreduce_end = nullptr;

    // Cross-host (inter-process) reduce, installed by the host application. Runs after the local
    // reduce at every reduce point; see ggml_backend_meta_set_cross_host_reduce.
    ggml_backend_meta_cross_host_reduce_t cross_host_reduce    = nullptr;
    void *                                cross_host_reduce_ud = nullptr;

    // Host staging for the cross-host reduce: the partial is read back into it from local device
    // 0, summed with the peer's, and written to every local device. Allocated once from the simple
    // backend's host buffer type - pinned where the backend provides one, which is what makes the
    // per-reduce D2H/H2D pair cheap - and grown only when a wider ubatch appears.
    ggml_backend_buffer_ptr cross_host_buf;
    std::vector<float>      cross_host_buf_fallback;

    // Bounded host run-ahead (GGML_META_RUNAHEAD, default 0 = off; N = subgraphs).
    //
    // Every submission path here is asynchronous: compute() enqueues a
    // subgraph on each device, begin_reduce()/end_reduce() enqueue the
    // AllReduce halves, and nothing waits for completion until the caller
    // reads outputs. The devices, however, depend on each other at every
    // AllReduce, and the vendor runtime's per-device command queue is
    // finite: when it is full the runtime BLOCKS the submitting host thread
    // until packets retire. With one host thread submitting to both devices
    // in order, that is a deadlock whenever device A's queue fills with work
    // that waits on device B while B's matching subgraph has not been
    // submitted yet (2026-09-16: the "16k wedge" -- host thread parked in
    // hipEventRecord inside libamdhip64's queue-full spin, AR watchdog
    // reporting rank 1 never reaching its send). It only triggers when the
    // slower device lags far enough for the faster one's queue to fill,
    // which is why longer contexts (slower attention on the 6900 XT) and
    // anything that adds packets per node (HIP-graph replays, per-node
    // profiling events) made it appear.
    //
    // The fix is to never let the host get more than `runahead_depth`
    // subgraph submissions ahead of the slowest device: before submitting
    // subgraph number s, wait (host-side) for submission s - depth to have
    // completed on every device. That wait cannot itself deadlock -- every
    // dependency of submission s - depth was submitted before it -- and it
    // bounds the per-device queue occupancy to depth x (packets per
    // subgraph), far below the queue size.
    std::vector<std::vector<ggml_backend_event_t>> runahead_ev; // [backend][ring]
    size_t                                          runahead_depth = 0;
    uint64_t                                        runahead_seq   = 0;

    void runahead_before_submit() {
        if (runahead_depth == 0) {
            return;
        }
        if (runahead_seq >= runahead_depth) {
            const size_t k = runahead_seq % runahead_depth;
            for (auto & evs : runahead_ev) {
                ggml_backend_event_synchronize(evs[k]);
            }
        }
    }

    void runahead_after_submit(size_t j) {
        if (runahead_depth == 0) {
            return;
        }
        const size_t k = runahead_seq % runahead_depth;
        ggml_backend_event_record(runahead_ev[j][k], backend_configs[j].backend);
    }

    void runahead_advance() {
        if (runahead_depth != 0) {
            runahead_seq++;
        }
    }

    float * cross_host_staging(size_t nbytes) {
        if (!cross_host_buf || ggml_backend_buffer_get_size(cross_host_buf.get()) < nbytes) {
            ggml_backend_buffer_type_t host_buft =
                ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_configs[0].backend));
            cross_host_buf.reset(host_buft ? ggml_backend_buft_alloc_buffer(host_buft, nbytes) : nullptr);
        }
        if (cross_host_buf) {
            return (float *) ggml_backend_buffer_get_base(cross_host_buf.get());
        }
        // No host buffer type on this backend (e.g. a CPU-only test build): plain memory. Resized
        // only when it grows, never per call.
        if (cross_host_buf_fallback.size() * sizeof(float) < nbytes) {
            cross_host_buf_fallback.resize(nbytes / sizeof(float));
        }
        return cross_host_buf_fallback.data();
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            // Default OFF: measured 2026-09-16 on qwen38-27b-q8-tp, depth 4 cost
            // ~25% on an aligned 8192-token batch (1350 -> 1000 t/s) because the
            // host wait lands before the slower device's queue is refilled. Keep
            // as an opt-in for diagnosing host-side queue-full blocks.
            size_t depth = 0;
            if (const char * e = getenv("GGML_META_RUNAHEAD")) {
                depth = (size_t) atoi(e);
            }
            runahead_ev.resize(n_devs);
            for (size_t i = 0; i < n_devs && depth > 0; i++) {
                ggml_backend_dev_t simple_dev = ggml_backend_get_device(simple_backends[i]);
                for (size_t k = 0; k < depth; k++) {
                    ggml_backend_event_t ev = ggml_backend_event_new(simple_dev);
                    if (ev == nullptr) {
                        depth = 0; // backend has no events: cannot bound, run unbounded as before
                        break;
                    }
                    runahead_ev[i].push_back(ev);
                }
            }
            runahead_depth = depth;
            if (runahead_depth == 0) {
                for (auto & evs : runahead_ev) {
                    for (auto ev : evs) {
                        ggml_backend_event_free(ev);
                    }
                    evs.clear();
                }
            }
        }

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            comm_allreduce_begin = (ggml_backend_comm_allreduce_begin_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_begin");
            comm_allreduce_end = (ggml_backend_comm_allreduce_end_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_end");
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & evs : runahead_ev) {
            for (auto ev : evs) {
                ggml_backend_event_free(ev);
            }
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const size_t n_world    = ggml_backend_meta_buffer_n_world(tensor->buffer);
    const size_t rank_first = ggml_backend_meta_buffer_rank_first(tensor->buffer);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        // Layouts the async splice does not cover: synchronous buffer path.
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t jw = 0; jw < n_world; jw++){
                const size_t chunk_size_j = ggml_backend_meta_world_chunk_size(tensor, split_state, jw, chunk_size_full);
                if (chunk_size_j == 0) {
                    continue;
                }
                if (jw >= rank_first && jw < rank_first + n_backends) {
                    const size_t j = jw - rank_first;
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    GGML_ASSERT(simple_tensor->nb[split_state.axis + 1] == chunk_size_j);
                    ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                        i_stop - i_start, chunk_size_j, chunk_size_full);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            // Layouts the async splice does not cover: take the synchronous
            // buffer path (correct, just not queued behind in-flight work).
            static int warned = 0;
            if (warned < 4) {
                warned++;
                GGML_LOG_WARN("%s: tensor %s: split axis %d (n_segments=%zu nr0=%zu) type=%s ne=[%lld,%lld,%lld,%lld] view_src=%s not handled asynchronously, using synchronous set\n",
                              __func__, tensor->name, (int) split_state.axis, split_state.n_segments, split_state.nr[0],
                              ggml_type_name(tensor->type), (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                              tensor->view_src ? tensor->view_src->name : "-");
            }
            ggml_backend_tensor_set(tensor, data, offset, size);
        }
    }
}

// See the note on ggml_backend_meta_buffer_get_tensor: remote slices cannot be gathered here.
static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const size_t n_world    = ggml_backend_meta_buffer_n_world(tensor->buffer);
    const size_t rank_first = ggml_backend_meta_buffer_rank_first(tensor->buffer);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t jw = 0; jw < n_world; jw++){
                const size_t chunk_size_j = ggml_backend_meta_world_chunk_size(tensor, split_state, jw, chunk_size_full);
                if (chunk_size_j == 0) {
                    continue;
                }
                GGML_ASSERT((jw >= rank_first && jw < rank_first + n_backends) &&
                    "cannot read back a remote slice of a split tensor");
                const size_t j = jw - rank_first;
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                GGML_ASSERT(simple_tensor->nb[split_state.axis + 1] == chunk_size_j);
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ctx->simple.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_record(ctx->simple[i], ggml_backend_meta_simple_backend(backend, i));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ctx->simple.size() == n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, i), ctx->simple[i]);
    }
}

static void ggml_backend_meta_graph_prepare(
        ggml_backend_t backend, struct ggml_cgraph * cgraph, const size_t i_slot) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    auto & gs = backend_ctx->graph_states[i_slot];

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != gs.uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > gs.max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes[i_slot].resize(cgraph->n_nodes);
            bcj.cgraphs[i_slot].resize(cgraph->n_nodes);
        }
        gs.max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    const size_t trace_rank_first = ggml_backend_meta_dev_rank_first(backend->device);

    if (needs_rebuild && ggml_backend_meta_trace_enabled()) {
        // WP_TP_TRACE: one line per graph build for the init_tensor clearing site
        // (ggml_backend_meta_buffer_init_tensor_impl, "a source has a zero-sized slice here").
        // The counters cover every tensor initialised since the previous build, which is exactly
        // the allocation round for this graph.
        g_ggml_backend_meta_trace_build++;
        auto & acc = g_ggml_backend_meta_trace_init;
        std::string names;
        for (const std::string & n : acc.names) {
            if (!names.empty()) {
                names += " ";
            }
            names += n;
        }
        GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu site=init_tensor_zero_slice "
                      "n_tokens=%d n_nodes=%d cleared_tensors=%zu cleared_slots=%zu%s%s\n",
                trace_rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                (int) g_ggml_backend_meta_trace_n_tokens, cgraph->n_nodes,
                acc.n_tensors, acc.n_slots,
                names.empty() ? "" : " names=", names.c_str());
        acc.reset();
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i_slot][i] = node;
                    continue;
                }
                bcj.nodes[i_slot][i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i_slot][i]);
            }
        }

        {
            // World size of this meta device. The subgraph boundary set below must be derived from
            // it and never from the local device window: two ranks with different LOCAL device sets
            // would otherwise derive different n_subgraphs from the same graph and deadlock the
            // moment they try to exchange partials in lockstep.
            const size_t n_world_g = ggml_backend_meta_dev_n_world(backend->device);

            // World-invariant restatement of "the simple tensor of `node` on device jw has
            // GGML_TENSOR_FLAG_COMPUTE set". Reproduces exactly the rule applied per local device in
            // ggml_backend_meta_buffer_init_tensor_impl - a node is disabled on a device when any of
            // its meta-buffer sources has a zero-sized slice there - but evaluated against the WORLD
            // split state, so every rank gets the same answer for every world device.
            auto node_computes_world = [&](const ggml_tensor * node, const size_t jw) -> bool {
                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    return false;
                }
                for (int is = 0; is < GGML_MAX_SRC; is++) {
                    const ggml_tensor * src = node->src[is];
                    if (src == nullptr || !ggml_backend_buffer_is_meta(src->buffer)) {
                        continue;
                    }
                    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(src, /*assume_sync =*/ true);
                    if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
                        continue;
                    }
                    int64_t ne_sum = 0;
                    for (size_t sg = 0; sg < ss.n_segments; sg++) {
                        ne_sum += ss.ne[sg*n_world_g + jw] * ss.nr[sg];
                    }
                    if (ne_sum == 0) {
                        return false;
                    }
                }
                return true;
            };

            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    // WORLD, not n_backends: see node_computes_world above (spec B.4).
                    for (size_t jw = 0; jw < n_world_g; jw++) {
                        const bool compute       = node_computes_world(cgraph->nodes[i],       jw);
                        const bool compute_other = node_computes_world(cgraph->nodes[i_other], jw);
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            // WP_TP_TRACE accumulators for the delayed-AllReduce clearing sweep. One summary
            // line per graph build is emitted after the boundary loop below.
            size_t                   trace_sweep_windows      = 0;
            size_t                   trace_sweep_window_nodes = 0;
            size_t                   trace_sweep_tainted      = 0;
            size_t                   trace_sweep_cleared      = 0;
            std::vector<std::string> trace_sweep_names;

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                //
                // ...but only the nodes that actually CONSUME node i. [i+1, i_delayed] is an index
                // range, not a dependency cone: get_i_delayed()'s skip_unrelated() deliberately
                // steps over MIRRORED nodes that do not consume node i, so the range also holds
                // ordinary, independent work - including the side-effecting writes into the
                // PERSISTENT KV / recurrent-state buffers (the ggml_scale_inplace that zeroes the
                // reused recurrent row, src/llama-graph.cpp:4217-4218, and the ggml_cpy that
                // stores the new state, :4227-4232). Clearing their COMPUTE flag makes the device
                // skip them outright, which silently drops those writes: invisible on the first
                // request, whose recurrent rows are still zero from the construction-time
                // ggml_backend_buffer_clear, and wrong on every request after it. Zero-sized
                // slices only exist when a tensor is restricted to fewer than n_world devices
                // (n_head_devices, src/llama.cpp:667), which is a cross-host-only configuration -
                // which is why a single-host -sm tensor run never showed this.
                if (i_delayed > i) {
                    // Transitive consumers of node i inside the window, on the ORIGINAL graph:
                    // the dependency structure is world-invariant, only the per-device COMPUTE
                    // flag below is not.
                    std::set<const ggml_tensor *> tainted;
                    std::vector<int>              tainted_idx;
                    tainted.insert(cgraph->nodes[i]);
                    for (int ii = i + 1; ii <= i_delayed; ii++) {
                        const ggml_tensor * n_ii = cgraph->nodes[ii];
                        bool depends = false;
                        for (const ggml_tensor * v = n_ii->view_src; v != nullptr && !depends; v = v->view_src) {
                            depends = tainted.count(v) > 0;
                        }
                        for (int is = 0; is < GGML_MAX_SRC && !depends; is++) {
                            for (const ggml_tensor * src = n_ii->src[is]; src != nullptr; src = src->view_src) {
                                if (tainted.count(src) > 0) {
                                    depends = true;
                                    break;
                                }
                            }
                        }
                        if (depends) {
                            tainted.insert(n_ii);
                            tainted_idx.push_back(ii);
                        }
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i_slot][i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
                            continue;
                        }
                        for (const int ii : tainted_idx) {
                            // Never touch the shared original. For the s_copy views bcj.nodes[i_slot][ii]
                            // IS cgraph->nodes[ii] (see the FIXME where bcj.nodes is filled), and
                            // clearing the flag there would be seen by every local device AND by
                            // node_computes_world() - which every rank evaluates to derive the
                            // subgraph boundaries - turning a per-device decision into a per-rank
                            // mutation of a world-invariant input.
                            if (bcj.nodes[i_slot][ii] == cgraph->nodes[ii]) {
                                continue;
                            }
                            bcj.nodes[i_slot][ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            trace_sweep_cleared++;
                        }
                    }

                    if (ggml_backend_meta_trace_enabled()) {
                        trace_sweep_windows++;
                        trace_sweep_window_nodes += size_t(i_delayed - i);
                        trace_sweep_tainted      += tainted_idx.size();
                        // The nodes the old blanket sweep would have disabled and this one does
                        // not: the whole point of the trace. Capped.
                        for (int ii = i + 1; ii <= i_delayed && trace_sweep_names.size() < 24; ii++) {
                            if (std::find(tainted_idx.begin(), tainted_idx.end(), ii) != tainted_idx.end()) {
                                continue;
                            }
                            trace_sweep_names.emplace_back(
                                std::string(cgraph->nodes[ii]->name) + "[" + ggml_op_name(cgraph->nodes[ii]->op) + "]");
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[i_slot][n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);

            if (ggml_backend_meta_trace_enabled()) {
                std::string names;
                for (const std::string & n : trace_sweep_names) {
                    if (!names.empty()) {
                        names += " ";
                    }
                    names += n;
                }
                GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu site=delayed_allreduce_sweep "
                              "n_tokens=%d n_nodes=%d n_subgraphs=%zu windows=%zu window_nodes=%zu "
                              "dependent=%zu cleared_slots=%zu%s%s\n",
                        trace_rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                        (int) g_ggml_backend_meta_trace_n_tokens, cgraph->n_nodes, n_subgraphs,
                        trace_sweep_windows, trace_sweep_window_nodes, trace_sweep_tainted,
                        trace_sweep_cleared,
                        names.empty() ? "" : " spared=", names.c_str());
            }
        }

        // WP_TP_TRACE: DISABLED-PRODUCER / ENABLED-CONSUMER CHECK.
        //
        // The zero-slice rule (ggml_backend_meta_buffer_init_tensor_impl) clears COMPUTE on a
        // node for a device when one of its sources has no rows there. It does NOT propagate:
        // a node whose own sources all have rows keeps COMPUTE even if one of those sources is
        // itself a node that was disabled on this device. Such a consumer then reads a compute
        // buffer region that nothing ever wrote - zero on a fresh allocation (so the first
        // request of a process looks correct) and whatever the previous graph left there
        // afterwards (so identical requests drift, and the drift depends on the previous
        // graph's allocation layout, i.e. on the previous ubatch's n_tokens).
        //
        // Zero-sized slices only exist when a device's share of some tensor rounds to nothing,
        // which needs an uneven world split - so this can only bite cross-host, and only on the
        // rank holding the small shares. That is exactly the observed failure.
        //
        // This runs after BOTH clearing sites (the per-tensor zero-slice rule at init_tensor and
        // the delayed-AllReduce sweep above), so what it sees is the final flag state the
        // devices will actually execute. Capped; one line per violation.
        //
        // If this prints anything, the fix is to propagate the disable to the consumer (or to
        // zero the producer's output on that device the way allreduce_fallback zeroes a disabled
        // subgraph tail). If it prints nothing, the reads are all fed by enabled producers and
        // the stale-memory hypothesis is dead.
        if (ggml_backend_meta_trace_enabled()) {
            const size_t max_lines = 40;
            const size_t n_world_chk = ggml_backend_meta_dev_n_world(backend->device);
            size_t n_viol = 0, n_lines = 0;
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    ggml_tensor * node   = cgraph->nodes[i];
                    ggml_tensor * node_j = bcj.nodes[i_slot][i];
                    // the host-side s_copy views share the original tensor; they have no
                    // per-device copy and no COMPUTE decision of their own
                    if (node_j == nullptr || node_j == node) {
                        continue;
                    }
                    if ((node_j->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue; // this consumer is itself disabled: fine
                    }
                    for (int is = 0; is < GGML_MAX_SRC; is++) {
                        // walk the view_src chain too: a consumer often reads a VIEW of the
                        // disabled producer rather than the producer itself
                        for (const ggml_tensor * src = node->src[is]; src != nullptr; src = src->view_src) {
                            if (src == node || src->buffer == nullptr ||
                                    !ggml_backend_buffer_is_meta(src->buffer)) {
                                continue;
                            }
                            // graph inputs/leaves are written by set_input (or are weights), not
                            // by a producer node with a COMPUTE decision of its own: a disabled
                            // COMPUTE flag on their "copy" just means this device never needed a
                            // local slice, not that nothing wrote it. Only a genuine computed
                            // producer (has an op, not itself a leaf) can be "never written".
                            if ((src->flags & GGML_TENSOR_FLAG_INPUT) != 0 || src->op == GGML_OP_NONE) {
                                continue;
                            }
                            // only graph-produced values can be "never written"; a weight or a
                            // persistent cache row always holds something meaningful
                            if (ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                                continue;
                            }
                            const ggml_tensor * src_j = ggml_backend_meta_buffer_simple_tensor(src, j);
                            if (src_j == nullptr || (src_j->flags & GGML_TENSOR_FLAG_COMPUTE) != 0) {
                                continue;
                            }
                            // Is the producer's slice on this device actually empty? If it is,
                            // the consumer reads zero rows of it and the violation is benign.
                            // If it is NOT, the consumer reads rows nothing wrote: the bug.
                            const ggml_backend_meta_split_state ss =
                                ggml_backend_meta_get_split_state(src, /*assume_sync =*/ true);
                            int64_t src_rows = -1; // -1 = not row-split (MIRRORED/PARTIAL)
                            if (ss.axis >= 0 && ss.axis < GGML_MAX_DIMS) {
                                src_rows = 0;
                                for (size_t s2 = 0; s2 < ss.n_segments; s2++) {
                                    src_rows += ss.ne[s2*n_world_chk + trace_rank_first + j] * ss.nr[s2];
                                }
                            }
                            // src_rows == 0: the producer's local slice is empty, so the consumer
                            // reads zero rows of it - benign, don't report.
                            // src_rows == -1: not row-split (MIRRORED/PARTIAL) - we can't tell
                            // whether the device's copy is meaningful from the split state alone,
                            // so don't guess; only a definite non-empty slice (src_rows > 0) is a
                            // genuine READS_UNWRITTEN.
                            if (src_rows <= 0) {
                                break;
                            }
                            n_viol++;
                            if (n_lines < max_lines) {
                                n_lines++;
                                GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu+1 "
                                              "site=disabled_producer dev=%zu node=%s[%s] "
                                              "src%d=%s[%s] src_slice_rows=%lld src_nel_dev=%lld "
                                              "verdict=READS_UNWRITTEN\n",
                                        trace_rank_first,
                                        (unsigned long long) g_ggml_backend_meta_trace_build,
                                        trace_rank_first + j, node->name, ggml_op_name(node->op),
                                        is, src->name, ggml_op_name(src->op),
                                        (long long) src_rows,
                                        (long long) ggml_nelements(src_j));
                            }
                            break; // one report per (node, src) chain
                        }
                    }
                }
            }
            GGML_LOG_INFO("WP_TP_TRACE meta rank_first=%zu build=%llu+1 site=disabled_producer_summary "
                          "n_tokens=%d violations=%zu printed=%zu\n",
                    trace_rank_first, (unsigned long long) g_ggml_backend_meta_trace_build,
                    (int) g_ggml_backend_meta_trace_n_tokens, n_viol, n_lines);
        }

        gs.uid         = cgraph->uid;
        gs.n_subgraphs = n_subgraphs;

        if (max_tmp_size > gs.max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i_slot][i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            gs.max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > gs.max_subgraphs) {
            gs.max_subgraphs = std::max(gs.max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = gs.max_subgraphs*ggml_graph_overhead_custom(gs.max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*gs.max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*gs.max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            gs.ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i_slot][i].cgraph_main = ggml_new_graph_custom(gs.ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            gs.cgraphs_aux.resize(n_backends*n_cgraphs_per_device*gs.max_subgraphs);
            for (size_t k = 0; k < gs.cgraphs_aux.size(); k++) {
                gs.cgraphs_aux[k] = ggml_new_graph_custom(gs.ctx.get(), 1, cgraph->grads);
            }
            gs.nodes_aux.resize(n_backends*n_nodes_per_device*gs.max_subgraphs);
            for (size_t k = 0; k < gs.nodes_aux.size(); k++) {
                gs.nodes_aux[k] = ggml_new_tensor_1d(gs.ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_slot][i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_slot][i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_slot][i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_slot][i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

}

struct ggml_backend_meta_graph_runner {
    ggml_backend_meta_context * backend_ctx;
    size_t                      i_slot;
    size_t                      n_backends;
    size_t                      i_graph_aux = 0;
    size_t                      i_node_aux  = 0;

    ggml_tensor * get_node_aux(ggml_tensor * t) {
        auto & gs = backend_ctx->graph_states[i_slot];
        ggml_tensor * ret = gs.nodes_aux[i_node_aux++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    }

    void set_tmp_data(ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        auto & gs = backend_ctx->graph_states[i_slot];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_slot][i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < gs.max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, gs.max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    }

    ggml_cgraph * get_cgraph_aux() {
        return backend_ctx->graph_states[i_slot].cgraphs_aux[i_graph_aux++];
    }

    ggml_status allreduce_fallback(const size_t i) {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i_slot][i].cgraph_main->nodes[bcj.cgraphs[i_slot][i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE;
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i_slot][i].cgraph_main->nodes[bcj_src.cgraphs[i_slot][i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i_slot][i].cgraph_main->nodes[bcj_dst.cgraphs[i_slot][i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);
            // MEASURED (2026-09-06, j9-j16, cross-host CUDA0+Vulkan0 pair): both devices' own
            // PRE-reduce partials are individually bit-reproducible across fresh-process runs
            // (site=sub0_node), but after this local butterfly reduce, CUDA0's combined value is
            // NOT reproducible while Vulkan0's is - even after adding an explicit
            // ggml_backend_synchronize(bcj_dst.backend) right after the copy (commit e49787ccb),
            // which ruled out plain missing-wait-before-the-ADD timing.
            //
            // ggml_backend_tensor_copy_async's generic fallback (ggml-backend.cpp:517-533), for a
            // cross-vendor pair with no vendor fast path, does a host-bounce copy via the
            // BUFFER-level ggml_backend_tensor_set (ggml-backend.cpp:505), which for CUDA calls
            // ggml_backend_cuda_buffer_set_tensor (ggml-cuda.cu:990-1005): a plain, host-blocking
            // cudaMemcpy on the LEGACY DEFAULT STREAM. CUDA streams in this backend are created
            // with cudaStreamNonBlocking (ggml-cuda.cu:822 - ggml_cuda_context::stream()), which
            // are BY DESIGN exempt from the legacy stream's implicit cross-stream synchronization.
            // MAD-114's comment claims the blocking memcpy gives "device-wide ordering" - true for
            // HIP/ROCm, where that fix was made, but NOT the same guarantee CUDA's own stream
            // model gives against a cudaStreamNonBlocking stream: a legacy-stream operation is not
            // ordered against work enqueued on such a stream just because the host waited for it.
            // The ADD that reads node_tmp gets submitted moments later to exactly that kind of
            // stream (cuda_ctx->stream()), so it is not guaranteed to see the bounce copy's write.
            // Same class of defect as 603147cf5 in the worker.
            //
            // Fix: use the BACKEND-level async set (ggml_backend_tensor_set_async), which for CUDA
            // is ggml_backend_cuda_set_tensor_async (ggml-cuda.cu:3730) - cudaMemcpyAsync issued
            // ON cuda_ctx->stream() itself. The ADD is submitted to that SAME stream right after,
            // so plain stream FIFO ordering (not a separate host-side wait) makes it correct by
            // construction, on any backend, not just this pair. Falls back to a synchronous
            // set through ggml_backend_tensor_set_async's own dispatcher when a backend has no
            // .set_tensor_async (ggml-backend.cpp:271-282), same as before for such backends.
            //
            // The source read stays a plain (non-async) ggml_backend_tensor_get: it already forces
            // full completion (blocking) into a host buffer we then own, which is what we need
            // before handing that buffer to the async set below - and get_tensor's own
            // completeness is a separate question from this stream-ordering bug (see the
            // site=node_tmp_after_copy trace below, which now runs after the FIXED copy so a
            // remaining divergence there points at the read side instead).
            {
                const size_t nbytes = ggml_nbytes(node_src);
                std::vector<char> host_bounce(nbytes);
                ggml_backend_tensor_get(node_src, host_bounce.data(), 0, nbytes);
                ggml_backend_tensor_set_async(bcj_dst.backend, node_tmp, host_bounce.data(), 0, nbytes);
            }
            ggml_backend_synchronize(bcj_dst.backend);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);
            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }
            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];
            ggml_tensor * node_src = bcj_src.cgraphs[i_slot][i].cgraph_main->nodes[bcj_src.cgraphs[i_slot][i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i_slot][i].cgraph_main->nodes[bcj_dst.cgraphs[i_slot][i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }
        return GGML_STATUS_SUCCESS;
    }

    ggml_status compute(const size_t i) {
        backend_ctx->runahead_before_submit();
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i_slot][i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            backend_ctx->runahead_after_submit(j);
        }
        backend_ctx->runahead_advance();
        return GGML_STATUS_SUCCESS;
    }

    bool blocking_reduce(const size_t i) {
        bool success = false;
        if (backend_ctx->comm_ctx) {
            std::vector<ggml_tensor *> nodes;
            nodes.reserve(n_backends);
            for (size_t j = 0; j < n_backends; j++) {
                ggml_cgraph * cgraph_ij = backend_ctx->backend_configs[j].cgraphs[i_slot][i].cgraph_main;
                nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes - 1]);
            }
            success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
        }
        return success;
    }

    ggml_status begin_reduce(const size_t i, const int i_op, bool & pending) {
        pending = false;
        std::vector<ggml_tensor *> nodes;
        nodes.reserve(n_backends);
        for (size_t j = 0; j < n_backends; j++) {
            ggml_cgraph * cgraph_ij = backend_ctx->backend_configs[j].cgraphs[i_slot][i].cgraph_main;
            nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes - 1]);
        }

        // A cross_host_reduce hook needs the LOCAL reduce fully complete, synchronously, before
        // it can run (it reads the locally-reduced total back to host and exchanges it with the
        // peer rank). The async vendor comm_allreduce_begin/_end path leaves the local reduce
        // pending across this call, with no hook here to run cross-host work once it lands, so
        // skip it entirely when a cross-host reducer is installed and always go through the
        // synchronous reduce() path below instead.
        if (backend_ctx->cross_host_reduce == nullptr &&
                backend_ctx->comm_ctx && backend_ctx->comm_allreduce_begin != nullptr &&
                backend_ctx->comm_allreduce_end != nullptr &&
                backend_ctx->comm_allreduce_begin(backend_ctx->comm_ctx, nodes.data(), i_op)) {
            pending = true;
            return GGML_STATUS_SUCCESS;
        }
        return reduce(i);
    }

    ggml_status end_reduce(const int i_op) {
        return backend_ctx->comm_allreduce_end(backend_ctx->comm_ctx, i_op) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }

    // Local (intra-process) reduce for subgraph i, followed by the cross-host reduce (a no-op
    // when no cross_host_reduce hook is installed -- see cross_host_reduce_step()). Every
    // SYNCHRONOUS reduce call site should go through this, not blocking_reduce()/
    // allreduce_fallback() directly, so a cross-host run always gets its chance to contribute
    // the peer rank's partial sum before the next subgraph runs.
    ggml_status reduce(const size_t i) {
        if (!blocking_reduce(i) && allreduce_fallback(i) != GGML_STATUS_SUCCESS) {
            return GGML_STATUS_FAILED;
        }
        return cross_host_reduce_step(i);
    }

    // Cross-host reduce. The local reduce above (blocking_reduce/allreduce_fallback, or the
    // vendor comm_allreduce_begin/_end pair when no cross-host reducer is installed) left every
    // local device holding this rank's partial sum over its own devices; add the peer rank's
    // partial to it so that every device in the WORLD holds the same total before the next
    // subgraph runs.
    //
    // This is done here, at the meta-backend hook, and not as a graph op: the reduce is not a
    // node in this design, the tensor being reduced is the last node of a subgraph that the meta
    // backend materialises separately per device, and there is no single ggml context owning all
    // of the device copies.
    ggml_status cross_host_reduce_step(const size_t i) {
        if (backend_ctx->cross_host_reduce == nullptr) {
            return GGML_STATUS_SUCCESS;
        }

        ggml_cgraph * cgraph_i0 = backend_ctx->backend_configs[0].cgraphs[i_slot][i].cgraph_main;
        ggml_tensor * node0     = cgraph_i0->nodes[cgraph_i0->n_nodes - 1];
        GGML_ASSERT(node0->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(node0));

        const size_t nbytes   = ggml_nbytes(node0);
        const size_t n_values = nbytes / sizeof(float);
        float * staging = backend_ctx->cross_host_staging(nbytes);

        if (n_backends == 1 && (node0->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            // A rank with a single local device whose slice is zero-sized never ran this node,
            // so its buffer holds whatever was there before. With more than one local device
            // allreduce_fallback has already zeroed such a node and the butterfly has left
            // device 0 holding the correct local total; with exactly one there is no local
            // reduce to do that, so contribute an explicit zero rather than garbage. Skewed
            // tensor_splits DO produce zero-sized attention slices, so this is a live path, not
            // a defensive one.
            memset(staging, 0, nbytes);
        } else {
            // After the local reduce every local device holds the same values, so device 0 is
            // as good as any; read it back once.
            ggml_backend_tensor_get_async(backend_ctx->backend_configs[0].backend, node0, staging, 0, nbytes);
            ggml_backend_synchronize(backend_ctx->backend_configs[0].backend);
        }

        if (!backend_ctx->cross_host_reduce(backend_ctx->cross_host_reduce_ud, staging, n_values)) {
            return GGML_STATUS_FAILED;
        }

        // The tensor is logically MIRRORED after the reduce, so a plain set per device is
        // correct.
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cgraph_ij = bcj.cgraphs[i_slot][i].cgraph_main;
            ggml_tensor * node_j    = cgraph_ij->nodes[cgraph_ij->n_nodes - 1];
            GGML_ASSERT(ggml_nbytes(node_j) == nbytes);
            // The tensor is MIRRORED from here on, so every local device gets the total,
            // including one whose own slice was zero-sized: downstream nodes read it.
            ggml_backend_tensor_set_async(bcj.backend, node_j, staging, 0, nbytes);
        }
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
        }
        return GGML_STATUS_SUCCESS;
    }
};

// Opt-in step-loop profile (GGML_META_STEP_STATS=1). The decode path submits
// one graph_compute_async per SUBGRAPH per backend, and subgraphs are delimited
// by AllReduce boundaries -- so a single decode token is many small submissions
// plus many cross-device rendezvous. This splits that wall into prepare /
// submit / reduce so the fragmentation cost can be attributed instead of
// inferred.
struct ggml_backend_meta_step_stats {
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> n_steps{0};
    std::atomic<uint64_t> ns_prepare{0};
    std::atomic<uint64_t> ns_submit{0};
    std::atomic<uint64_t> ns_reduce{0};
};
static ggml_backend_meta_step_stats g_meta_step_stats;

static bool ggml_backend_meta_step_stats_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_META_STEP_STATS");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

static void ggml_backend_meta_step_stats_print() {
    const uint64_t nc = g_meta_step_stats.n_calls.load();
    if (nc == 0) {
        return;
    }
    const uint64_t ns = g_meta_step_stats.n_steps.load();
    const double pr = g_meta_step_stats.ns_prepare.load() / 1e6;
    const double su = g_meta_step_stats.ns_submit.load()  / 1e6;
    const double rd = g_meta_step_stats.ns_reduce.load()  / 1e6;
    fprintf(stderr,
            "ggml-meta step-stats: calls=%" PRIu64 " steps=%" PRIu64 " (%.1f/call) "
            "prepare=%.1f ms (%.1f us/call) submit=%.1f ms (%.1f us/step) reduce=%.1f ms (%.1f us/step) "
            "total=%.1f ms (%.3f ms/call)\n",
            nc, ns, (double) ns / (double) nc,
            pr, pr * 1000.0 / (double) nc,
            su, su * 1000.0 / (double) (ns ? ns : 1),
            rd, rd * 1000.0 / (double) (ns ? ns : 1),
            pr + su + rd, (pr + su + rd) / (double) nc);
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const bool stats = ggml_backend_meta_step_stats_enabled();
    if (stats) {
        static std::once_flag once;
        std::call_once(once, []() { atexit(ggml_backend_meta_step_stats_print); });
    }
    const uint64_t t_prep0 = stats ? ggml_backend_meta_now_ns() : 0;
    ggml_backend_meta_graph_prepare(backend, cgraph, 0);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    ggml_backend_meta_graph_runner runner = { backend_ctx, 0, ggml_backend_meta_n_backends(backend) };
    const size_t n_subgraphs = backend_ctx->graph_states[0].n_subgraphs;
    if (stats) {
        g_meta_step_stats.ns_prepare.fetch_add(ggml_backend_meta_now_ns() - t_prep0);
        g_meta_step_stats.n_calls.fetch_add(1);
        g_meta_step_stats.n_steps.fetch_add(n_subgraphs);
    }
    for (size_t i = 0; i < n_subgraphs; i++) {
        const uint64_t t0 = stats ? ggml_backend_meta_now_ns() : 0;
        const ggml_status status = runner.compute(i);
        if (stats) {
            g_meta_step_stats.ns_submit.fetch_add(ggml_backend_meta_now_ns() - t0);
        }
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        // With a single local device there is nothing to reduce LOCALLY -- but there may still
        // be a peer rank, so this gate must not also guard the cross-host reduce inside
        // reduce()/cross_host_reduce_step(). A world of two ranks with one device each (the
        // loopback tripwire) hits exactly this case.
        if ((runner.n_backends > 1 || backend_ctx->cross_host_reduce != nullptr) && i + 1 < n_subgraphs) {
            const uint64_t t1 = stats ? ggml_backend_meta_now_ns() : 0;
            const ggml_status reduce_status = runner.reduce(i);
            if (stats) {
                g_meta_step_stats.ns_reduce.fetch_add(ggml_backend_meta_now_ns() - t1);
            }
            if (reduce_status != GGML_STATUS_SUCCESS) {
                return reduce_status;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_meta_graph_compute_step_begin(
        ggml_backend_t backend, struct ggml_cgraph * cgraph, size_t i_slot, size_t * n_steps) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    GGML_ASSERT(i_slot < ggml_backend_meta_context::n_graph_slots);
    GGML_ASSERT(n_steps != nullptr);
    if (!ggml_backend_meta_overlap_enabled(backend)) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_meta_graph_prepare(backend, cgraph, i_slot);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    auto & gs = backend_ctx->graph_states[i_slot];
    gs.next_subgraph = 0;
    *n_steps = gs.n_subgraphs;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_meta_graph_compute_step(
        ggml_backend_t backend, size_t i_slot, int i_op, bool * pending, bool * finished) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    GGML_ASSERT(i_slot < ggml_backend_meta_context::n_graph_slots);
    GGML_ASSERT(pending != nullptr);
    GGML_ASSERT(finished != nullptr);

    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    auto & gs = backend_ctx->graph_states[i_slot];
    GGML_ASSERT(gs.next_subgraph < gs.n_subgraphs);

    ggml_backend_meta_graph_runner runner = { backend_ctx, i_slot, ggml_backend_meta_n_backends(backend) };
    const size_t i = gs.next_subgraph++;
    ggml_status status = runner.compute(i);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    *pending = false;
    if (runner.n_backends > 1 && i + 1 < gs.n_subgraphs) {
        status = runner.begin_reduce(i, i_op, *pending);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    *finished = gs.next_subgraph == gs.n_subgraphs;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_meta_graph_compute_step_end(
        ggml_backend_t backend, size_t i_slot, int i_op) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    GGML_ASSERT(i_slot < ggml_backend_meta_context::n_graph_slots);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    ggml_backend_meta_graph_runner runner = { backend_ctx, i_slot, ggml_backend_meta_n_backends(backend) };
    return runner.end_reduce(i_op);
}

enum ggml_status ggml_backend_meta_graph_compute_pair(
        ggml_backend_t backend, struct ggml_cgraph * cgraph_a, struct ggml_cgraph * cgraph_b) {
    if (!ggml_backend_meta_overlap_enabled(backend)) {
        enum ggml_status status = ggml_backend_meta_graph_compute(backend, cgraph_a);
        return status == GGML_STATUS_SUCCESS ? ggml_backend_meta_graph_compute(backend, cgraph_b) : status;
    }

    ggml_backend_meta_graph_prepare(backend, cgraph_a, 0);
    ggml_backend_meta_graph_prepare(backend, cgraph_b, 1);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    const size_t n_subgraphs_a = backend_ctx->graph_states[0].n_subgraphs;
    const size_t n_subgraphs_b = backend_ctx->graph_states[1].n_subgraphs;
    if (n_subgraphs_a != n_subgraphs_b) {
        ggml_backend_meta_graph_runner runner_a = { backend_ctx, 0, n_backends };
        ggml_backend_meta_graph_runner runner_b = { backend_ctx, 1, n_backends };
        for (ggml_backend_meta_graph_runner * runner : { &runner_a, &runner_b }) {
            const size_t n_subgraphs = backend_ctx->graph_states[runner->i_slot].n_subgraphs;
            for (size_t i = 0; i < n_subgraphs; i++) {
                const ggml_status status = runner->compute(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                if (i + 1 < n_subgraphs) {
                    const ggml_status reduce_status = runner->reduce(i);
                    if (reduce_status != GGML_STATUS_SUCCESS) {
                        return reduce_status;
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    ggml_backend_meta_graph_runner runner_a = { backend_ctx, 0, n_backends };
    ggml_backend_meta_graph_runner runner_b = { backend_ctx, 1, n_backends };
    const size_t n_subgraphs = n_subgraphs_a;
    bool pending_b = false;
    for (size_t i = 0; i < n_subgraphs; i++) {
        ggml_status status = runner_a.compute(i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        bool pending_a = false;
        if (i + 1 < n_subgraphs) {
            status = runner_a.begin_reduce(i, 0, pending_a);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        if (i > 0 && pending_b) {
            status = runner_b.end_reduce(1);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            pending_b = false;
        }

        status = runner_b.compute(i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        if (i + 1 < n_subgraphs) {
            status = runner_b.begin_reduce(i, 1, pending_b);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        if (pending_a) {
            status = runner_a.end_reduce(0);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_meta_set_cross_host_reduce(
        ggml_backend_t meta_backend, ggml_backend_meta_cross_host_reduce_t reduce, void * ud) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    backend_ctx->cross_host_reduce    = reduce;
    backend_ctx->cross_host_reduce_ud = ud;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

bool ggml_backend_meta_overlap_enabled(ggml_backend_t meta_backend) {
    if (!ggml_backend_is_meta(meta_backend)) {
        return false;
    }
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    const char * gate = getenv("GGML_META_OVERLAP");
    return gate != nullptr && strcmp(gate, "1") == 0 &&
        backend_ctx->backend_configs.size() == 2 &&
        backend_ctx->comm_allreduce_begin != nullptr && backend_ctx->comm_allreduce_end != nullptr;
}

bool ggml_backend_meta_is_meta(ggml_backend_t backend) {
    return ggml_backend_is_meta(backend);
}

size_t ggml_backend_meta_n_local(ggml_backend_t meta_backend) {
    return ggml_backend_meta_n_backends(meta_backend);
}

size_t ggml_backend_meta_n_world(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    return ggml_backend_meta_dev_n_world(meta_backend->device);
}

size_t ggml_backend_meta_rank_first(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    return ggml_backend_meta_dev_rank_first(meta_backend->device);
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

// MAD-LAB: public wrapper for the static ggml_backend_meta_buffer_simple_tensor()
// above -- see its declaration comment in ggml-backend.h. Kept as a thin wrapper
// rather than un-static'ing the original so the many internal call sites (which
// take/return const/non-const combinations this public signature does not need
// to distinguish) are untouched.
struct ggml_tensor * ggml_backend_meta_get_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    return ggml_backend_meta_buffer_simple_tensor(tensor, index);
}

// MAD-LAB (WP_DFLASH_BORROW_META, dflash-borrow-meta-0912.txt): see the declaration
// comment in ggml-backend.h -- this is a device lookup only, it says nothing about
// whether the found index's simple tensor is the FULL tensor (MIRRORED) or a SHARD
// (any real split axis). Callers are responsible for knowing which case applies to
// the tensor they're calling this on.
int ggml_backend_meta_find_device_index_for_tensor(const struct ggml_tensor * tensor, ggml_backend_dev_t dev) {
    if (tensor == nullptr || tensor->buffer == nullptr || dev == nullptr ||
            !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return -1;
    }
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    for (size_t i = 0; i < n_bufs; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_meta_buffer_simple_buffer(tensor->buffer, i);
        if (simple_buf == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(simple_buf);
        if (buft != nullptr && ggml_backend_buft_get_device(buft) == dev) {
            return (int) i;
        }
    }
    return -1;
}
