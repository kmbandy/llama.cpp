// dram_meter.cpp
//
// Standalone VRAM read-traffic meter for the AMD R9700 (gfx1201) GPU using the
// rocprofiler-sdk *device counting* service (agent-level hardware counters).
//
// Device counting is AGENT-level: the GPU hardware counters accumulate ALL
// activity on the selected GPU agent, regardless of which process issued the
// work. dram_meter therefore:
//
//   1. Initializes HSA + rocprofiler in its own process and sets up a
//      device-counting context bound to the gfx1201 agent only (the box also
//      contains a gfx1030 6900XT which must NOT be selected).
//   2. Takes a baseline counter sample.
//   3. fork()/execvp()'s argv[1..] as a subprocess and waitpid()'s for it. The
//      subprocess does the actual GPU work (via raw KFD); dram_meter does no
//      GPU work itself, it just keeps the counting context alive across the
//      window.
//   4. Takes a second sample. The DELTA across the window is the traffic.
//   5. Reports FETCH_SIZE bytes/MB, L2 hit rate, and raw counter deltas.
//
// RDNA4 / gfx1201 counter naming (IMPORTANT):
//   On RDNA4 the L2 cache block that the original CDNA/RDNA2 code called "TCC"
//   is named "GL2C" (Graphics L2 Cache), and read requests come in THREE sizes
//   (32B, 64B and 128B), not two. So the correct counters are:
//     GL2C_EA_RDREQ        - total EA read requests (32B|64B|128B)
//     GL2C_EA_RDREQ_32B    - 32-byte EA read requests
//     GL2C_EA_RDREQ_64B    - 64-byte EA read requests
//     GL2C_EA_RDREQ_128B   - 128-byte EA read requests
//     GL2C_HIT / GL2C_MISS - L2 hit/miss
//   and the fetch size (matching the SDK's built-in FETCH_SIZE expression) is
//     FETCH_SIZE_bytes = RDREQ_32B*32 + RDREQ_64B*64 + RDREQ_128B*128.
//   (The original 2-size TCC formula RDREQ_32B*32 + (RDREQ-RDREQ_32B)*64 is a
//    fallback only and would mis-weight 128B traffic on RDNA4.)
//
// Adapted from:
//   /opt/rocm/share/rocprofiler-sdk/samples/counter_collection/
//       device_counting_sync_client.cpp
// (the counter_sampler class).

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

namespace
{
// ---------------------------------------------------------------------------
// counter_sampler  (adapted from device_counting_sync_client.cpp)
// ---------------------------------------------------------------------------
class counter_sampler
{
public:
    explicit counter_sampler(rocprofiler_agent_id_t agent);

    std::string decode_record_name(const rocprofiler_counter_record_t& rec) const;

    // Resolve / create the counter profile for a set of counters and stash it in
    // profile_ so it is installed when the context is started. Does NOT start the
    // context or sample. Returns ROCPROFILER_STATUS_SUCCESS on success, or the
    // underlying error (e.g. ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT) so the
    // caller can fall back to a smaller counter set. On success, out is sized to
    // hold one sample's worth of records.
    rocprofiler_status_t prepare(const std::vector<std::string>&            counters,
                                 std::vector<rocprofiler_counter_record_t>& out);

    // Start the device-counting context. profile_ must already be set (via
    // prepare) because rocprofiler invokes the set_profile callback here.
    void start() const { rocprofiler_start_context(ctx_); }

    // Take ONE sample of the currently-active context's accumulated counters.
    // Does NOT start or stop the context. out must already be sized by prepare().
    rocprofiler_status_t sample_now(std::vector<rocprofiler_counter_record_t>& out) const;

    static std::vector<rocprofiler_agent_v0_t> get_available_agents();

    void flush() const { rocprofiler_flush_buffer(buf_); }
    void stop() const { rocprofiler_stop_context(ctx_); }

private:
    rocprofiler_agent_id_t          agent_   = {};
    rocprofiler_context_id_t        ctx_     = {};
    rocprofiler_buffer_id_t         buf_     = {};
    rocprofiler_counter_config_id_t profile_ = {.handle = 0};

    std::map<std::vector<std::string>, rocprofiler_counter_config_id_t> cached_profiles_;
    std::map<uint64_t, uint64_t>                                        profile_sizes_;
    mutable std::map<uint64_t, std::string>                             id_to_name_;

    void set_profile(rocprofiler_context_id_t ctx, rocprofiler_device_counting_agent_cb_t cb) const;

    static size_t get_counter_size(rocprofiler_counter_id_t counter);

    static std::unordered_map<std::string, rocprofiler_counter_id_t> get_supported_counters(
        rocprofiler_agent_id_t agent);
};

counter_sampler::counter_sampler(rocprofiler_agent_id_t agent)
: agent_(agent)
{
    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx_), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_create_buffer(
                         ctx_,
                         4096,
                         2048,
                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                         [](rocprofiler_context_id_t,
                            rocprofiler_buffer_id_t,
                            rocprofiler_record_header_t**,
                            size_t,
                            void*,
                            uint64_t) {},
                         nullptr,
                         &buf_),
                     "buffer creation failed");
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "failure creating callback thread");
    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(buf_, client_thread),
                     "failed to assign thread for buffer");

    ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(
                         ctx_,
                         buf_,
                         agent,
                         [](rocprofiler_context_id_t context_id,
                            rocprofiler_agent_id_t,
                            rocprofiler_device_counting_agent_cb_t set_config,
                            void*                                  user_data) {
                             if(user_data)
                             {
                                 auto* sampler = static_cast<counter_sampler*>(user_data);
                                 sampler->set_profile(context_id, set_config);
                             }
                         },
                         this),
                     "Could not setup device counting service");
}

std::string
counter_sampler::decode_record_name(const rocprofiler_counter_record_t& rec) const
{
    if(id_to_name_.empty())
    {
        auto name_to_id = counter_sampler::get_supported_counters(agent_);
        for(const auto& [name, id] : name_to_id)
            id_to_name_.emplace(id.handle, name);
    }

    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(rec.id, &counter_id);
    auto it = id_to_name_.find(counter_id.handle);
    if(it == id_to_name_.end()) return "UNKNOWN";
    return it->second;
}

rocprofiler_status_t
counter_sampler::prepare(const std::vector<std::string>&            counters,
                         std::vector<rocprofiler_counter_record_t>& out)
{
    auto profile_cached = cached_profiles_.find(counters);
    if(profile_cached == cached_profiles_.end())
    {
        size_t                                expected_size = 0;
        rocprofiler_counter_config_id_t       profile       = {};
        std::vector<rocprofiler_counter_id_t> gpu_counters;
        auto                                  roc_counters = get_supported_counters(agent_);
        for(const auto& counter : counters)
        {
            auto it = roc_counters.find(counter);
            if(it == roc_counters.end())
            {
                std::cerr << "Counter " << counter << " not found on agent\n";
                continue;
            }
            gpu_counters.push_back(it->second);
            expected_size += get_counter_size(it->second);
        }
        if(gpu_counters.empty())
        {
            std::cerr << "No requested counters were available on the agent\n";
            return ROCPROFILER_STATUS_ERROR;
        }
        // Do NOT throw here: a too-large set returns
        // ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT; return it so the caller can
        // fall back to a smaller counter set.
        auto cfg_status = rocprofiler_create_counter_config(
            agent_, gpu_counters.data(), gpu_counters.size(), &profile);
        if(cfg_status != ROCPROFILER_STATUS_SUCCESS) return cfg_status;
        cached_profiles_.emplace(counters, profile);
        profile_sizes_.emplace(profile.handle, expected_size);
        profile_cached = cached_profiles_.find(counters);
    }
    try
    {
        out.resize(profile_sizes_.at(profile_cached->second.handle));
    } catch(const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return ROCPROFILER_STATUS_ERROR;
    }
    // Stash the profile so it is installed via the set_profile callback when the
    // context is started (rocprofiler invokes that callback from start_context).
    profile_ = profile_cached->second;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_sampler::sample_now(std::vector<rocprofiler_counter_record_t>& out) const
{
    // ONLY read the accumulated counters of the already-running context. Does not
    // start or stop the context, so the counting window spans whatever happens
    // between start() and this call.
    size_t out_size = out.size();
    auto   status   = rocprofiler_sample_device_counting_service(
        ctx_, {}, ROCPROFILER_COUNTER_FLAG_NONE, out.data(), &out_size);
    out.resize(out_size);
    return status;
}

std::vector<rocprofiler_agent_v0_t>
counter_sampler::get_available_agents()
{
    std::vector<rocprofiler_agent_v0_t>     agents;
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* rocp_agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*rocp_agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    return agents;
}

void
counter_sampler::set_profile(rocprofiler_context_id_t               ctx,
                             rocprofiler_device_counting_agent_cb_t cb) const
{
    if(profile_.handle != 0) cb(ctx, profile_);
}

size_t
counter_sampler::get_counter_size(rocprofiler_counter_id_t counter)
{
    rocprofiler_counter_info_v1_t info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info)),
                     "Could not query info for counter");
    return info.dimensions_instances_count;
}

std::unordered_map<std::string, rocprofiler_counter_id_t>
counter_sampler::get_supported_counters(rocprofiler_agent_id_t agent)
{
    std::unordered_map<std::string, rocprofiler_counter_id_t> out;
    std::vector<rocprofiler_counter_id_t>                     gpu_counters;

    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             auto* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                                 vec->push_back(counters[i]);
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info for counter");
        out.emplace(info.name, counter);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tool globals + entry points
// ---------------------------------------------------------------------------
constexpr const char* kTargetGfx = "gfx1201";  // R9700 (must NOT pick gfx1030)

rocprofiler_client_finalize_t    g_finalize  = nullptr;
rocprofiler_client_id_t*         g_client_id = nullptr;
std::shared_ptr<counter_sampler> g_sampler   = {};
std::atomic<bool>                g_ready{false};
std::atomic<bool>                g_init_failed{false};

int
tool_init(rocprofiler_client_finalize_t fini_func, void*)
{
    g_finalize = fini_func;

    auto agents = counter_sampler::get_available_agents();
    if(agents.empty())
    {
        std::cerr << "[dram_meter] No GPU agents found\n";
        g_init_failed.store(true);
        return -1;
    }

    // Select the gfx1201 (R9700) agent ONLY. The box also has a gfx1030.
    const rocprofiler_agent_v0_t* chosen = nullptr;
    std::cerr << "[dram_meter] GPU agents:\n";
    for(const auto& a : agents)
    {
        const char* nm = a.name ? a.name : "(null)";
        std::cerr << "[dram_meter]   node_id=" << a.node_id << " name=" << nm
                  << " product=" << (a.product_name ? a.product_name : "(null)") << "\n";
        if(a.name && std::strcmp(a.name, kTargetGfx) == 0 && chosen == nullptr) chosen = &a;
    }

    if(chosen == nullptr)
    {
        std::cerr << "[dram_meter] ERROR: no agent named '" << kTargetGfx << "' found\n";
        g_init_failed.store(true);
        return -1;
    }

    std::cerr << "[dram_meter] Selected agent: " << chosen->name
              << " (node_id=" << chosen->node_id << ")\n";

    try
    {
        g_sampler = std::make_shared<counter_sampler>(chosen->id);
    } catch(const std::exception& e)
    {
        std::cerr << "[dram_meter] Failed to create sampler: " << e.what() << "\n";
        g_init_failed.store(true);
        return -1;
    }

    g_ready.store(true);
    return 0;
}

void
tool_fini(void*)
{
    if(g_sampler)
    {
        g_sampler->stop();
        g_sampler->flush();
        g_sampler.reset();
    }
    g_client_id = nullptr;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    id->name      = "dram_meter";
    g_client_id   = id;

    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;
    std::cerr << "[dram_meter] using rocprofiler-sdk v" << major << "." << minor << "." << patch
              << " (" << runtime_version << "), priority=" << priority << "\n";

    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            nullptr};
    return &cfg;
}

namespace
{
// Counters we want, in RDNA4 (gfx1201) "GL2C" naming. The three per-size read
// counters are required for FETCH_SIZE; HIT/MISS are best-effort (L2 hit rate).
// If a single hardware pass cannot collect the full set, fall back to just the
// three read-size counters needed for FETCH_SIZE.
// Candidate counter sets, tried largest-first. The GL2C hardware block has a
// limited number of simultaneous counters per pass, so the full 6-counter set
// may exceed the hardware limit on gfx1201; we degrade gracefully to the three
// read-size counters that FETCH_SIZE strictly requires.
const std::vector<std::vector<std::string>> kCounterTiers = {
    {"GL2C_EA_RDREQ_32B",
     "GL2C_EA_RDREQ_64B",
     "GL2C_EA_RDREQ_128B",
     "GL2C_HIT",
     "GL2C_MISS"},
    {"GL2C_EA_RDREQ_32B", "GL2C_EA_RDREQ_64B", "GL2C_EA_RDREQ_128B"},
};

// Sum all instance records (already sampled into `records`) for each counter
// name into a name->value map.
void
sum_records(const std::vector<rocprofiler_counter_record_t>& records,
            std::map<std::string, double>&                    out)
{
    out.clear();
    for(const auto& rec : records)
    {
        auto name = g_sampler->decode_record_name(rec);
        out[name] += rec.counter_value;
    }
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <command> [args...]\n"
                  << "  Measures VRAM read traffic on " << kTargetGfx
                  << " across the lifetime of <command>.\n";
        return 2;
    }

    // HSA must be initialized in THIS process for device sampling to work
    // (rocprofiler device counting requires the HSA runtime loaded).
    if(hsa_init() != HSA_STATUS_SUCCESS)
    {
        std::cerr << "[dram_meter] ERROR: hsa_init() failed\n";
        return 1;
    }

    // Force rocprofiler to configure using our in-executable rocprofiler_configure.
    if(int status = 0; rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS &&
                       status == 0)
    {
        auto rc = rocprofiler_force_configure(&rocprofiler_configure);
        if(rc != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "[dram_meter] ERROR: rocprofiler_force_configure failed: "
                      << rocprofiler_get_status_string(rc) << "\n";
            return 1;
        }
    }

    if(g_init_failed.load() || !g_ready.load() || !g_sampler)
    {
        std::cerr << "[dram_meter] ERROR: tool initialization failed (no usable "
                  << kTargetGfx << " agent / sampler)\n";
        return 1;
    }

    // Probe up front which counter set the agent can configure, picking the
    // largest viable tier. prepare() resolves/creates the profile and stashes it
    // in profile_ (installed via the set_profile callback when the context is
    // started) but does NOT start the context or sample. We must fix the set
    // BEFORE the measurement window so the baseline and after samples match.
    std::vector<std::string>                  counters;
    std::vector<rocprofiler_counter_record_t> records;  // sized by prepare()
    bool                                      have_hitmiss = false;

    for(const auto& tier : kCounterTiers)
    {
        if(g_sampler->prepare(tier, records) == ROCPROFILER_STATUS_SUCCESS)
        {
            counters     = tier;
            have_hitmiss = (std::find(tier.begin(), tier.end(), "GL2C_HIT") != tier.end() &&
                            std::find(tier.begin(), tier.end(), "GL2C_MISS") != tier.end());
            break;
        }
        std::cerr << "[dram_meter] NOTE: counter set of size " << tier.size()
                  << " could not be configured in one pass; trying a smaller set.\n";
    }

    if(counters.empty())
    {
        std::cerr << "[dram_meter] ERROR: no viable counter set configured on the agent\n";
        return 1;
    }
    if(!have_hitmiss)
        std::cerr << "[dram_meter] NOTE: L2 hit/miss not collected in this pass; "
                     "FETCH_SIZE still reported, L2 hit rate unavailable.\n";

    // START the device-counting context ONCE, BEFORE fork(). The counting window
    // must span the entire subprocess lifetime so the GPU counters accumulate the
    // subprocess's traffic. (The set_profile callback fires here, installing the
    // profile stashed by prepare().)
    g_sampler->start();

    // Baseline sample: read the accumulated counters at the very start of the
    // window (before any subprocess work). We diff this against the after-sample,
    // which is robust whether the device counters reset on context-start or
    // persist across it.
    std::map<std::string, double> baseline;
    {
        if(g_sampler->sample_now(records) != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "[dram_meter] ERROR: baseline sample failed\n";
            g_sampler->stop();
            return 1;
        }
        sum_records(records, baseline);
    }

    // Fork/exec the subprocess and wait for it. dram_meter does no GPU work;
    // it just keeps the counting context active across this window so the agent
    // hardware counters accumulate the subprocess's DRAM traffic.
    pid_t pid = fork();
    if(pid < 0)
    {
        std::cerr << "[dram_meter] ERROR: fork failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    if(pid == 0)
    {
        execvp(argv[1], &argv[1]);
        std::cerr << "[dram_meter] ERROR: execvp('" << argv[1]
                  << "') failed: " << std::strerror(errno) << "\n";
        _exit(127);
    }

    int   wstatus     = 0;
    pid_t waited      = waitpid(pid, &wstatus, 0);
    int   child_rc    = -1;
    bool  child_clean = false;
    if(waited == pid)
    {
        if(WIFEXITED(wstatus))
        {
            child_rc    = WEXITSTATUS(wstatus);
            child_clean = true;
        }
        else if(WIFSIGNALED(wstatus))
        {
            std::cerr << "[dram_meter] subprocess terminated by signal "
                      << WTERMSIG(wstatus) << "\n";
        }
    }

    // After sample: read the accumulated counters now that the subprocess has
    // exited but the context is still active, then stop the context. The delta
    // (after - baseline) across the window is the traffic.
    std::map<std::string, double> after;
    {
        auto status = g_sampler->sample_now(records);
        g_sampler->stop();
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "[dram_meter] ERROR: post-subprocess sample failed\n";
            return 1;
        }
        sum_records(records, after);
    }

    auto delta = [&](const char* k) -> double {
        double a = after.count(k) ? after.at(k) : 0.0;
        double b = baseline.count(k) ? baseline.at(k) : 0.0;
        double d = a - b;
        return d < 0.0 ? 0.0 : d;  // counters are monotonic; guard wraparound/reset
    };

    double rdreq_32b  = delta("GL2C_EA_RDREQ_32B");
    double rdreq_64b  = delta("GL2C_EA_RDREQ_64B");
    double rdreq_128b = delta("GL2C_EA_RDREQ_128B");
    // GL2C_EA_RDREQ (total, all sizes) is only present in the full pass.
    bool   have_total = (after.count("GL2C_EA_RDREQ") != 0);
    double rdreq      = have_total ? delta("GL2C_EA_RDREQ") : (rdreq_32b + rdreq_64b + rdreq_128b);

    // FETCH_SIZE (bytes) = #32B*32 + #64B*64 + #128B*128, matching the SDK's
    // built-in FETCH_SIZE expression for gfx1201 (which then divides by 1024).
    double fetch_bytes =
        rdreq_32b * 32.0 + rdreq_64b * 64.0 + rdreq_128b * 128.0;
    double fetch_mb = fetch_bytes / (1024.0 * 1024.0);

    std::cout << "==== dram_meter results ====\n";
    std::cout << "agent              = " << kTargetGfx << "\n";
    std::cout << "command            =";
    for(int i = 1; i < argc; ++i) std::cout << ' ' << argv[i];
    std::cout << "\n";
    if(child_clean)
        std::cout << "subprocess_exit    = " << child_rc << "\n";
    else
        std::cout << "subprocess_exit    = (abnormal)\n";
    std::cout << "counters_collected =";
    for(const auto& c : counters) std::cout << ' ' << c;
    std::cout << "\n";

    std::cout << "GL2C_EA_RDREQ      = " << static_cast<long long>(rdreq)
              << (have_total ? "" : " (derived = 32B+64B+128B)") << "\n";
    std::cout << "GL2C_EA_RDREQ_32B  = " << static_cast<long long>(rdreq_32b) << "\n";
    std::cout << "GL2C_EA_RDREQ_64B  = " << static_cast<long long>(rdreq_64b) << "\n";
    std::cout << "GL2C_EA_RDREQ_128B = " << static_cast<long long>(rdreq_128b) << "\n";

    if(have_hitmiss)
    {
        double hit  = delta("GL2C_HIT");
        double miss = delta("GL2C_MISS");
        std::cout << "GL2C_HIT           = " << static_cast<long long>(hit) << "\n";
        std::cout << "GL2C_MISS          = " << static_cast<long long>(miss) << "\n";
        double denom = hit + miss;
        if(denom > 0.0)
            std::cout << "L2_hit_rate        = " << (hit / denom) << "\n";
        else
            std::cout << "L2_hit_rate        = n/a (no hit/miss activity)\n";
    }
    else
    {
        std::cout << "GL2C_HIT           = n/a (not collected in this pass)\n";
        std::cout << "GL2C_MISS          = n/a (not collected in this pass)\n";
        std::cout << "L2_hit_rate        = n/a\n";
    }

    std::cout << "FETCH_SIZE_bytes   = " << static_cast<long long>(fetch_bytes) << "\n";
    std::cout << "FETCH_SIZE_MB      = " << fetch_mb << "\n";
    std::cout << "============================\n";
    std::cout << std::flush;

    if(g_sampler)
    {
        g_sampler->stop();
        g_sampler->flush();
    }
    if(g_client_id && g_finalize) g_finalize(*g_client_id);

    hsa_shut_down();
    return child_clean ? 0 : 1;
}
