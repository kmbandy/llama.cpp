#pragma once

// Thermal governor for the NVMe expert read path.
//
// Pure logic: no ggml / GPU dependencies, so tools/wp-expert-worker/test-wp-thermal.cpp
// can build it standalone. The worker wires it in at two points: a 1 Hz sensor
// thread (WpThermalSensor in wp-expert-worker.cpp) feeds Controller::on_sample(), and
// ExpertSlotPool::read_page_range calls gate() before each pread.
//
// States (thresholds T1 < T2 < T3, degrees C, from the drive's temp1_max):
//   COOL  temp < T1          gate is a no-op (one relaxed atomic load)
//   WARM  temp >= T1         speculative reads block; demand reads take a token bucket
//   HOT   temp >= T2         as WARM, and the bucket cap is cut by AIMD every tick
//   PAUSE temp >= T3         all gated reads block until temp <= T2; if the pause lasts
//                            more than pause_release_s, demand reads resume at the floor
// Exit hysteresis: WARM->COOL and HOT->WARM need temp below the threshold minus
// hysteresis_c. PAUSE->HOT needs temp <= T2.
//
// The gate only delays. It never skips, reorders, or fails a read.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(__linux__)
#    include <sys/stat.h>
#    include <sys/sysmacros.h>
#endif

namespace wp_thermal {

namespace fs = std::filesystem;

enum class State : uint8_t { COOL = 0, WARM = 1, HOT = 2, PAUSE = 3 };

inline const char * state_name(State s) {
    switch (s) {
        case State::COOL:  return "COOL";
        case State::WARM:  return "WARM";
        case State::HOT:   return "HOT";
        case State::PAUSE: return "PAUSE";
    }
    return "?";
}

// How the gate treats a read. The caller maps WpReadOrigin onto this.
enum class ReadClass : uint8_t {
    EXEMPT,  // never gated (KV-park / seed landing)
    SPEC,    // speculative / prefetch / landing: held whenever the gate is active
    DEMAND,  // request-driven: token bucket while WARM/HOT, held while PAUSE
};

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

struct Thresholds {
    double t[3] = { 78.0, 80.0, 85.0 };   // T1, T2, T3 in degrees C
    std::string source;                    // provenance, for the startup log line
    std::string warning;                   // non-empty when an env value was rejected
};

// Parses "a,b,c" into three doubles. Returns false on any other shape.
inline bool parse_triple(const char * s, double out[3]) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    std::string str(s);
    std::stringstream ss(str);
    std::string tok;
    int n = 0;
    while (std::getline(ss, tok, ',')) {
        if (n >= 3) {
            return false;
        }
        char * end = nullptr;
        const double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str() || *end != '\0') {
            return false;
        }
        out[n++] = v;
    }
    return n == 3;
}

inline bool strictly_ascending(const double t[3]) {
    return t[0] > 0.0 && t[0] < t[1] && t[1] < t[2] && t[2] < 200.0;
}

// WP_THERMAL_C wins outright. Otherwise WP_THERMAL_PCT (default 87,89,95) is applied
// to temp1_max (millidegrees). When temp1_max is missing or 0 the fixed 78,80,85 is used.
inline Thresholds derive_thresholds(const char * pct_env, const char * c_env, int64_t max_mdeg) {
    Thresholds th;
    double v[3] = { 0.0, 0.0, 0.0 };
    if (c_env != nullptr && *c_env != '\0') {
        if (parse_triple(c_env, v) && strictly_ascending(v)) {
            std::copy(v, v + 3, th.t);
            th.source = "WP_THERMAL_C";
            return th;
        }
        th.warning = std::string("WP_THERMAL_C='") + c_env + "' rejected (want a,b,c ascending); ";
    }
    double pct[3] = { 87.0, 89.0, 95.0 };
    if (pct_env != nullptr && *pct_env != '\0') {
        if (parse_triple(pct_env, v) && strictly_ascending(v)) {
            std::copy(v, v + 3, pct);
        } else {
            th.warning += std::string("WP_THERMAL_PCT='") + pct_env + "' rejected; using 87,89,95; ";
        }
    }
    if (max_mdeg > 0) {
        for (int i = 0; i < 3; ++i) {
            th.t[i] = (double) max_mdeg / 1000.0 * pct[i] / 100.0;
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%g%%,%g%%,%g%% of temp1_max=%lld mdegC",
                      pct[0], pct[1], pct[2], (long long) max_mdeg);
        th.source = buf;
    } else {
        th.t[0] = 78.0;
        th.t[1] = 80.0;
        th.t[2] = 85.0;
        th.source = "fixed 78,80,85 (temp1_max missing or 0)";
    }
    return th;
}

// ---------------------------------------------------------------------------
// Sensor discovery
// ---------------------------------------------------------------------------

struct SensorPaths {
    std::string temp_input;   // millidegrees C
    std::string temp_max;     // millidegrees C; may name a missing file
    std::string source;       // how the sensor was found, for the log
};

inline bool read_i64_file(const std::string & path, int64_t & out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    long long v = 0;
    if (!(in >> v)) {
        return false;
    }
    out = (int64_t) v;
    return true;
}

// Override via WP_THERMAL_HWMON: a directory means <dir>/temp1_input and
// <dir>/temp1_max; a regular file is used as temp1_input directly (fake sensor for tests).
inline bool sensor_from_override(const std::string & path, SensorPaths & out) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        out.temp_input = (fs::path(path) / "temp1_input").string();
        out.temp_max   = (fs::path(path) / "temp1_max").string();
        out.source     = "WP_THERMAL_HWMON=" + path;
        return true;
    }
    if (fs::is_regular_file(path, ec)) {
        out.temp_input = path;
        out.temp_max.clear();
        out.source     = "WP_THERMAL_HWMON=" + path + " (file)";
        return true;
    }
    return false;
}

inline bool is_nvme_controller_name(const std::string & name) {
    if (name.size() < 5 || name.compare(0, 4, "nvme") != 0) {
        return false;
    }
    for (size_t i = 4; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
    }
    return true;
}

// Collects /sys/class/hwmon/hwmonN dirs whose `name` is "nvme".
inline std::vector<fs::path> nvme_hwmon_dirs(const std::string & sys_root) {
    std::vector<fs::path> out;
    std::error_code ec;
    const fs::path cls = fs::path(sys_root) / "class" / "hwmon";
    for (fs::directory_iterator it(cls, ec), end; !ec && it != end; it.increment(ec)) {
        std::string name;
        std::ifstream in(it->path() / "name");
        if (in && std::getline(in, name)) {
            while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) {
                name.pop_back();
            }
            if (name == "nvme") {
                out.push_back(it->path());
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline bool hwmon_sensor(const fs::path & hwmon, SensorPaths & out) {
    out.temp_input = (hwmon / "temp1_input").string();
    out.temp_max   = (hwmon / "temp1_max").string();
    out.source     = hwmon.string();
    return true;
}

inline bool discover_by_devnum(unsigned maj, unsigned min, const std::string & sys_root,
                               SensorPaths & out, std::string & why);

// Discovers the NVMe temp sensor that backs `blob_path`.
// stat(blob) -> st_dev -> /sys/dev/block/MAJ:MIN (symlink resolved; a partition walks to
// its parent disk) -> nvme controller dir (nvmeN) -> hwmon whose `name` is "nvme" and
// whose `device` resolves to that controller.
// When st_dev has major 0 (btrfs and similar anonymous devices) the block device cannot be
// mapped; the fallback is used only if exactly one nvme hwmon exists.
// sys_root is "/sys" in production and a fake tree in tests.
inline bool discover_nvme_sensor(const std::string & blob_path, const std::string & sys_root,
                                 SensorPaths & out, std::string & why) {
#if defined(__linux__)
    struct stat st;
    if (::stat(blob_path.c_str(), &st) != 0) {
        why = "stat(" + blob_path + ") failed: " + std::strerror(errno);
        return false;
    }
    const unsigned maj = major(st.st_dev);
    const unsigned min = minor(st.st_dev);
    if (maj == 0) {
        const std::vector<fs::path> hw = nvme_hwmon_dirs(sys_root);
        if (hw.size() == 1) {
            hwmon_sensor(hw[0], out);
            out.source += " (anonymous st_dev, sole nvme hwmon)";
            return true;
        }
        why = "st_dev of " + blob_path + " is anonymous (major 0) and " +
              std::to_string(hw.size()) + " nvme hwmon sensors exist (need exactly 1)";
        return false;
    }
    return discover_by_devnum(maj, min, sys_root, out, why);
#else
    (void) blob_path;
    (void) sys_root;
    (void) out;
    why = "thermal governor requires Linux sysfs";
    return false;
#endif
}

// Block-device step of discover_nvme_sensor: MAJ:MIN -> /sys/dev/block -> disk -> controller -> hwmon.
inline bool discover_by_devnum(unsigned maj, unsigned min, const std::string & sys_root,
                               SensorPaths & out, std::string & why) {
    std::error_code ec;
    const fs::path link = fs::path(sys_root) / "dev" / "block" /
                          (std::to_string(maj) + ":" + std::to_string(min));
    fs::path dev = fs::canonical(link, ec);
    if (ec) {
        why = "cannot resolve " + link.string() + ": " + ec.message();
        return false;
    }
    if (fs::exists(dev / "partition", ec)) {
        dev = dev.parent_path();
    }
    fs::path ctrl = dev;
    while (ctrl.has_relative_path() && !is_nvme_controller_name(ctrl.filename().string())) {
        ctrl = ctrl.parent_path();
    }
    if (!ctrl.has_relative_path()) {
        why = "no nvme controller above " + dev.string();
        return false;
    }
    const fs::path ctrl_real = fs::canonical(ctrl, ec);
    if (ec) {
        why = "cannot resolve controller " + ctrl.string();
        return false;
    }
    for (const fs::path & hw : nvme_hwmon_dirs(sys_root)) {
        const fs::path dev_link = fs::canonical(hw / "device", ec);
        if (!ec && dev_link == ctrl_real) {
            hwmon_sensor(hw, out);
            out.source += " (via " + ctrl_real.string() + ")";
            return true;
        }
    }
    why = "no nvme hwmon sensor under controller " + ctrl_real.string();
    return false;
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

struct Config {
    double t1 = 78.0;             // degrees C
    double t2 = 80.0;
    double t3 = 85.0;
    double floor_bps = 200e6;     // WP_THERMAL_FLOOR_MBPS, bytes/s
    double pause_release_s = 45.0;
    double hysteresis_c = 1.0;
    double md_factor = 0.8;       // HOT: cap *= md_factor per tick
    double ai_step_min_bps = 50e6;
    double bucket_depth_s = 0.25; // bucket depth = cap * bucket_depth_s
    double slice_s = 0.005;       // sleep slice for held reads (clamped to [1 ms, 10 ms])
};

struct Admission {
    bool   pass;       // true: proceed now
    double sleep_s;    // when !pass: sleep this long, then ask again
};

struct Transition {
    bool  changed = false;
    State from = State::COOL;
    State to   = State::COOL;
};

struct Snapshot {
    double   temp_c = 0.0;
    State    state = State::COOL;
    double   cap_bps = 0.0;
    uint64_t demand_throttled_ns = 0;
    uint64_t n_pauses = 0;
    uint64_t n_spec_held = 0;
};

class Controller {
public:
    explicit Controller(const Config & cfg) : cfg_(cfg) {
        cap_bps_ = cfg_.floor_bps;
    }
    Controller(const Controller &) = delete;
    Controller & operator=(const Controller &) = delete;

    const Config & config() const { return cfg_; }

    // Hot-path state read: the gate's COOL check.
    State state() const noexcept { return state_.load(std::memory_order_relaxed); }

    // Bytes that went through the gate, for measured throughput. Called on every gated read.
    void account_read(uint64_t bytes) noexcept {
        bytes_read_.fetch_add(bytes, std::memory_order_relaxed);
    }

    // Sensor thread only. Called once per second with the drive temperature in millidegrees.
    // now_s is a monotonic clock in seconds. Returns the state transition, if any.
    Transition on_sample(int64_t temp_mdeg, double now_s) {
        std::lock_guard<std::mutex> lk(mu_);
        const double temp_c = (double) temp_mdeg / 1000.0;
        temp_c_ = temp_c;

        const uint64_t total = bytes_read_.load(std::memory_order_relaxed);
        ring_[ring_idx_] = total - last_bytes_;
        ring_idx_ = (ring_idx_ + 1) % kRing;
        last_bytes_ = total;
        if (ticks_ < kRing) {
            ++ticks_;
        }
        double sum = 0.0;
        for (size_t i = 0; i < ticks_; ++i) {
            sum += (double) ring_[i];
        }
        const double measured_bps = sum / (double) std::max<size_t>(ticks_, 1);

        const State prev = state_.load(std::memory_order_relaxed);
        const State next = next_state(prev, temp_c);

        if (prev == State::COOL && next != State::COOL) {
            cap_bps_ = std::max(cfg_.floor_bps, measured_bps);
        } else if (temp_c >= cfg_.t2) {
            cap_bps_ = std::max(cfg_.floor_bps, cap_bps_ * cfg_.md_factor);
        } else if (next == State::WARM) {
            cap_bps_ += std::max(cfg_.floor_bps / 2.0, cfg_.ai_step_min_bps);
        }

        Transition tr;
        tr.from = prev;
        tr.to = next;
        tr.changed = next != prev;
        if (tr.changed) {
            state_.store(next, std::memory_order_relaxed);
            if (next == State::PAUSE) {
                ++n_pauses_;
                pause_since_s_ = now_s;
                pause_release_ = false;
            }
            if (prev == State::PAUSE && next != State::PAUSE) {
                pause_release_ = false;
            }
        }
        if (next == State::PAUSE && !pause_release_ &&
                now_s - pause_since_s_ > cfg_.pause_release_s) {
            pause_release_ = true;
        }
        return tr;
    }

    // Any thread. Decides whether a read may proceed now. See the header comment for the rules.
    // A demand admit that passes has already debited the bucket.
    Admission admit(ReadClass cls, uint64_t bytes, double now_s) {
        if (cls == ReadClass::EXEMPT) {
            return { true, 0.0 };
        }
        std::lock_guard<std::mutex> lk(mu_);
        const State s = state_.load(std::memory_order_relaxed);
        if (s == State::COOL) {
            return { true, 0.0 };
        }
        if (cls == ReadClass::SPEC) {
            return { false, slice() };
        }
        if (s == State::PAUSE && !pause_release_) {
            return { false, slice() };
        }
        const double cap = s == State::PAUSE ? std::max(cfg_.floor_bps, 1.0)
                                             : std::max(cap_bps_, 1.0);
        const double depth = cap * cfg_.bucket_depth_s;
        if (!bucket_init_) {
            bucket_init_ = true;
            tokens_ = depth;
            last_refill_s_ = now_s;
        }
        if (now_s > last_refill_s_) {
            tokens_ = std::min(depth, tokens_ + cap * (now_s - last_refill_s_));
            last_refill_s_ = now_s;
        }
        tokens_ = std::min(tokens_, depth);
        // A read larger than the bucket waits for a full bucket and then goes into debt.
        // Waiting on min(bytes, depth) keeps every read admissible, so nothing deadlocks.
        const double need = std::min((double) bytes, depth);
        if (tokens_ >= need) {
            tokens_ -= (double) bytes;
            return { true, 0.0 };
        }
        const double wait = (need - tokens_) / cap;
        return { false, std::min(std::max(wait, 0.001), kMaxSliceS) };
    }

    void note_spec_held() noexcept { n_spec_held_.fetch_add(1, std::memory_order_relaxed); }
    void note_demand_throttled(uint64_t ns) noexcept {
        demand_throttled_ns_.fetch_add(ns, std::memory_order_relaxed);
    }

    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        Snapshot s;
        s.temp_c = temp_c_;
        s.state = state_.load(std::memory_order_relaxed);
        s.cap_bps = cap_bps_;
        s.demand_throttled_ns = demand_throttled_ns_.load(std::memory_order_relaxed);
        s.n_pauses = n_pauses_;
        s.n_spec_held = n_spec_held_.load(std::memory_order_relaxed);
        return s;
    }

private:
    static constexpr size_t kRing = 10;               // 10 one-second buckets
    static constexpr double kMaxSliceS = 0.010;

    double slice() const { return std::min(std::max(cfg_.slice_s, 0.001), kMaxSliceS); }

    State next_state(State prev, double temp_c) const {
        const double h = cfg_.hysteresis_c;
        if (temp_c >= cfg_.t3) {
            return State::PAUSE;
        }
        if (prev == State::PAUSE) {
            return temp_c <= cfg_.t2 ? State::HOT : State::PAUSE;
        }
        if (temp_c >= cfg_.t2) {
            return State::HOT;
        }
        if (prev == State::HOT) {
            return temp_c < cfg_.t2 - h ? State::WARM : State::HOT;
        }
        if (temp_c >= cfg_.t1) {
            return State::WARM;
        }
        if (prev == State::WARM) {
            return temp_c < cfg_.t1 - h ? State::COOL : State::WARM;
        }
        return State::COOL;
    }

    Config cfg_;
    mutable std::mutex mu_;
    std::atomic<State> state_{ State::COOL };
    std::atomic<uint64_t> bytes_read_{ 0 };
    std::atomic<uint64_t> demand_throttled_ns_{ 0 };
    std::atomic<uint64_t> n_spec_held_{ 0 };

    // Guarded by mu_.
    double   temp_c_ = 0.0;
    double   cap_bps_ = 0.0;
    uint64_t n_pauses_ = 0;
    double   pause_since_s_ = 0.0;
    bool     pause_release_ = false;
    uint64_t ring_[kRing] = {};
    size_t   ring_idx_ = 0;
    size_t   ticks_ = 0;
    uint64_t last_bytes_ = 0;
    bool     bucket_init_ = false;
    double   tokens_ = 0.0;
    double   last_refill_s_ = 0.0;
};

// Gate one read. Blocks the calling thread as needed. The COOL path is one relaxed load.
// now() returns monotonic seconds. sleep_for(seconds) sleeps. Both are injected for tests.
template <class Now, class Sleep>
inline void gate(Controller & c, ReadClass cls, uint64_t bytes, Now && now, Sleep && sleep_for) {
    if (c.state() == State::COOL) {
        return;
    }
    const double t0 = now();
    bool held = false;
    for (;;) {
        const Admission a = c.admit(cls, bytes, now());
        if (a.pass) {
            break;
        }
        if (cls == ReadClass::SPEC && !held) {
            held = true;
            c.note_spec_held();
        }
        sleep_for(a.sleep_s);
    }
    if (cls == ReadClass::DEMAND) {
        const double waited = now() - t0;
        if (waited > 0.0) {
            c.note_demand_throttled((uint64_t) (waited * 1e9));
        }
    }
}

inline double steady_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// "wp thermal: STATE WARM->HOT temp=80.5 T=(78.2,80.0,85.4) cap=350 MB/s"
inline std::string format_transition(const Transition & tr, double temp_c, const double t[3],
                                     double cap_bps) {
    char buf[224];
    std::snprintf(buf, sizeof(buf), "wp thermal: STATE %s->%s temp=%.1f T=(%.1f,%.1f,%.1f) cap=%llu MB/s",
                  state_name(tr.from), state_name(tr.to), temp_c, t[0], t[1], t[2],
                  (unsigned long long) (cap_bps / 1e6));
    return buf;
}

}  // namespace wp_thermal
