// Standalone unit test for wp-thermal.h (no ggml, no GPU).
// Build: g++ -std=c++17 -O1 -pthread -Wall -Wextra -o test-wp-thermal test-wp-thermal.cpp

#include "wp-thermal.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;
using namespace wp_thermal;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) {                                                                 \
            ++g_pass;                                                               \
        } else {                                                                    \
            ++g_fail;                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double) (a) - (double) (b)) <= (tol))

static const double MB = 1e6;

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------
static void test_thresholds() {
    // Main's drive: temp1_max 89850 mdegC.
    Thresholds th = derive_thresholds(nullptr, nullptr, 89850);
    CHECK_NEAR(th.t[0], 78.2, 0.05);
    CHECK_NEAR(th.t[1], 80.0, 0.05);
    CHECK_NEAR(th.t[2], 85.4, 0.05);
    CHECK(th.warning.empty());

    // Explicit pct env applied to the same max.
    th = derive_thresholds("80,85,90", nullptr, 89850);
    CHECK_NEAR(th.t[0], 71.88, 0.01);
    CHECK_NEAR(th.t[2], 80.865, 0.01);

    // Missing or zero temp1_max -> fixed 78,80,85.
    th = derive_thresholds(nullptr, nullptr, 0);
    CHECK_NEAR(th.t[0], 78.0, 1e-9);
    CHECK_NEAR(th.t[1], 80.0, 1e-9);
    CHECK_NEAR(th.t[2], 85.0, 1e-9);
    CHECK(th.source.find("fixed") != std::string::npos);

    // WP_THERMAL_C wins over pct and max.
    th = derive_thresholds("80,85,90", "70,75,80", 89850);
    CHECK_NEAR(th.t[0], 70.0, 1e-9);
    CHECK_NEAR(th.t[1], 75.0, 1e-9);
    CHECK_NEAR(th.t[2], 80.0, 1e-9);
    CHECK(th.source == "WP_THERMAL_C");

    // Malformed WP_THERMAL_C is rejected with a warning; pct path still applies.
    th = derive_thresholds(nullptr, "70,75", 89850);
    CHECK(!th.warning.empty());
    CHECK_NEAR(th.t[0], 78.2, 0.05);

    // Non-ascending WP_THERMAL_C is rejected.
    th = derive_thresholds(nullptr, "80,75,85", 0);
    CHECK(!th.warning.empty());
    CHECK_NEAR(th.t[0], 78.0, 1e-9);

    // Malformed WP_THERMAL_PCT falls back to the default 87,89,95.
    th = derive_thresholds("abc", nullptr, 89850);
    CHECK(!th.warning.empty());
    CHECK_NEAR(th.t[0], 78.2, 0.05);
}

// ---------------------------------------------------------------------------
// Sensor reading and override
// ---------------------------------------------------------------------------
static void write_file(const fs::path & p, const std::string & s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << s;
}

static void test_sensor_files(const fs::path & root) {
    const fs::path f = root / "fake_temp";
    write_file(f, "84250\n");
    int64_t v = 0;
    CHECK(read_i64_file(f.string(), v));
    CHECK(v == 84250);
    CHECK(!read_i64_file((root / "missing").string(), v));

    // WP_THERMAL_HWMON as a regular file: used as temp1_input directly.
    SensorPaths sp;
    CHECK(sensor_from_override(f.string(), sp));
    CHECK(sp.temp_input == f.string());
    CHECK(sp.temp_max.empty());

    // WP_THERMAL_HWMON as a directory: temp1_input and temp1_max inside it.
    const fs::path d = root / "hwmon_dir";
    write_file(d / "temp1_input", "50000\n");
    write_file(d / "temp1_max", "89850\n");
    CHECK(sensor_from_override(d.string(), sp));
    CHECK(sp.temp_input == (d / "temp1_input").string());
    CHECK(sp.temp_max == (d / "temp1_max").string());
    int64_t mx = 0;
    CHECK(read_i64_file(sp.temp_max, mx) && mx == 89850);

    CHECK(!sensor_from_override((root / "nope").string(), sp));
}

// Fake /sys tree: block device 259:0 -> partition nvme0n1p2 under nvme0n1 under nvme0,
// and one nvme hwmon whose device points at the controller, plus one unrelated hwmon.
static void build_fake_sysfs(const fs::path & sys) {
    const fs::path ctrl = sys / "devices/pci0000:00/0000:03/nvme/nvme0";
    const fs::path disk = ctrl / "nvme0n1";
    const fs::path part = disk / "nvme0n1p2";
    fs::create_directories(part);
    write_file(part / "partition", "2\n");
    fs::create_directories(sys / "dev/block");
    fs::create_directory_symlink(part, sys / "dev/block/259:2");

    // Whole-disk devnum 259:0 resolves to the disk dir.
    fs::create_directory_symlink(disk, sys / "dev/block/259:0");

    const fs::path hw = sys / "class/hwmon/hwmon5";
    fs::create_directories(hw);
    write_file(hw / "name", "nvme\n");
    write_file(hw / "temp1_input", "57850\n");
    write_file(hw / "temp1_max", "89850\n");
    fs::create_directory_symlink(ctrl, hw / "device");

    const fs::path other = sys / "class/hwmon/hwmon2";
    fs::create_directories(other);
    write_file(other / "name", "k10temp\n");
    fs::create_directory_symlink(sys / "devices/pci0000:00/0000:00:18.3", other / "device");
}

static void test_discovery_fake_tree(const fs::path & root) {
    const fs::path sys = root / "sys";
    build_fake_sysfs(sys);
    SensorPaths sp;
    std::string why;
    // Partition devnum walks up to the disk, then to the controller.
    CHECK(discover_by_devnum(259, 2, sys.string(), sp, why));
    CHECK(sp.temp_input == (sys / "class/hwmon/hwmon5/temp1_input").string());
    CHECK(sp.temp_max == (sys / "class/hwmon/hwmon5/temp1_max").string());
    // Whole-disk devnum resolves too.
    CHECK(discover_by_devnum(259, 0, sys.string(), sp, why));
    // Unknown devnum fails with a reason.
    CHECK(!discover_by_devnum(8, 0, sys.string(), sp, why));
    CHECK(!why.empty());

    // Anonymous st_dev fallback: exactly one nvme hwmon -> used.
    CHECK(nvme_hwmon_dirs(sys.string()).size() == 1);
    // A second nvme hwmon makes the anonymous-st_dev fallback refuse (2 != 1).
    const fs::path hw2 = sys / "class/hwmon/hwmon7";
    fs::create_directories(hw2);
    write_file(hw2 / "name", "nvme\n");
    CHECK(nvme_hwmon_dirs(sys.string()).size() == 2);
}

// Real sysfs on this host, through a real blob path. Machine-dependent: report, and check
// only the invariants (a sensor file that exists and readable temperature).
static void test_discovery_real(const std::string & blob) {
    SensorPaths sp;
    std::string why;
    if (!fs::exists(blob)) {
        std::printf("  [info] real discovery skipped: %s missing\n", blob.c_str());
        return;
    }
    const bool ok = discover_nvme_sensor(blob, "/sys", sp, why);
    std::printf("  [info] real discovery %s blob=%s\n", ok ? "OK" : "FAILED", blob.c_str());
    std::printf("  [info]   temp_input=%s\n", sp.temp_input.c_str());
    std::printf("  [info]   temp_max=%s\n", sp.temp_max.c_str());
    std::printf("  [info]   source=%s\n", sp.source.c_str());
    if (!ok) {
        std::printf("  [info]   why=%s\n", why.c_str());
        return;
    }
    int64_t t = 0;
    CHECK(read_i64_file(sp.temp_input, t));
    CHECK(t > 0);
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------
static Config base_cfg() {
    Config c;
    c.t1 = 78.0;
    c.t2 = 80.0;
    c.t3 = 85.0;
    c.floor_bps = 200 * MB;
    c.hysteresis_c = 1.0;   // the transition tests below are written against a 1 C band
    return c;
}

// Feeds one sensor tick per simulated second.
struct Feeder {
    Controller & c;
    double now = 0.0;
    Transition tick(double temp_c) {
        now += 1.0;
        return c.on_sample((int64_t) std::llround(temp_c * 1000.0), now);
    }
};

static void test_state_transitions() {
    Controller c(base_cfg());
    Feeder f{ c };
    CHECK(c.state() == State::COOL);
    CHECK(f.tick(70.0).changed == false);
    CHECK(c.state() == State::COOL);

    Transition tr = f.tick(78.5);
    CHECK(tr.changed && tr.from == State::COOL && tr.to == State::WARM);
    tr = f.tick(80.5);
    CHECK(tr.changed && tr.to == State::HOT);
    tr = f.tick(85.5);
    CHECK(tr.changed && tr.to == State::PAUSE);
    CHECK(c.snapshot().n_pauses == 1);

    // Paused until temp <= T2. 84 is still above T2, so it stays PAUSE.
    tr = f.tick(84.0);
    CHECK(!tr.changed && c.state() == State::PAUSE);
    tr = f.tick(80.0);   // <= T2 -> HOT
    CHECK(tr.changed && tr.to == State::HOT);

    // Hysteresis: HOT drops to WARM only below T2-1 = 79.
    tr = f.tick(79.5);
    CHECK(!tr.changed && c.state() == State::HOT);
    tr = f.tick(78.5);
    CHECK(tr.changed && tr.to == State::WARM);

    // WARM drops to COOL only below T1-1 = 77.
    tr = f.tick(77.5);
    CHECK(!tr.changed && c.state() == State::WARM);
    tr = f.tick(76.5);
    CHECK(tr.changed && tr.to == State::COOL);

    // A jump straight into PAUSE from COOL is allowed.
    tr = f.tick(90.0);
    CHECK(tr.changed && tr.from == State::COOL && tr.to == State::PAUSE);
    CHECK(c.snapshot().n_pauses == 2);
}

static void test_format_transition() {
    Transition tr;
    tr.changed = true;
    tr.from = State::WARM;
    tr.to = State::HOT;
    const double t[3] = { 78.2, 80.0, 85.4 };
    const std::string s = format_transition(tr, 80.5, t, 350 * MB);
    CHECK(s == "wp thermal: STATE WARM->HOT temp=80.5 T=(78.2,80.0,85.4) cap=350 MB/s");
}

static void test_aimd() {
    Config cfg = base_cfg();
    Controller c(cfg);
    Feeder f{ c };
    // Ten ticks of 500 MB/s reads while COOL: that is the measured throughput.
    for (int i = 0; i < 10; ++i) {
        c.account_read((uint64_t) (500 * MB));
        f.tick(70.0);
    }
    // First tick past T1 sets cap from measured throughput (500 MB/s over 10 s).
    c.account_read((uint64_t) (500 * MB));
    f.tick(78.5);
    CHECK_NEAR(c.snapshot().cap_bps, 500 * MB, 1.0);

    // HOT: multiplicative decrease by 0.8 each tick.
    f.tick(80.5);
    CHECK_NEAR(c.snapshot().cap_bps, 400 * MB, 1.0);
    f.tick(81.0);
    CHECK_NEAR(c.snapshot().cap_bps, 320 * MB, 1.0);
    f.tick(81.0);
    CHECK_NEAR(c.snapshot().cap_bps, 256 * MB, 1.0);
    f.tick(81.0);
    CHECK_NEAR(c.snapshot().cap_bps, 204.8 * MB, 0.01 * MB);
    f.tick(81.0);
    CHECK_NEAR(c.snapshot().cap_bps, 200 * MB, 1.0);   // floor
    f.tick(81.0);
    CHECK_NEAR(c.snapshot().cap_bps, 200 * MB, 1.0);   // never below floor

    // HOT at 79.5 (inside the hysteresis band): state stays HOT, cap unchanged.
    f.tick(79.5);
    CHECK(c.state() == State::HOT);
    CHECK_NEAR(c.snapshot().cap_bps, 200 * MB, 1.0);

    // Below T2-1 -> WARM, and the additive step applies: max(floor/2, 50 MB/s) = 100 MB/s.
    // Heavy traffic keeps the measured-rate ceiling out of the way.
    c.account_read((uint64_t) (50000 * MB));
    f.tick(78.5);
    CHECK(c.state() == State::WARM);
    CHECK_NEAR(c.snapshot().cap_bps, 300 * MB, 1.0);
    // WARM below T2 keeps recovering additively.
    c.account_read((uint64_t) (50000 * MB));
    f.tick(78.5);
    CHECK_NEAR(c.snapshot().cap_bps, 400 * MB, 1.0);
}

static void test_cap_ceiling() {
    Controller c(base_cfg());
    Feeder f{ c };
    // WARM with ~400 MB/s of traffic: the additive step stops at 1.25x the measured rate.
    for (int i = 0; i < 10; ++i) {
        c.account_read((uint64_t) (400 * MB));
        f.tick(70.0);
    }
    for (int i = 0; i < 20; ++i) {
        c.account_read((uint64_t) (400 * MB));
        f.tick(78.5);
    }
    CHECK(c.state() == State::WARM);
    CHECK_NEAR(c.snapshot().cap_bps, 500 * MB, 1.0);
    // Traffic stops: the ceiling falls with the measured rate, down to the floor.
    for (int i = 0; i < 12; ++i) {
        f.tick(78.5);
    }
    CHECK_NEAR(c.snapshot().cap_bps, 200 * MB, 1.0);
}

// Live 2026-10-14: the SN850X reports in ~2 C steps (76.8 <-> 78.8). With a 1 C band the
// state flapped COOL<->WARM every few seconds; the default 2 C band must hold WARM across one step.
static void test_default_hysteresis_holds_one_sensor_step() {
    Config cfg = base_cfg();
    cfg.hysteresis_c = Config().hysteresis_c;
    CHECK(cfg.hysteresis_c >= 2.0);
    cfg.t1 = 78.2;
    cfg.t2 = 80.0;
    cfg.t3 = 85.4;
    Controller c(cfg);
    Feeder f{ c };
    f.tick(76.8);
    CHECK(c.state() == State::COOL);
    f.tick(78.8);
    CHECK(c.state() == State::WARM);
    for (int i = 0; i < 5; ++i) {   // the observed one-step bounce no longer leaves WARM
        f.tick(76.8);
        CHECK(c.state() == State::WARM);
        f.tick(78.8);
        CHECK(c.state() == State::WARM);
    }
    f.tick(75.8);                   // a second step down does
    CHECK(c.state() == State::COOL);
}

static void test_spec_waits_behind_demand() {
    Controller c(base_cfg());
    Feeder f{ c };
    f.tick(78.5);   // WARM, cap = floor 200 MB/s, depth 50 MB
    CHECK(c.state() == State::WARM);
    const double t0 = f.now;
    // A full bucket admits a speculative read (demand is leaving budget unused).
    CHECK(c.admit(ReadClass::SPEC, 1 * MB, t0).pass);
    // Demand drains the bucket; speculative reads then wait for it to refill completely.
    CHECK(c.admit(ReadClass::DEMAND, 49 * MB, t0).pass);
    const Admission a = c.admit(ReadClass::SPEC, 1 * MB, t0);
    CHECK(!a.pass);
    CHECK(a.sleep_s > 0.0 && a.sleep_s <= 0.010 + 1e-12);
    // The bucket is empty now, so demand waits too.
    CHECK(!c.admit(ReadClass::DEMAND, 1 * MB, t0).pass);
    // With no demand for a full bucket's worth of time (0.25 s), spec passes: a demand
    // step blocked on that very speculative read can never deadlock it.
    CHECK(c.admit(ReadClass::SPEC, 1 * MB, t0 + 0.26).pass);
    // Exempt reads always pass.
    CHECK(c.admit(ReadClass::EXEMPT, 1 * MB, f.now).pass);
}

static void test_token_bucket_large_read() {
    Controller c(base_cfg());
    Feeder f{ c };
    f.tick(78.5);   // WARM, cap = floor = 200 MB/s (no bytes measured); depth 50 MB
    CHECK(c.state() == State::WARM);
    const double t0 = f.now;

    // First demand read bigger than the bucket: passes on a full bucket, goes into debt.
    const uint64_t big = (uint64_t) (1024 * MB);
    Admission a = c.admit(ReadClass::DEMAND, big, t0);
    CHECK(a.pass);

    // Small read right after: must wait (debt), and the wait is a bounded slice.
    a = c.admit(ReadClass::DEMAND, 4096, t0);
    CHECK(!a.pass);
    CHECK(a.sleep_s > 0.0 && a.sleep_s <= 0.010 + 1e-12);

    // Debt is ~1 GB at 200 MB/s, about 5 s. Advance and confirm it eventually admits.
    double now = t0;
    int waits = 0;
    while (!c.admit(ReadClass::DEMAND, 4096, now).pass) {
        now += 0.01;
        ++waits;
        CHECK(waits < 100000);   // no deadlock: the loop must terminate
        if (waits >= 100000) {
            break;
        }
    }
    CHECK(now - t0 > 4.0 && now - t0 < 6.0);
}

static void test_gate_cool_noop() {
    Controller c(base_cfg());
    int sleeps = 0;
    gate(c, ReadClass::SPEC, 1 * MB, [] { return 0.0; }, [&](double) { ++sleeps; });
    gate(c, ReadClass::DEMAND, 1 * MB, [] { return 0.0; }, [&](double) { ++sleeps; });
    CHECK(sleeps == 0);
    CHECK(c.snapshot().n_spec_held == 0);
    CHECK(c.snapshot().demand_throttled_ns == 0);
}

static void test_gate_spec_admitted_on_refill() {
    Controller c(base_cfg());
    Feeder f{ c };
    f.tick(78.5);   // WARM, depth 50 MB at 200 MB/s
    double clock = f.now;
    CHECK(c.admit(ReadClass::DEMAND, 50 * MB, clock).pass);   // bucket empty
    int sleeps = 0;
    // Still WARM throughout: the speculative read passes once the bucket refills (~0.25 s).
    gate(c, ReadClass::SPEC, 1 * MB,
         [&] { return clock; },
         [&](double s) { clock += s; ++sleeps; });
    CHECK(c.state() == State::WARM);
    CHECK(sleeps >= 25 && sleeps <= 260);
    CHECK(clock - f.now > 0.24 && clock - f.now < 0.3);
    CHECK(c.snapshot().n_spec_held == 1);   // one read held, counted once
}

static void test_gate_pause_release_after_45s() {
    Controller c(base_cfg());
    Feeder f{ c };
    f.tick(86.0);   // PAUSE at t=1, pause_since = 1
    CHECK(c.state() == State::PAUSE);
    const double pause_t0 = f.now;

    // Ticks at 86 C until 45 s have elapsed.
    while (f.now - pause_t0 < 30.0) {
        f.tick(86.0);
    }
    const double at30 = f.now;
    CHECK(c.state() == State::PAUSE);
    CHECK(!c.admit(ReadClass::DEMAND, 1 * MB, at30).pass);   // not released yet
    CHECK(!c.admit(ReadClass::SPEC, 1 * MB, at30).pass);

    while (f.now - pause_t0 <= 45.0) {
        f.tick(86.0);
    }
    CHECK(c.state() == State::PAUSE);
    // Released: demand passes at the floor rate; spec waits behind it.
    Admission a{ false, 0.0 };
    double now = f.now;
    for (int i = 0; i < 1000 && !a.pass; ++i) {
        a = c.admit(ReadClass::DEMAND, 1 * MB, now);
        if (!a.pass) {
            now += a.sleep_s;
        }
    }
    CHECK(a.pass);
    CHECK(!c.admit(ReadClass::SPEC, 1 * MB, now).pass);

    // Cooling to HOT ends the pause, and release is cleared.
    f.tick(80.0);
    CHECK(c.state() == State::HOT);
    CHECK(c.snapshot().n_pauses == 1);
}

static void test_bucket_rate() {
    // WARM with cap 100 MB/s: over 2 s of simulated time, demand admits about 200 MB.
    Config cfg = base_cfg();
    cfg.floor_bps = 100 * MB;
    Controller c(cfg);
    Feeder f{ c };
    f.tick(78.5);
    double now = f.now;
    const double t_end = now + 2.0;
    uint64_t admitted = 0;
    while (now < t_end) {
        const Admission a = c.admit(ReadClass::DEMAND, 1 * MB, now);
        if (a.pass) {
            admitted += 1 * MB;
        } else {
            now += a.sleep_s;
        }
    }
    // Initial depth is 25 MB, then 100 MB/s for 2 s.
    CHECK(admitted >= 200 * MB - 2 * MB && admitted <= 230 * MB);
}

int main(int argc, char ** argv) {
    const std::string real_blob = argc > 1 ? argv[1] : "/home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf";
    const fs::path tmp = fs::temp_directory_path() / ("wp-thermal-test-" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    test_thresholds();
    test_sensor_files(tmp);
    test_discovery_fake_tree(tmp);
    test_discovery_real(real_blob);
    test_state_transitions();
    test_format_transition();
    test_aimd();
    test_default_hysteresis_holds_one_sensor_step();
    test_spec_waits_behind_demand();
    test_cap_ceiling();
    test_token_bucket_large_read();
    test_gate_cool_noop();
    test_gate_spec_admitted_on_refill();
    test_gate_pause_release_after_45s();
    test_bucket_rate();

    fs::remove_all(tmp);
    std::printf("wp-thermal tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
