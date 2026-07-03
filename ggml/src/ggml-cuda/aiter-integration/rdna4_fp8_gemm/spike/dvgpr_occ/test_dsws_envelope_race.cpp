// Forward-progress race for the DSWS compute-burst sum-envelope (.Lcompute_reserve/grow/release).
// NCOMP threads each loop {claim rowblk -> reserve_spin(+Δ) -> bounded burst -> reserve_release(-Δ)}.
// A watchdog trips if any thread fails to finish its rowblks (models a permit-starvation hang on CPU
// what cannot be safely bisected on the compositor-attached gfx1201). Target: 0 stalls for PEAK_CONC>=1.
#include "dsws_ctrl_model.cpp"
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>

static constexpr uint32_t VLEAN = 32, NFV = 112, D = NFV - VLEAN;   // Δ = 80
static constexpr uint32_t WAVES = 8, NCOMP = 4;
static constexpr uint32_t ROWBLKS = 64;                              // rowblks each compute wave completes
static constexpr uint64_t SPIN_LIMIT = 50'000'000ull;

static bool run_once(uint32_t peak_conc) {
    const uint32_t budget = WAVES * VLEAN + peak_conc * D;
    std::atomic<uint32_t> resv{WAVES * VLEAN};
    std::atomic<uint32_t> done{0};
    std::atomic<bool> stalled{false};
    std::vector<std::thread> ts;
    for (uint32_t w = 0; w < NCOMP; ++w) {
        ts.emplace_back([&]{
            for (uint32_t r = 0; r < ROWBLKS && !stalled.load(); ++r) {
                uint64_t spins = 0;
                while (!reserve_grow(resv, D, budget)) {
                    if (++spins > SPIN_LIMIT) { stalled.store(true); return; }
                    std::this_thread::yield();
                }
                // bounded "burst": a few atomic touches, then release
                for (int k = 0; k < 8; ++k) done.fetch_add(0);
                reserve_release(resv, D);
            }
            done.fetch_add(1);
        });
    }
    for (auto& t : ts) t.join();
    bool ok = !stalled.load() && done.load() >= NCOMP && resv.load() == WAVES * VLEAN;
    if (!ok) printf("  STALL peak_conc=%u  resv=%u done=%u stalled=%d\n",
                    peak_conc, resv.load(), done.load(), (int)stalled.load());
    return !ok;
}

int main() {
    const int TRIALS = 200;
    int fails = 0;
    for (uint32_t pc = 1; pc <= 3; ++pc) {
        int f = 0;
        for (int i = 0; i < TRIALS; ++i) if (run_once(pc)) f++;
        printf("[peak_conc=%u] %d/%d trials stalled\n", pc, f, TRIALS);
        fails += f;
    }
    if (fails == 0) printf("dsws_envelope_race: NO STALL — envelope guarantees forward progress\n");
    else            printf("dsws_envelope_race: STALL REPRODUCED (%d)\n", fails);
    return fails ? 1 : 0;
}
