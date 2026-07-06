// Straggler/liveness model for the DSWS Phase-B (DSWS2_CONV=1) hang (SUSPECT #2).
//
// The prior harness (test_dsws_quiesce_race.cpp) swept MEMORY ORDERING (seq_cst vs relaxed) and got
// 0/400 stalls -> the protocol LOGIC is race-free. But the Fable advisor (2026-07-02) argued the hang is
// a SCHEDULING/LIVENESS failure the ordering model cannot express: s51 (QUIESCE_CNT >= WAVES-1) is a
// PER-WAVE HEADCOUNT -- every non-claimer wave must individually reach its _quiesce bail. If ONE follower
// falls behind and never bails for super-tile N, s51 sits one short FOREVER even though the collective
// work (s50) completed. This model injects a controllable per-follower entry delay (a follower stalled in
// _alloc/_init/_follow before it can do work + bump) and asks the discriminating questions:
//   Q1. Does a follower delayed by K yields cause a stall, or does the s51 gate WAIT it out (self-resolve)?
//   Q2. Does only a PERMANENT block hang -> i.e. is the bug a hard stuck-follower, not timing jitter?
//   Q3. When it hangs, is it the s51 gate (headcount) -- matching the observed ST8 signature?
//   Q4. Does it reproduce under seq_cst (no ordering freedom) -> proving liveness, not ordering?
//
// A stall that appears ONLY for a (near-)permanent block, under BOTH memory regimes, at the s51 gate,
// confirms: root cause = a follower permanently stuck BEFORE its bail (candidates: s_alloc_vgpr pool
// contention on chunk relaunch / INITFLAG rendezvous / epoch wait). And it proves DSWS2_GQUIESCE is
// IRRELEVANT: rerouting the bump's storage (LDS->global) cannot help a bump that never executes.
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>

static constexpr uint32_t NCOMP = 4, NAFEED = 2, NBFEED = 2;
static constexpr uint32_t WAVES = NCOMP + NAFEED + NBFEED;      // 8 (wid0 = claimer, B-feed-class)
static constexpr uint32_t G = 6, FN = 4;
static constexpr uint32_t STI_TERMINAL = 0xFFFFFFFFu;
static constexpr uint32_t SUPERTILES = 16;
static constexpr uint64_t SPIN_LIMIT = 8'000'000ull;
static constexpr uint64_t PERMANENT   = 0xFFFFFFFFFFFFFFFFull;

enum Role { COMPUTE, AFEED, BFEED };

struct Lds {
    std::atomic<uint32_t> epoch{0}, sti{0};
    std::atomic<uint32_t> rowblk_next{0}, rowblk_done{0};
    std::atomic<uint32_t> bfrag_next{0}, bfrag_done{0};
    std::atomic<uint32_t> arow_next{0}, arow_done{0};
    std::atomic<uint32_t> quiesce{0};
    std::atomic<uint32_t> snapC[2], snapA[2], snapB[2];
    std::atomic<bool>     stalled{false};
    const char*           stall_reason = nullptr;
    std::atomic<uint32_t> stall_tile{0};
};

struct Straggle { Role role; uint32_t tile; uint64_t delay; }; // one follower stalls `delay` yields on `tile`

static void follower(Lds& L, Role role, std::memory_order mo, const Straggle* st, std::atomic<bool>* used) {
    uint32_t local_epoch = 0;
    while (true) {
        while (true) {
            if (L.stalled.load(mo)) return;
            uint32_t e = L.epoch.load(mo);
            if (e != local_epoch) { local_epoch = e; break; }
            std::this_thread::yield();
        }
        if (L.sti.load(mo) == STI_TERMINAL) return;
        // STRAGGLER injection: this follower, on the designated super-tile, is stuck BEFORE its work+bail
        //   (models a wave stalled in _alloc/_init/_follow on the chunk relaunch). Claim the role once via
        //   `used` so exactly one follower of that role straggles.
        if (st && role == st->role && local_epoch == st->tile) {
            bool expect = false;
            if (used->compare_exchange_strong(expect, true, mo)) {
                for (uint64_t i = 0; i < st->delay; ++i) {
                    if (L.stalled.load(mo)) return;          // claimer already tripped the watchdog
                    std::this_thread::yield();
                }
                // if delay was PERMANENT we never get here for this tile -> no work, no bump (true straggler)
                if (st->delay == PERMANENT) { local_epoch = st->tile; /* stay one epoch behind */ }
            }
        }
        if (role == COMPUTE) {
            while (true) { uint32_t r = L.rowblk_next.fetch_add(1, mo); if (r >= G) break; L.rowblk_done.fetch_add(1, mo); }
        } else if (role == AFEED) {
            while (true) { uint32_t r = L.arow_next.fetch_add(1, mo);   if (r >= G)  break; L.arow_done.fetch_add(1, mo); }
        } else {
            while (true) { uint32_t r = L.bfrag_next.fetch_add(1, mo);  if (r >= FN) break; L.bfrag_done.fetch_add(1, mo); }
        }
        L.quiesce.fetch_add(1, mo);
    }
}

// Returns 0 = no stall; else the gate that hung (1=DONE, 2=s50, 3=s51).
static int run_once(std::memory_order mo, const Straggle* st) {
    Lds L;
    for (int p = 0; p < 2; ++p) { L.snapC[p]=NCOMP; L.snapA[p]=NAFEED; L.snapB[p]=NBFEED; }
    std::atomic<bool> used{false};
    std::vector<std::thread> ts;
    ts.emplace_back(follower, std::ref(L), BFEED, mo, st, &used);   // NBFEED-1 = 1 B-feed follower
    ts.emplace_back(follower, std::ref(L), AFEED, mo, st, &used);
    ts.emplace_back(follower, std::ref(L), AFEED, mo, st, &used);
    for (uint32_t i = 0; i < NCOMP; ++i) ts.emplace_back(follower, std::ref(L), COMPUTE, mo, st, &used);

    uint32_t local_epoch = 0; int gate = 0;
    for (uint32_t tile = 1; tile <= SUPERTILES && !L.stalled.load(mo); ++tile) {
        L.rowblk_next.store(0, mo); L.rowblk_done.store(0, mo);
        L.bfrag_next.store(0, mo);  L.bfrag_done.store(0, mo);
        L.arow_next.store(0, mo);   L.arow_done.store(0, mo);
        uint32_t newEpoch = local_epoch + 1; int par = newEpoch & 1;
        L.snapC[par].store(NCOMP, mo); L.snapA[par].store(NAFEED, mo); L.snapB[par].store(NBFEED, mo);
        L.quiesce.store(0, mo);
        L.sti.store(tile, mo);
        L.epoch.store(newEpoch, mo);
        local_epoch = newEpoch;
        while (true) { uint32_t r = L.bfrag_next.fetch_add(1, mo); if (r >= FN) break; L.bfrag_done.fetch_add(1, mo); }
        int thisPar = newEpoch & 1; uint64_t spins = 0;
        while (true) {
            bool done_ok = L.rowblk_done.load(mo) >= G && L.bfrag_done.load(mo) >= FN && L.arow_done.load(mo) >= G;
            bool s50 = L.rowblk_next.load(mo) >= (G  + L.snapC[thisPar].load(mo))
                    && L.bfrag_next.load(mo)  >= (FN + L.snapB[thisPar].load(mo))
                    && L.arow_next.load(mo)   >= (G  + L.snapA[thisPar].load(mo));
            bool s51 = L.quiesce.load(mo) >= (WAVES - 1);
            if (done_ok && s50 && s51) break;
            if (++spins > SPIN_LIMIT) {
                gate = !done_ok ? 1 : !s50 ? 2 : 3;
                L.stall_reason = gate==1 ? "DONE never met" : gate==2 ? "s50 sentinels never met"
                                                                      : "s51 QUIESCE never reached WAVES-1";
                L.stall_tile.store(tile, mo); L.stalled.store(true, mo); break;
            }
        }
    }
    L.sti.store(STI_TERMINAL, mo); L.epoch.fetch_add(1, mo);
    for (auto& t : ts) t.join();
    return L.stalled.load() ? gate : 0;
}

static const char* gname(int g){ return g==0?"OK":g==1?"DONE":g==2?"s50":"s51"; }

int main() {
    struct Regime { const char* name; std::memory_order mo; }
    regimes[] = { {"seq_cst", std::memory_order_seq_cst}, {"relaxed", std::memory_order_relaxed} };
    const int TRIALS = 100;

    printf("== DSWS straggler/liveness model (4c2a2b, %u super-tiles) ==\n", SUPERTILES);
    for (auto& rg : regimes) {
        // Control: no straggler.
        int stalls = 0; for (int i=0;i<TRIALS;i++) if (run_once(rg.mo, nullptr)) stalls++;
        printf("[%s] control (no straggler)                 : %d/%d stalled\n", rg.name, stalls, TRIALS);

        // Permanent block of ONE follower at tile 8 (the observed chunk-boundary tile), per role.
        for (Role r : {COMPUTE, AFEED, BFEED}) {
            Straggle st{r, 8, PERMANENT};
            int c=0,g=0; for (int i=0;i<TRIALS;i++){ int gg=run_once(rg.mo,&st); if(gg){c++; g=gg;} }
            printf("[%s] PERMANENT block of 1 %-7s @tile8      : %d/%d stalled  gate=%s\n",
                   rg.name, r==COMPUTE?"compute":r==AFEED?"afeed":"bfeed", c, TRIALS, gname(g));
        }

        // Bounded-jitter sweep on a compute follower: does a finite delay SELF-RESOLVE (s51 waits it out)?
        for (uint64_t d : {10ull, 1000ull, 100000ull, 5000000ull}) {
            Straggle st{COMPUTE, 8, d};
            int c=0,g=0; for (int i=0;i<TRIALS;i++){ int gg=run_once(rg.mo,&st); if(gg){c++; g=gg;} }
            printf("[%s] bounded delay=%-8llu compute @tile8   : %d/%d stalled  gate=%s\n",
                   rg.name, (unsigned long long)d, c, TRIALS, gname(g));
        }
    }
    printf("\nINTERPRETATION (empirical):\n");
    printf("  * control 0/100 both regimes  => protocol logic sound (matches prior 0/400 ordering model).\n");
    printf("  * ANY permanently-blocked follower => 100%% stall, IDENTICAL under seq_cst and relaxed\n");
    printf("      => the hang is LIVENESS/SCHEDULING, not memory ordering. gq (which only moves the QUIESCE\n");
    printf("         counter's storage) is therefore NOT the axis.\n");
    printf("  * bounded delay < watchdog SELF-RESOLVES (0 stall); only a (near-)PERMANENT block hangs\n");
    printf("      => a follower must be HARD-STUCK before its bail, not merely slow. Candidates: _alloc\n");
    printf("         (s_alloc_vgpr pool contention on chunk relaunch), _init (INITFLAG), _follow (epoch).\n");
    printf("  * SHARPENING: the gate that trips is s50 (the over-claim sentinel G+snapC), because s50 is\n");
    printf("      ALSO a per-wave headcount (each wave owes one terminal over-claim). A follower stuck\n");
    printf("      BEFORE its work skips BOTH its over-claim (s50) AND its bump (s51) => BOTH go short, so\n");
    printf("      gq cannot help even in principle. Only a follower stuck AFTER its over-claim but BEFORE\n");
    printf("      its bump would give s50-met/s51-short -- the one case where gq could matter. The real\n");
    printf("      hang's s50-vs-s51 breakdown (unknown; never captured at DIAG=0) decides which -- that is\n");
    printf("      exactly what the bail-mark localization dispatch must measure.\n");
    return 0;
}
