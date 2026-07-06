// Offline reproduction harness for the DSWS Phase-B "chunk-1-passes / chunk-2-hangs" GPU stall
// (occ_kernel_dsws.s, DSWS2_CONV=1, conversions DORMANT). It models the barrier-free claimer+followers
// QUIESCE/epoch handshake of .Lclaimer_wait_done as real std::threads and trips a watchdog if the
// claimer ever fails to advance -- reproducing on CPU what cannot be safely bisected on the
// compositor-attached gfx1201.
//
// Faithful to the asm ordering (claim loop lines 748-798, followers 930-1258):
//   claimer per super-tile: reset NEXT/DONE=0 ; snapshot(parity newEpoch)=launch mix ; QUIESCE=0 ;
//                           publish EPOCH=newEpoch ; BSTAGE (claimer is a B-feed-class wave) ; wait_done
//   wait_done advances iff: DONE counters met  AND  s50 (NEXT >= snapshot-sized sentinels)
//                           AND  s51 (QUIESCE >= WAVES-1)
//   follower per super-tile: wait EPOCH!=local ; claim-loop its role counter to over-claim ; QUIESCE++ ;
//                            re-dispatch -> wait NEXT epoch
//
// Two memory-order regimes are swept:
//   STRONG (seq_cst)   -- logical soundness check of the protocol.
//   WEAK   (relaxed)   -- closest CPU proxy for gfx1201's weak LDS ordering (the kernel uses NO s_barrier;
//                         EPOCH_OFF / QUIESCE reset are plain lds_put). A stall that appears ONLY here
//                         points at a missing ordering/fence between the QUIESCE-reset+EPOCH-publish and
//                         the followers' observation, rather than a control-flow bug.
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>

// ---- 4c2a2b role mix ----
static constexpr uint32_t NCOMP = 4, NAFEED = 2, NBFEED = 2;
static constexpr uint32_t WAVES = NCOMP + NAFEED + NBFEED;      // 8
static constexpr uint32_t G = 6, FN = 4;
static constexpr uint32_t STI_TERMINAL = 0xFFFFFFFFu;
static constexpr uint32_t SUPERTILES = 24;                      // > one chunk (8), to catch the boundary bug
static constexpr uint64_t SPIN_LIMIT = 20'000'000ull;          // watchdog: claimer stall => reproduced hang

struct Lds {
    std::atomic<uint32_t> epoch{0}, sti{0};
    std::atomic<uint32_t> rowblk_next{0}, rowblk_done{0};
    std::atomic<uint32_t> bfrag_next{0}, bfrag_done{0};
    std::atomic<uint32_t> arow_next{0}, arow_done{0};
    std::atomic<uint32_t> quiesce{0};
    std::atomic<uint32_t> snapC[2], snapA[2], snapB[2];         // per-parity snapshot of the live role mix
    std::atomic<bool>     stalled{false};
    const char*           stall_reason = nullptr;
    std::atomic<uint32_t> stall_tile{0};
};

enum Role { COMPUTE, AFEED, BFEED };

// One follower wave: sync on epoch, claim-loop its role counter until the over-claim, bump QUIESCE, repeat.
static void follower(Lds& L, Role role, std::memory_order mo) {
    uint32_t local_epoch = 0;
    while (true) {
        // wait for the next super-tile (epoch change) -- mirrors .L*_follow
        while (true) {
            if (L.stalled.load(mo)) return;
            uint32_t e = L.epoch.load(mo);
            if (e != local_epoch) { local_epoch = e; break; }
            std::this_thread::yield();
        }
        if (L.sti.load(mo) == STI_TERMINAL) return;             // A7 terminal -> retire
        // role work: claim until the fetch_add returns >= threshold (the terminal over-claim), then bail
        if (role == COMPUTE) {
            while (true) { uint32_t r = L.rowblk_next.fetch_add(1, mo); if (r >= G) break; L.rowblk_done.fetch_add(1, mo); }
        } else if (role == AFEED) {
            while (true) { uint32_t r = L.arow_next.fetch_add(1, mo);   if (r >= G)  break; L.arow_done.fetch_add(1, mo); }
        } else { // BFEED
            while (true) { uint32_t r = L.bfrag_next.fetch_add(1, mo);  if (r >= FN) break; L.bfrag_done.fetch_add(1, mo); }
        }
        L.quiesce.fetch_add(1, mo);                             // commit-before-bump: exactly one bump/super-tile
    }
}

// Returns true if the claimer stalled (bug reproduced).
static bool run_once(std::memory_order mo) {
    Lds L;
    for (int p = 0; p < 2; ++p) { L.snapC[p]=NCOMP; L.snapA[p]=NAFEED; L.snapB[p]=NBFEED; }

    std::vector<std::thread> ts;
    ts.emplace_back(follower, std::ref(L), BFEED, mo);          // 1 B-feed follower (wid in [1,NBFEED))
    ts.emplace_back(follower, std::ref(L), AFEED, mo);
    ts.emplace_back(follower, std::ref(L), AFEED, mo);          // NAFEED A-feed
    for (uint32_t i = 0; i < NCOMP; ++i) ts.emplace_back(follower, std::ref(L), COMPUTE, mo);

    uint32_t local_epoch = 0;
    for (uint32_t tile = 1; tile <= SUPERTILES && !L.stalled.load(mo); ++tile) {
        // reset per-super-tile counters BEFORE the epoch bump (followers see them reset) -- asm 763-769
        L.rowblk_next.store(0, mo); L.rowblk_done.store(0, mo);
        L.bfrag_next.store(0, mo);  L.bfrag_done.store(0, mo);
        L.arow_next.store(0, mo);   L.arow_done.store(0, mo);
        uint32_t newEpoch = local_epoch + 1;
        // Step 4: snapshot live mix into parity(newEpoch) (dormant => launch mix), reset QUIESCE -- asm 782-787
        int par = newEpoch & 1;
        L.snapC[par].store(NCOMP, mo); L.snapA[par].store(NAFEED, mo); L.snapB[par].store(NBFEED, mo);
        L.quiesce.store(0, mo);
        L.sti.store(tile, mo);                                  // valid tile
        L.epoch.store(newEpoch, mo);                            // publish EPOCH LAST -- asm 789
        local_epoch = newEpoch;
        // BSTAGE: the claimer is a B-feed-class wave and stages B this super-tile (contributes one over-claim)
        while (true) { uint32_t r = L.bfrag_next.fetch_add(1, mo); if (r >= FN) break; L.bfrag_done.fetch_add(1, mo); }
        // A7 advance gate (.Lclaimer_wait_done): DONE met AND s50 sentinels AND s51 QUIESCE cross-check
        int thisPar = newEpoch & 1;
        uint64_t spins = 0;
        while (true) {
            bool done_ok = L.rowblk_done.load(mo) >= G && L.bfrag_done.load(mo) >= FN && L.arow_done.load(mo) >= G;
            bool s50 = L.rowblk_next.load(mo) >= (G  + L.snapC[thisPar].load(mo))
                    && L.bfrag_next.load(mo)  >= (FN + L.snapB[thisPar].load(mo))
                    && L.arow_next.load(mo)   >= (G  + L.snapA[thisPar].load(mo));
            bool s51 = L.quiesce.load(mo) >= (WAVES - 1);
            if (done_ok && s50 && s51) break;
            if (++spins > SPIN_LIMIT) {
                L.stall_reason = !done_ok ? "DONE counters never met"
                               : !s50     ? "s50 snapshot sentinels never met"
                               :            "s51 QUIESCE never reached WAVES-1";
                L.stall_tile.store(tile, mo);
                L.stalled.store(true, mo);
                break;
            }
        }
    }
    // terminal: release followers (both the normal-exit and stall paths funnel through here)
    bool stalled = L.stalled.load(mo);
    L.sti.store(STI_TERMINAL, mo);
    L.epoch.fetch_add(1, mo);           // wake any followers still spinning on an epoch change
    for (auto& t : ts) t.join();
    if (stalled)
        printf("  STALL @ super-tile %u: %s  (rn=%u rd=%u bn=%u bd=%u an=%u ad=%u q=%u)\n",
               L.stall_tile.load(), L.stall_reason,
               L.rowblk_next.load(), L.rowblk_done.load(), L.bfrag_next.load(), L.bfrag_done.load(),
               L.arow_next.load(), L.arow_done.load(), L.quiesce.load());
    return stalled;
}

int main() {
    struct Regime { const char* name; std::memory_order mo; };
    Regime regimes[] = { {"STRONG(seq_cst)", std::memory_order_seq_cst},
                         {"WEAK(relaxed)   ", std::memory_order_relaxed} };
    const int TRIALS = 400;
    int total_stalls = 0;
    for (auto& rg : regimes) {
        int stalls = 0;
        for (int i = 0; i < TRIALS; ++i) if (run_once(rg.mo)) stalls++;
        printf("[%s] %d/%d trials stalled\n", rg.name, stalls, TRIALS);
        total_stalls += stalls;
    }
    if (total_stalls == 0)
        printf("dsws_quiesce_race: NO STALL reproduced -- protocol logic is race-free under thread stress\n");
    else
        printf("dsws_quiesce_race: STALL REPRODUCED (%d total)\n", total_stalls);
    return 0;
}
