// TDD for the DSWS control-law logic (SPEC_DSWS_CONTROLLER.md). These four pure functions
// are the *reference semantics* the Phase-3 gfx1201 asm must match 1:1 (gate_try_win ->
// ds_cmpst_b32 LDS CAS; reserve_grow -> atomic_add/sub on the vgpr_reserved LDS counter;
// watermark_decision/epoch_of -> the boundary decision math). Locking the logic here, under a
// real multi-thread race, proves the protocol before it's transcribed into hand-asm.
#include "dsws_ctrl_model.cpp"
#include <atomic>
#include <thread>
#include <vector>
#include <cassert>
#include <cstdio>

int main() {
  // ---- watermark bands: occ<low starved(+1), occ>high over-served(-1), else dead-zone(0) ----
  assert(watermark_decision(0, 2, 6) == +1);   // empty ring -> starved
  assert(watermark_decision(7, 2, 6) == -1);   // full ring  -> over-served
  assert(watermark_decision(4, 2, 6) ==  0);   // mid        -> dead-zone
  assert(watermark_decision(2, 2, 6) ==  0);   // on the low edge is NOT starved (occ<low is strict)
  assert(watermark_decision(6, 2, 6) ==  0);   // on the high edge is NOT over-served (occ>high is strict)

  // ---- epoch clock: E = segments_processed >> EPOCH_SHIFT ----
  assert(epoch_of(0, 3) == 0 && epoch_of(7, 3) == 0 && epoch_of(8, 3) == 1 &&
         epoch_of(15, 3) == 1 && epoch_of(16, 3) == 2);
  assert(epoch_of(5, 0) == 5);                 // shift 0 -> every segment is its own epoch

  // ---- gate: exactly ONE winner per epoch among many racing waves (the anti-thrash CAS) ----
  for (uint32_t E = 1; E < 50; ++E) {
    std::atomic<uint32_t> g{E - 1};            // gate last fired at epoch E-1; a fresh epoch E is open
    std::atomic<int> wins{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 64; ++i) ts.emplace_back([&] { if (gate_try_win(g, E)) wins++; });
    for (auto& t : ts) t.join();
    assert(wins.load() == 1);                  // single-winner invariant
    assert(g.load() == E);                     // gate advanced to E
  }
  // a SECOND attempt at the same epoch must lose (already fired this epoch)
  { std::atomic<uint32_t> g{5}; assert(!gate_try_win(g, 5)); assert(!gate_try_win(g, 4)); }

  // ---- reservation: concurrent grows never exceed budget; over-budget grows cleanly undo ----
  { std::atomic<uint32_t> r{0}; std::atomic<int> ok{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 10; ++i) ts.emplace_back([&] { if (reserve_grow(r, 30, 100)) ok++; });
    for (auto& t : ts) t.join();
    assert(r.load() <= 100);                   // envelope never blown
    assert(ok.load() == 3);                    // 3*30=90<=100; a 4th would be 120>100 -> rejected+undone
    assert(r.load() == 90);                    // exactly the 3 winners' reservations remain
  }

  // ---- snapshot/quiesce (Phase B Decision 1) ----
  {
    // snapshot freezes the counts used to size the quiesce sentinels
    WgSnap s = snapshot_counts(4, 2, 2);           // G=6, FN=4
    // not ready: rowblk short of G + nC terminal bails
    assert(!quiesce_ready(6 + 3, 4 + 2, 6 + 2, s, 6, 4));  // rowblk 9 < 6+4
    // ready: every counter reached threshold + snapshot bails
    assert( quiesce_ready(6 + 4, 4 + 2, 6 + 2, s, 6, 4));
    // a moved partition (3c3a2b) needs different sentinels; old snapshot is wrong high
    WgSnap s2 = snapshot_counts(3, 3, 2);                  // sentinels: rowblk>=9, bfrag>=6, arow>=9
    assert( quiesce_ready(6 + 3, 4 + 2, 6 + 3, s2, 6, 4)); // 9,6,9 all meet -> ready
    assert(!quiesce_ready(6 + 3, 4 + 2, 6 + 2, s2, 6, 4)); // arow 8 < 9 -> NOT ready
    // N-1 cross-check agrees at the ready point (N=8 -> 7 bails)
    assert( quiesce_ready_nm1(7, 8));
    assert(!quiesce_ready_nm1(6, 8));
  }

  {
    // Under any interleaving of N-1 bails, quiesce_ready_nm1 must not fire before the last bail.
    for (uint32_t trial = 0; trial < 64; ++trial) {
      std::atomic<uint32_t> cnt{0};
      std::atomic<bool> early{false};
      std::vector<std::thread> ts;
      const uint32_t N = 8;
      for (uint32_t w = 0; w < N - 1; ++w)
        ts.emplace_back([&]{
          if (quiesce_ready_nm1(cnt.load(), N)) early.store(true); // read BEFORE our bump
          cnt.fetch_add(1, std::memory_order_acq_rel);
        });
      for (auto& t : ts) t.join();
      assert(!early.load());               // never ready with a bail still outstanding
      assert(quiesce_ready_nm1(cnt.load(), N)); // ready once all N-1 landed
    }
  }

  // ---- Task 1: dispatch + cooldown + pool invariants ----
  assert(role_dispatch(24) == COMPUTE);
  assert(role_dispatch(28) == AFEED);
  assert(role_dispatch(32) == BFEED);
  // cooldown counts down and saturates at 0; in_cooldown true iff >0
  assert(cooldown_step(3) == 2 && cooldown_step(1) == 0 && cooldown_step(0) == 0);
  assert(in_cooldown(1) && !in_cooldown(0));
  // no-parking budget invariant: 16 lean waves fit iff budget >= 512
  assert( pool_fits_lean(16, 32, 512));
  assert(!pool_fits_lean(16, 32, 511));
  // quiesce cross-check generalizes WAVES-1 -> N_POOL-1
  assert( quiesce_ready_pool(11, 12) && !quiesce_ready_pool(10, 12));
  printf("dsws_ctrl_model: dispatch/cooldown/pool OK\n");

  // ---- Pool-T7 brick repro: first-time entry MUST run _alloc/_init before _follow ----
  //   A wave's first entry has to (a) commit its lean VGPR alloc (s_alloc_vgpr 32), (b) wait the
  //   claimer's INITFLAG==0xACED LDS rendezvous, and (c) seed its local epoch (s35=0) -- all of which
  //   live only inside the per-role _alloc/_init blocks. Landing first entry on _follow (what the
  //   .Ldispatch re-dispatch trampoline does) skips all three -> followers desync from the epoch clock,
  //   never reach their _quiesce bail, and the claimer spins forever in .Lclaimer_wait_done (the brick).
  {
    WaveEntry via_dispatch = simulate_first_entry(LAND_FOLLOW);     // seed arms -> .Ldispatch (buggy)
    assert(!via_dispatch.ran_alloc && !via_dispatch.waited_initflag && !via_dispatch.seeded_epoch);
    assert(!entry_safe(via_dispatch));                              // first entry via _follow is UNSAFE

    WaveEntry via_role = simulate_first_entry(LAND_ROLE_ENTRY);     // seed arms -> .Lbfeed/.Lafeed/.Lcompute
    assert(via_role.ran_alloc && via_role.waited_initflag && via_role.seeded_epoch);
    assert(entry_safe(via_role));                                   // first entry via role label is SAFE

    // End-to-end handshake: mix 4c2a2b -> WAVES=8, so 7 non-claimer followers must each bump QUIESCE_CNT
    // once per super-tile. Unsafe entry desyncs EVERY seed-entered follower (deterministic code-path
    // defect, not a timing race -> all 7 lost), so the claimer's QUIESCE_CNT>=WAVES-1 gate never closes.
    assert(!claimer_quiesce_converges(LAND_FOLLOW,    8));          // BUG: reproduces the Pool-T7 hang
    assert( claimer_quiesce_converges(LAND_ROLE_ENTRY, 8));         // FIX: claimer advances
  }
  printf("dsws_ctrl_model: entry-contract (Pool-T7 repro) OK\n");

  printf("dsws_ctrl_model: ALL PASS\n");
  return 0;
}
