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

  printf("dsws_ctrl_model: ALL PASS\n");
  return 0;
}
