// Flow-kernel (occ_kernel_dsws_flow.s) completion model — the offline discriminator the review round
// asked for (Fable F8, Grok §6.1). The existing test_dsws_* models cover the PHASE-B quiesce/envelope
// protocol; NONE model the flow kernel's 3-frontier + POOL_N=1 group-split completion, which is where the
// live "group-split hang" lives. This models exactly that, and injects the two hazards the reviewers
// flagged, so we can decide H1/H2/H3 without a GPU dispatch.
//
// It is a DETERMINISTIC round-robin interleaving simulator (no real threads => it can never itself hang):
// each wave performs at most one atomic-equivalent action per turn; "stuck" = no wave made progress for a
// full sweep. This mirrors the barrier-free LDS semantics faithfully at the level that matters for
// liveness: single-writer ASSIGN/reset, monotone ds_cmpstore frontier advances, per-slot fetch-add claims.
//
//   build+run:  g++ -std=c++17 -O2 test_dsws_flow_model.cpp -o /tmp/t && /tmp/t
//
// Models POOL_N=1 (the group-split bring-up config): exactly one operand slot, so ASSIGN <= DRAIN+1 and the
// pipeline is assign -> stage -> drain -> free -> assign-next, serialized. Group-split (GROUPS>1) is folded
// in as: total super-tiles SUPER = TILES * GROUPS * n_kseg, each a slot-fill with RBDONE target = ACC_N,
// ARDONE target = G (whole-tile A staging, as the CURRENT source does), BFDONE target = FN.
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>

enum Role { COMPUTE, AFEED, BFEED };

// ---- shared "LDS" state (plain ints; the simulator serializes actions so no atomics needed) ----
struct Flow {
    // frontiers (monotone)
    uint32_t ASSIGN = 0, STAGE = 0, DRAIN = 0;
    // single pool slot (POOL_N=1) per-slot counters
    uint32_t RBNEXT = 0, RBDONE = 0, BFNEXT = 0, BFDONE = 0, ARNEXT = 0, ARDONE = 0;
    uint32_t STAMP = 0;              // super-tile id currently in the slot
    bool     slot_reset = true;      // coordinator has reset the slot for the current ASSIGN
    // exit
    uint32_t occ0 = 0;               // live count (waves that live++'d and not yet live--'d)
    // config
    uint32_t SUPER = 0;              // total super-tiles to process
    uint32_t G = 6, FN = 4, ACC_N = 3;
    // hazard toggles / fix knobs
    bool multi_observer_drain = false;   // FIX(a): any compute observer of RBDONE>=ACC_N may advance DRAIN
    bool stage_gate_accn = false;        // BUG (Grok §2.2): STAGE gate uses ACC_N but A is staged only ACC_N rows
    int  inject_completer_stuck_at = -1; // super-tile whose unique completer enters the shrink spin (Grok §2.4)
    // FIX(b): bound the s_alloc_vgpr shrink spin. 0 = unbounded (current source, no deadman). >0 = escapes
    //   after this many sweeps. escape_does_drain: true = on escape the completer still performs its DRAIN
    //   CAS then retires (a bounded-shrink-that-continues); false = deadman force-retire WITHOUT the DRAIN CAS.
    uint32_t shrink_bound = 0;
    bool     escape_does_drain = false;
};

// A wave: role + a "stuck" latch (models the s_alloc_vgpr shrink spin: the wave stops progressing).
struct Wave { int wid; Role role; bool stuck = false; bool retired = false; uint32_t stuck_sweep = 0; };

// One compute action on the current DRAIN slot. Returns true if it made progress.
static bool compute_step(Flow& f, Wave& w, uint32_t sweep) {
    if (w.stuck) return false;
    if (f.DRAIN >= f.STAGE) return false;              // nothing fully staged -> (coast handled by caller)
    // claim a rowblk
    if (f.RBNEXT < f.ACC_N) {
        f.RBNEXT++;                                    // fetch-add
        f.RBDONE++;                                    // "compute + flush" then bump RBDONE
        // becoming the unique completer: RBDONE just reached ACC_N
        if (f.RBDONE == f.ACC_N) {
            if ((int)f.STAMP == f.inject_completer_stuck_at) {
                // models: after RBDONE++ the completer enters the s_alloc_vgpr shrink spin
                //   (occ_kernel_dsws_flow.s .Lflow_bshrink) BEFORE its DRAIN CAS. Escape handled in run_flow.
                w.stuck = true; w.stuck_sweep = sweep;
                return true;
            }
            f.DRAIN++;                                 // completer advances DRAIN (ds_cmpstore), frees the slot
        }
        return true;
    }
    // RBNEXT exhausted (>=ACC_N): the slot is (or should be) drained.
    if (f.multi_observer_drain && f.RBDONE >= f.ACC_N && f.DRAIN == f.STAMP) {
        f.DRAIN++;                                     // FIX(a): any observer advances DRAIN (idempotent)
        return true;
    }
    return false;
}

// One feed action (AFEED stages A rows, BFEED stages B frags) on the current STAGE slot.
static bool feed_step(Flow& f, Wave& w, Role fr) {
    if (w.stuck) return false;
    if (f.STAGE >= f.ASSIGN) return false;             // nothing assigned yet (coordinator behind)
    uint32_t ar_target = f.G;                          // CURRENT source: whole-tile A staging (ARDONE >= G)
    uint32_t ar_claim  = f.stage_gate_accn ? f.ACC_N : f.G; // BUG variant: only ACC_N A rows staged
    if (fr == AFEED) {
        if (f.ARNEXT < ar_claim) { f.ARNEXT++; f.ARDONE++; return true; }
    } else { // BFEED
        if (f.BFNEXT < f.FN) { f.BFNEXT++; f.BFDONE++; return true; }
    }
    // if slot fully staged, advance STAGE (monotone cmpstore; losers retry)
    if (f.ARDONE >= ar_target && f.BFDONE >= f.FN && f.STAGE == f.STAMP) {
        f.STAGE++;
        return true;
    }
    return false;
}

// Coordinator (wid0) assign duty: when the single slot is free (ASSIGN==DRAIN) and work remains, reset the
// slot and assign the next super-tile. Single writer => no CAS. Returns true if it assigned.
static bool coord_assign(Flow& f) {
    if (f.ASSIGN >= f.SUPER) return false;             // all work assigned
    if (f.ASSIGN - f.DRAIN >= 1) return false;         // POOL_N=1: slot busy
    // slot free & drained -> single-writer reset then publish
    f.RBNEXT = f.RBDONE = f.BFNEXT = f.BFDONE = f.ARNEXT = f.ARDONE = 0;
    f.STAMP = f.ASSIGN;
    f.ASSIGN++;                                        // release last
    return true;
}

// Run the flow to completion (or detect a stall). Returns true if ALL super-tiles drained AND all waves
// retired (occ0==0 via the normal path). `max_sweeps` bounds it so a real deadlock returns false, not hangs.
static bool run_flow(Flow f, std::vector<Wave> waves, uint32_t max_sweeps = 100000) {
    for (auto& w : waves) f.occ0++;                    // every wave live++ at entry
    for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool progress = false;
        // FIX(b): a bounded shrink spin lets a stuck completer ESCAPE after shrink_bound sweeps.
        for (auto& w : waves) {
            if (w.stuck && f.shrink_bound > 0 && (sweep - w.stuck_sweep) >= f.shrink_bound) {
                if (f.escape_does_drain && f.DRAIN == f.STAMP && f.DRAIN < f.SUPER)
                    f.DRAIN++;                          // bounded-shrink-that-continues: still do the owed DRAIN CAS
                w.stuck = false; w.retired = true; f.occ0--;  // then retire via .Lflow_retire
                progress = true;
            }
        }
        // coordinator assign duty first (wid0)
        if (coord_assign(f)) progress = true;
        // every non-retired wave takes one action in its role, with the free coast fallback
        for (auto& w : waves) {
            if (w.retired) continue;
            bool did = false;
            if (w.role == COMPUTE) {
                did = compute_step(f, w, sweep);
                if (!did) did = feed_step(f, w, BFEED);   // compute->feed coast (free, lean): help staging
                if (!did) did = feed_step(f, w, AFEED);
            } else {
                did = feed_step(f, w, w.role);
                if (!did) did = feed_step(f, w, w.role == AFEED ? BFEED : AFEED); // feed->feed help
                if (!did) did = compute_step(f, w, sweep); // a lean feed only helps the multi-observer DRAIN advance
            }
            if (did) progress = true;
        }
        // retirement: once all work is drained, non-stuck waves retire (occ0--). Normal RETIRE broadcast.
        if (f.DRAIN >= f.SUPER) {
            for (auto& w : waves)
                if (!w.retired && !w.stuck) { w.retired = true; f.occ0--; progress = true; }
        }
        bool all_retired = true; for (auto& w : waves) all_retired &= w.retired;
        if (f.DRAIN >= f.SUPER && f.occ0 == 0 && all_retired) return true;   // clean completion (host sees occ0==0)
        if (!progress) {
            // no state change this sweep. That's a genuine deadlock ONLY if nothing will ever change again:
            //   a stuck wave with a BOUNDED shrink is still spinning toward its escape (progress next sweep),
            //   so it is NOT a deadlock. An unbounded shrink (shrink_bound==0) never escapes -> real stall.
            bool pending_escape = false;
            if (f.shrink_bound > 0) for (auto& w : waves) if (w.stuck) pending_escape = true;
            if (!pending_escape) return false;          // genuine stall (deadlock)
        }
    }
    return false;                                       // hit the sweep bound -> treat as stall
}

static std::vector<Wave> seed_waves(uint32_t WAVES) {
    // seed per occ_kernel_dsws_flow.s :1470-1484: wid0 BFEED(coordinator), wid1 AFEED, wid2 BFEED, rest COMPUTE
    std::vector<Wave> ws;
    for (uint32_t i = 0; i < WAVES; ++i) {
        Role r = (i == 0) ? BFEED : (i == 1) ? AFEED : (i == 2) ? BFEED : COMPUTE;
        ws.push_back(Wave{(int)i, r});
    }
    return ws;
}

int main() {
    const uint32_t WAVES = 16;
    // group-split diag geometry: TILES small, GROUPS=2, n_kseg small -> SUPER super-tiles.
    auto mkflow = [&](uint32_t SUPER) {
        Flow f; f.SUPER = SUPER; f.G = 6; f.FN = 4; f.ACC_N = 3; return f;
    };

    // ---- 1. BASELINE: current source (whole-tile A staging, unique-completer DRAIN advance) COMPLETES ----
    {
        Flow f = mkflow(/*SUPER=*/8);                  // e.g. TILES=1 * GROUPS=2 * n_kseg=4
        assert(run_flow(f, seed_waves(WAVES)) && "baseline group-split flow must complete");
        printf("flow_model: 1. baseline group-split (GROUPS=2, POOL_N=1) COMPLETES\n");
    }

    // ---- 2. Grok #2.4 UNIQUE-COMPLETER HAZARD: completer enters an UNBOUNDED shrink spin before its DRAIN
    //         CAS. Under current source (unbounded shrink, single-writer DRAIN) => permanent stall. ----
    {
        Flow f = mkflow(8); f.inject_completer_stuck_at = 3;   // shrink_bound=0 (unbounded, current source)
        assert(!run_flow(f, seed_waves(WAVES)) && "unbounded stuck completer MUST hang");
        printf("flow_model: 2. completer stuck in UNBOUNDED shrink @tile3 -> HANGS (Grok #2.4)\n");
    }

    // ---- 3. FIX(b) minimal: bound the shrink so the completer ESCAPES and still performs its own DRAIN CAS,
    //         then retires. This ALONE fixes it — DRAIN advances and occ0 reaches 0. ----
    {
        Flow f = mkflow(8); f.inject_completer_stuck_at = 3;
        f.shrink_bound = 5; f.escape_does_drain = true;
        assert(run_flow(f, seed_waves(WAVES)) && "bounded shrink that still does DRAIN CAS must complete");
        printf("flow_model: 3. + bounded shrink that completes its DRAIN CAS -> COMPLETES (minimal fix)\n");
    }

    // ---- 4. SUBTLE: a DEADMAN escape that force-retires WITHOUT the DRAIN CAS, with single-writer DRAIN,
    //         does NOT fix it — DRAIN stays frozen so the pipeline can't drain the rest. (The model's key
    //         non-obvious result: "bound the spin" is insufficient if the escape skips the owed DRAIN CAS.) ----
    {
        Flow f = mkflow(8); f.inject_completer_stuck_at = 3;
        f.shrink_bound = 5; f.escape_does_drain = false;       // deadman force-retire, no DRAIN CAS
        assert(!run_flow(f, seed_waves(WAVES)) && "deadman-retire without DRAIN CAS (single-writer) MUST still hang");
        printf("flow_model: 4. + deadman escape WITHOUT DRAIN CAS (single-writer DRAIN) -> STILL HANGS\n");
    }

    // ---- 5. FIX(a)+(b): deadman escape (no DRAIN CAS) BUT any observer may advance DRAIN => completes.
    //         So the two viable fixes are: (3) bounded-shrink-that-continues, OR (5) deadman + multi-observer DRAIN. ----
    {
        Flow f = mkflow(8); f.inject_completer_stuck_at = 3;
        f.shrink_bound = 5; f.escape_does_drain = false; f.multi_observer_drain = true;
        assert(run_flow(f, seed_waves(WAVES)) && "deadman + multi-observer DRAIN must complete");
        printf("flow_model: 5. + deadman escape + multi-observer DRAIN advance -> COMPLETES\n");
    }

    // ---- 6. Grok #2.2 HALF-APPLIED A-STAGING TRAP: stage only ACC_N A rows while STAGE gate still wants G. ----
    {
        Flow f = mkflow(8); f.stage_gate_accn = true;
        assert(!run_flow(f, seed_waves(WAVES)) && "half-applied A-staging (claim ACC_N, gate G) MUST deadlock");
        printf("flow_model: 6. half-applied A-staging (claim ACC_N, gate G) -> HANGS (Grok #2.2 trap)\n");
    }

    // ---- 7. STRESS: many super-tiles, current source, must still complete (no accidental frontier stall) ----
    {
        Flow f = mkflow(64);
        assert(run_flow(f, seed_waves(WAVES)) && "64-super-tile group-split run must complete");
        printf("flow_model: 7. 64-super-tile stress COMPLETES\n");
    }

    printf("flow_model: ALL PASS\n");
    printf("  VERDICT: the unique-completer + unbounded shrink is a real deadlock class; the fix must either\n");
    printf("           (a) bound the shrink AND still perform the owed DRAIN CAS on escape, or\n");
    printf("           (b) bound the shrink (deadman) AND make DRAIN advance multi-observer. Bounding alone is NOT enough.\n");
    return 0;
}
