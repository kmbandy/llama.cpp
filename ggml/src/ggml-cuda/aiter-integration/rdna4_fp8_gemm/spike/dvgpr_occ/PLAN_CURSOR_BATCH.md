# 🗄️ SHELVED — CORRECT + VERIFIED, HANDLES A BOTTLENECK THAT ISN'T CURRENTLY BINDING (2026-07-20)

STATUS: a WORKING, VERIFIED lever, banked for when its bottleneck (reservation-CAS contention) binds. NOT a
  failed experiment. Default BATCH=1 is byte-identical to 1e78b027 (gate-proven), so it ships inert and flips
  on with one defsym the day reservation contention becomes the wall.

WHAT IT DOES + THAT IT WORKS: claims N consecutive ASSIGN indices per CAS (N=min(BATCH, DA_ZDONE-r,
  SSWIN-(r-d))), drains them serially one-ksi-per-burst. VERIFIED CORRECT on silicon at BATCH=2 and BATCH=4:
  computed=50331648 WORK-EXACT, BNDPROBE cursor CLOSES, ASSIGN!=z=0, oracle CLEAN, no latch. The mechanism is
  sound; the Grok impl + review held. A real tool in the box.

THE SCAN (fed chunk 12288, all WORK-EXACT / oracle CLEAN):
    BATCH | TF   | coast | window-full | concurrent holders (SSWIN/BATCH)
      1   | 21.0 | 58.1% |   6.3%      |  ~8
      2   | 20.5 | 89.9% |  92.2%      |  ~4
      4   | 16.0 | 96.9% |  96.6%      |  ~2

WHY IT DOESN'T HELP *TODAY* (the gift of running BATCH=2, not just BATCH=4): TF is FLAT while concurrent
  compute-holders >= ~4 (21.0 vs 20.5), then CLIFFS at ~2 (16.0). Throughput tracks CONCURRENCY, and ~4
  concurrent waves already SATURATE the pipeline. So the CURRENT wall is NOT admission / concurrency /
  reservation-CAS -- baseline already has more concurrency than it can use. Batching trades concurrency for
  fewer CAS ops, and we are above the concurrency knee, so it cannot lift TF *now*. That is a property of the
  CURRENT bottleneck, not a flaw in batching.

WHAT THE SCAN BOUGHT US: it RULED OUT the entire admission/concurrency axis as the current wall (compute
  already free via NOWMMA, operands free via NOBLOAD, and now concurrency shown sufficient). The 21 TF ceiling
  is PER-WAVE COORDINATION THROUGHPUT -- the fixed LDS-handshake / s_alloc_vgpr / ZLOCK-boundary cost each wave
  pays, which ~4 concurrent waves already max out. THAT is the next target.

WHEN TO UN-SHELVE BATCH: once per-wave coordination cost is cut and concurrency/wave-count scales up, the
  single-cursor reservation CAS (1.466 collisions/reserve, rising to 3.216 at BATCH=2) becomes the NEXT
  ceiling. Then batched reservation -- or its eager-stamp variant (stamp all N up front so queued indices stay
  computable in parallel, avoiding the window-starvation seen here) -- is the built-and-ready fix.

Below is the original plan, kept as the design record.

---

# PLAN — BATCHED RESERVATION: amortize the single-cursor CAS (2026-07-20, N4)

## Why (measured, not assumed)

RESVPROBE (run `s1_n4_resvprobe01`, fed, TF=21.0, WORK-EXACT, oracle CLEAN) split the
`.Lflow_da_peek` empties:

| signal | value |
|---|---|
| WIN (reservations) | 16777216 = TOTAL_super |
| CAS-loss | 24592252 → **1.466 collisions per successful reserve** |
| window-full bail | 3728949 → **6.3%** of empties |

The single `ASSIGN_HEAD` LDS word (`:3929` `lds_cas_rtn ASSIGN_HEAD_OFF, r → r+1`) is
the publication throttle: ~41M CAS attempts to place 16.8M reservations, 40% collide,
and the SSWIN=8 window almost never fills (6.3%) — the cursor CANNOT get ahead.
**NOT stage-bound. Cursor-contended.**

## The change, in one sentence

A wave that wins the reservation CAS claims **N consecutive indices in one bump**
(`ASSIGN_HEAD: r → r+N`) instead of 1, then processes them serially. CAS frequency
drops ~N×, so collisions collapse ~N×. **No cap, no wait, no blocking read added — the
wave just claims more per successful CAS. The river runs harder, not dammed.**

## N is bounded THREE ways (all already-present invariants — this adds no new gate)

`N = min( BATCH, DA_ZDONE - r, SSWIN - (r - DRAIN) )`, `N >= 1`.

1. **`BATCH`** — the tuning knob (defsym, default 1 = byte-identical to config of record).
   Start at 4, sweep to 8.
2. **`DA_ZDONE - r`** — **never reserve past the group/tile boundary.** This is THE
   correctness anchor: the existing peek already refuses `r >= DA_ZDONE` (`:3919`). A batch
   that stops at DA_ZDONE means the batch NEVER crosses a boundary, so the ZLOCK / bank-zero
   / DA_ZDONE interlock is **completely unchanged** — the over-reservation-past-DA_ZDONE bug
   (ASSIGN ahead of z) that we just spent two days killing CANNOT be reintroduced, because a
   batch is clamped to the same boundary the single-step path already respects.
3. **`SSWIN - (r - DRAIN)`** — never claim more than the control window can hold. The existing
   window-full bail (`:3923` `r-DRAIN >= SSWIN`) becomes "claim up to the remaining room."

## Duty-cycle safety (THE hard constraint — do NOT get this wrong)

**Reserving N does NOT mean computing N in one fat burst.** That would set effective
JDEPTH → N, extend TIME-AT-PEAK, drive duty → square wave, and kill the stagger. FORBIDDEN.

The wave processes its N-deep local backlog **one ksi per fat burst, shrinking to lean
between bursts**, exactly as JDEPTH=1 does today. Batching changes the RESERVATION
granularity only; the compute burst stays one ksi. `TIME AT PEAK ∝ JDEPTH*SEGK` is
untouched. The DUTYGUARD assembler check still holds and must still pass.

## What actually has to change (self-serve loop only, `occ_kernel_dsws_flow.s`)

- `.Lflow_da_peek`: compute N (the 3-way min), CAS `r → r+N` instead of `r+1`. On win, the
  wave holds indices `[r, r+N)`.
- The wave stamps + grows + stages + computes each held index **serially**, shrinking
  between, then returns to peek. Only ONE slot is live at a time from this wave (duty-safe);
  the other N-1 are reserved-but-this-wave's-to-stamp (no other wave can touch them — the
  cursor already moved past them).
- Slot mapping: index `r+i` maps to slot `(r+i) mod SLOT_N` via the existing `slot_of`. The
  batch must not alias two live indices onto the same slot → another reason N ≤ SSWIN.
- `BATCH` defsym + guards: `BATCH >= 1`; `BATCH == 1` byte-identical to `1e78b027`;
  `BATCH <= SSWIN` (else a batch could exceed the window / alias slots).

## Caveats (each is a scar)

- **THE RIVER: no dam.** N is a MAX-claim, never a wait. A wave that can only get N=1 (window
  nearly full, or next to a boundary) proceeds with 1 — it never spins for a bigger batch.
- **BOUNDARY UNCHANGED BY CONSTRUCTION.** The `DA_ZDONE - r` clamp is the whole safety story
  for the boundary. If review finds any path where a batch can straddle DA_ZDONE, STOP — that
  is the dropped-group race wearing a new hat.
- **DUTY CYCLE.** One ksi per burst, period. If the serial-drain of the backlog cannot be done
  without holding ACC across multiple ksi, report that instead of extending the window.
- **SELF-SERVE CONCENTRATION is acceptable HERE.** Fewer waves each doing N is fine: the run is
  58% coasting with grow-fail=0 (excess idle waves). But watch occ[1] occupancy and coast-frac
  — if concentration STARVES staging (feedMT rises, carrier-stall appears), that's a real cost.
- **WORK-EXACTNESS IS THE GATE.** computed must stay 50331648 EXACTLY. A batch that double-claims
  or skips an index shows here first. BNDPROBE=1 on bring-up (cursor accounting must still CLOSE,
  ASSIGN!=z must stay 0).
- **ZERO FREE SGPRs.** N, the backlog base, and the per-i cursor need registers that survive a
  compute burst. Audit `.set` aliases AND literal sNN AND s[a:b] ranges before claiming any free.

## Gates (offline, no GPU)

- `BATCH=1` **byte-identical** to `1e78b027` (proof it is inert until switched on).
- `BATCH=4` and `BATCH=8` assemble zero-spill; `gate_lds.sh` still agrees (LDS=54016, batch is
  control-flow only, no new LDS).
- `gate_duty.sh` still 10/10 (batching must NOT move J*SEGK).
- Disasm check: the CAS at `:3929` bumps by N, and N is clamped by DA_ZDONE and the window.

## Bring-up (ONE dispatch, then STOP — rule 2)

`BATCH=4` + `BNDPROBE=1` + `RESVPROBE=1`, fed chunk 12288. Expect: computed=50331648 WORK-EXACT,
oracle CLEAN, cursor accounting CLOSES, CAS-loss/reserve drops from 1.466 toward ~0.37 (÷4),
window-full stays low. If TF moves, the cursor was the wall. If CAS-loss drops but TF does NOT,
the wall is downstream (ZLOCK boundary stall or staging) — report that, do not tune around it.

## Falsification (set in advance)

- If CAS-loss/reserve does NOT drop ~N× at BATCH=N → the batch isn't taking effect (N clamped to
  1 everywhere?) → measure N's actual distribution, don't add machinery.
- If CAS-loss drops but span does not improve → cursor contention was NOT the binding constraint
  (a more interesting finding) → report; the next suspect is the ZLOCK boundary stall.
