# PLAN — SELFSERVE as carry-through on DECENTASN (2026-07-19)

## Goal

Self-serve is **not a new mechanism**. It is decentralized-assign (DECENTASN) declining a pointless
handoff. Today a wave that wins a reservation stamps a ring slot and walks away, leaving a feeder to
stage it and a compute wave to claim it. At the winning config the ring is **one slot** (POOL_N=1,
LDS-bound, cannot be widened) and during heavy work that slot cannot be kept full — so the handoff
buys nothing. Instead: **if there is no work in the ring, the assigning wave carries its reservation
all the way through compute itself**, self-loading operands from L2/L3 rather than waiting for LDS
staging. Prestaging the ring stays a **slack-time activity only**.

Priority ladder for a wave (this is the architecture, do not alter it):
1. work staged in the ring -> take it, compute it (existing path, unchanged)
2. ring empty -> reserve via DECENTASN and **carry it through compute yourself** (this task)
3. work has slowed / spare capacity -> refill the ring (existing feed path, unchanged)

## Scope

One file: `occ_kernel_dsws_flow.s`. All new code gated `.if SELFSERVE` (new defsym, default 0).
`occ_dispatch.cpp` already has forensics prints; you may ignore or reuse them, but no host change is
required. **No GPU execution.** Offline assembly only.

## The seam

`.Lflow_da_stamp` (the DECENTASN reservation stamp) currently ends `s_branch .Lflow_loop`. That branch
is the handoff. Carry-through replaces it: the wave already holds the decoded reservation
(tile / ksi / group, via the coupled cursor), so it proceeds to compute rather than returning.

## What the carry-through wave must do

- Self-load its A and B operands from global into the WMMA fragment registers, then WMMA, then
  `ds_add` into the shared per-group accumulator bank, for each of the reservation's rowblks.
- Bump the per-group completion counter so the existing write-once C completer fires normally.
- **Settle the slot as pre-completed** so both frontiers advance without a feeder or a ring computer
  ever touching it. There is already an idiom for exactly this in `.Lflow_da_rollback` ("publish a
  pre-completed sentinel so no consumer wedges") — follow it. This settle is forced, not a choice:
  `STAGE` only walks past a slot whose generation matches and whose pending bits are clear, and
  `DRAIN` only past one whose done-count has reached its target.

## Reference for the compute body (do not re-derive)

A previously-written, offline-verified body exists at
`/tmp/claude-1000/-home-kmbandy/81643b17-4487-4d56-9a56-666cde724d1a/scratchpad/kernel_pre_carrythrough.s`
(look for `.Lflow_selfserve`). Its **operand-addressing and WMMA/reduce section is correct and was
proven on hardware** — 96 tiles computed, all closers accounted for. Lift that approach:
- B addressing mirrors `BSTAGE_R`, A addressing mirrors `ASTAGE_R`, including the 64-bit B offset fix.
- The staging store and the compute load are both `v9`-based (identity per-lane), so the lane->data
  transform lives entirely at load time: B via the transpose load on `v9`, A via the plain load on
  `v8`, straight into the fragment registers, **no LDS round-trip**.
- The WMMA burst and the banked `ds_add` are unchanged from the ring's compute body.
**Discard everything else in that file** — its parallel claim counter, its own tile-claim, its own
group/tile boundary and terminal paths. Those are the bug being removed.

## Caveats and nuances (not derivable from the code — these cost real GPU runs today)

- **Never touch coast->feed.** A wave that cannot compute goes and stages for others; that is the glue
  of the economy and the only thing that puts enough waves in flight for the VGPR budget to bind.
  Disabling it turned ~1900 waves into an idle spin and recreated a failure this project already fought.
- **One driver only.** Do NOT add a second claim counter, a second tile claim (`occ[20]`), or a second
  group roll. DECENTASN owns the tile/group lifecycle; carry-through is a consumer inside it.
- **No caps, no dams.** Do not add or re-enable `MAXFAT`/`FATTOK` admission gating. `BATONGATE=1` (the
  baton) is correct and stays; `BATONGATE=0` reinstates an old dam that was deliberately removed.
  The only throttle is the physical grow-fail -> coast.
- **Grow before claim.** A grow that fails *after* a claim drops the item and silently shortens the
  work count. Order matters.
- **`s_alloc_vgpr` does not drain VMEM stores.** Drain before every reallocation. Also: a drain placed
  *before a call* is worthless if the callee itself emits — check whether the callee's own drain is
  compiled out by a config knob.
- **Never add a watchdog or probe to a spin path without first walking the loop to its head** and
  enumerating what message-bus/store traffic is already there. A redundant watchdog in the coast spin
  doubled REALTIME message traffic and wedged the card.
- **`s49` is the reserved `exec_lo` save slot** for every LDS write/atomic macro. Never keep a live
  value there across one.
- **Exec-masked atomics must target the first *active* lane**, not physical lane 0, and the result must
  be read while that mask is installed — otherwise a skipped atomic yields a stale return.
- **`n_kseg` is runtime** (`KT/SEGK`). Any bound involving it must be a runtime check, and must not
  compute a product that can overflow 32 bits before the comparison.
- **POOL_N=1 is fixed** at the winning config and cannot be widened (LDS). Do not design around a
  deeper ring.
- Scope: `JDEPTH=1`, `KMAJOR=0` (self-serve builds a tile-major sti; the K-major decoder disagrees),
  `BANKZERO=1`, `DECENTASN=1`.

## Gates (all offline)

- `SELFSERVE=0` byte-identical to the canonical bin, sha256 prefix `43beb08264e0c1d0`.
- `SELFSERVE=1` assembles with **zero** scratch/spill instructions.
- A `POOL_N=2 SEGK=128` variant also assembles.
- `KMAJOR=1` must refuse to assemble.
- Config of record: `WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 DECENTASN=1 BANKZERO=1 STAGGER=1`.

Build note: the sandbox filesystem may be read-only — do not run `build_flow.sh` (it creates a `.o`
and `/tmp/flow_build.err`). Reproduce its clang assembler invocation with output to `/dev/null`/stdout.
