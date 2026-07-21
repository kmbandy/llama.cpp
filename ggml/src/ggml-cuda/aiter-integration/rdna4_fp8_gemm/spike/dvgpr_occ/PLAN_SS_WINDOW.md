# PLAN — decouple the SELFSERVE reservation window from the operand-pool depth (2026-07-19)

## Goal

Under `SELFSERVE=1` the carry-through wave self-loads its operands from L2/L3 and **never touches the
operand pool**. Yet the reservation gate still rations reservations as if it did:

```
occ_kernel_dsws_flow.s:3599
    s_sub_u32 s47, s44, s45        // r - d  (ASSIGN - DRAIN)
    s_cmp_ge_u32 s47, POOL_N
    s_cbranch_scc1 .Lflow_feedmt_sleep
```

At the config of record `POOL_N=1`, so **one outstanding super-tile per workgroup** — 64 carriers
across 1920 waves. Measured consequence (run #9): TF=15.6 at 5.1% of peak, `grow-fail=0`,
`occ[98] baton=0`, i.e. the VGPR budget never binds and the dyn-VGPR moat has never been exercised.
Reserve→drain cycle is ~6.4us per WG, which is exposed memory latency with no occupancy to hide it.

`POOL_N` is pinned at 1 because it sizes the **operand pool** (`OP_BASE + POOL_N*OPSTRIDE`), which is
LDS-expensive. The **slot control block** is `SLOTC_STRIDE` = 32 bytes. This task separates the two:
keep the operand pool at `POOL_N` for the ring fast-path, and let carry-through reservations run
`SSWIN` deep against a wider array of cheap control blocks.

Expected fingerprint if correct (from `DSWS_SELFSERVE_DESIGN.md`): `grow-fail` 0 -> large,
`occ[98]` baton > 0, TF > 15.6. If `grow-fail` stays 0 the hypothesis is wrong — report that, do not
chase it further.

## Scope

One file: `occ_kernel_dsws_flow.s`. New defsym `SSWIN`, **default = `POOL_N`** so the current build is
bit-identical until it is raised. `SSWIN` only affects the `SELFSERVE` path. **No GPU execution** —
offline assembly only. Host `occ_dispatch.cpp` needs no change.

## What has to happen

- The reservation gate must bound `ASSIGN - DRAIN` by `SSWIN` instead of `POOL_N` **on the carry-through
  path only**. The ring path's relationship to the operand pool is unchanged.
- The slot control blocks must be `SSWIN` deep (`SLOTC_STRIDE` each) rather than `POOL_N` deep, and
  `slot_of` must index that wider array for carry-through. The operand pool stays `POOL_N` deep.
- The LDS layout guard at `:701` must account for the wider control array and still refuse >65536B.

## Caveats and nuances (not derivable from the code — these each cost a real GPU run)

- **The window is load-bearing for BOUNDARY SAFETY, not just flow control.** The comment under the
  reservation CAS states the reserve is sound *because* holding a slot keeps `DRAIN < ASSIGN`, so no
  tile/group boundary can fire underneath it and `DA_TILE`/`DA_BASE` stay frozen. Widening the window
  lets more reservations straddle a boundary. The `ZLOCK` check and `DA_ZDONE` (bank-zero) interlock
  must still hold at `SSWIN > 1`. **This is the hard part of the task — treat it as the primary risk,
  and if the interlock cannot be preserved, say so rather than weakening it.**
- **`GSTORED`, not `DRAIN`, is the real bank-reuse barrier.** The boundary gates on BOTH
  `DRAIN >= ASSIGN` and `GSTORED >= (z >> shift)`. Do not substitute one for the other.
- **Never touch coast->feed.** A wave that cannot compute goes and stages for others. Disabling it
  turned ~1900 waves into an idle spin earlier in this project.
- **No caps, no dams.** Do not add or re-enable `MAXFAT`/`FATTOK`. `BATONGATE=1` is the baton and stays.
  The only intended throttle is the physical `s_alloc_vgpr` grow-fail — the whole point of this task is
  to let that finally bind.
- **One driver only.** Do not add a second claim counter, tile claim, or group roll. DECENTASN owns the
  tile/group lifecycle.
- **`s49` is the reserved `exec_lo` save** for every `lds_*` write/atomic macro. Never hold a live value
  there across one.
- **To find a free SGPR, grep the `.set` table, NEVER the register spelling.** `s99`/`s101` look free but
  are `FATHELD` and `DM_PROG` via aliases; writing them corrupted the fat-token flag and disabled the
  deadman earlier today. `s103` is the only genuinely free SGPR.
- **Instrumentation cost scales with TIMES EXECUTED, not call sites.** Do not add eager `flow_gauge`
  atomics on a per-work-item path; that cost 63s of a 66s run today. Use SALU `cnt_inc` + retire emit.
- **`s_alloc_vgpr` does not drain VMEM stores.** Drain before every reallocation.
- **Exec-masked atomics must target the first ACTIVE lane** and the result read while that mask is
  installed.
- `POOL_N=1` stays fixed for the OPERAND pool at the config of record — do not widen it to solve this.

## Gates (all offline, no GPU)

Config of record: `WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1
BANKZERO=1 STAGGER=1 SELFSERVE=1 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1`.

- `SELFSERVE=0` byte-identical to the canonical bin, sha256 prefix `43beb08264e0c1d0`.
- `SSWIN` unset (== `POOL_N`) is **bit-identical** to the current build, sha256 prefix `be1bb047632e57c9`.
  This is the proof the change is inert until switched on.
- `SSWIN=8` assembles with **zero** scratch/spill instructions, and reports LDS < 65536B.
- `SSWIN=16` either assembles or fails on the LDS guard with a clear message — not silently.
- `KMAJOR=1` and `DUTYPROBE=1` must still refuse to assemble.

Build note: the sandbox filesystem may be read-only — do not run `build_flow.sh` (it writes a `.o` and
`/tmp/flow_build.err`). Reproduce its clang assembler invocation with output to `/dev/null`.
