Phase B write access verified; session started 2026-07-22.

Scope: offline only. No GPU executable, dispatch script, or HIP/ROCm program was run. No files outside this spike directory were edited, staged, or committed.

Build profile used for every SHA below (the recorded A1/A0 profile):
`/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=1 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=256 -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,POOL_N=1 -Wa,-defsym,ACC_N=3 -Wa,-defsym,WOFLUSH=0 -Wa,-defsym,WAVES=30 -Wa,-defsym,VBUDGET=1536 -Wa,-defsym,PHASEPROBE=0 -Wa,-defsym,PHSHIFT=8 -Wa,-defsym,PHSPLIT=0 -Wa,-defsym,NOCFLUSH=0 -Wa,-defsym,KMAJOR=0 -Wa,-defsym,JDEPTH=1 -Wa,-defsym,STAGGER=1 -Wa,-defsym,MAXFAT=0 -Wa,-defsym,STAGERS=4 -Wa,-defsym,DUTYPROBE=0 -Wa,-defsym,NTLOAD=0 -Wa,-defsym,RBU=1 -Wa,-defsym,NOFEED=0 -Wa,-defsym,MULTISLOT=0 -Wa,-defsym,MSCOMP=0 -Wa,-defsym,MSSCAN=0 -Wa,-defsym,MSDRAIN=0 -Wa,-defsym,MSFEED=0 -Wa,-defsym,BATCHASN=0 -Wa,-defsym,DECENTASN=1 -Wa,-defsym,SELFSERVE=1 -Wa,-defsym,SSWIN=32 -Wa,-defsym,PHIST=0 -Wa,-defsym,NOBLOAD=0 -Wa,-defsym,NODSADD=0 -Wa,-defsym,NOWMMA=0 -Wa,-defsym,BNDPROBE=0 -Wa,-defsym,RESVPROBE=0 -Wa,-defsym,BATCH=1 -Wa,-defsym,INITBAR=1 -Wa,-defsym,TERMFIX=1 -Wa,-defsym,DUTY_EVERY=64 -Wa,-defsym,CSTORE=0 -Wa,-defsym,SLEEPN=2 -Wa,-defsym,COORD_PERIOD=64 -Wa,-defsym,TFPROBE=1 -Wa,-defsym,DEADMAN=1 -Wa,-defsym,DEADMAN_TICKS=50000000 -Wa,-defsym,STAGINSTR=1 -Wa,-defsym,CNTLEAN=0 -Wa,-defsym,SPANFLIP=0 -Wa,-defsym,TRACE=0 -Wa,-defsym,FORENSICS=0 -Wa,-defsym,BANKZERO=1 -Wa,-defsym,FATGAUGE=0 -Wa,-defsym,BATONGATE=1`, plus `-Wa,-defsym,DSWS2_CONV=<0|1> -Wa,-defsym,CFASSIGN=<0|1> -c occ_kernel_dsws_flow.s` and `llvm-objcopy -O binary --only-section=.text`.

Task A - landed

- Source audit: `occ_kernel_dsws_flow.s:3292-3312` reads `ROLE[wid]` at dispatch; `:4074-4110` derives a CFASSIGN reservation only from the fixed `(cohort,wid)` pair and preserves the served cohort end in `s15`. The initial role split is at `:3025-3040`.
- Determination: the present flow kernel has no validated per-super-tile conversion/snapshot protocol to prove that this fixed-wid CFASSIGN reservation remains safe across a role change. Phase B bring-up is therefore A0: `CFASSIGN=0`, baseline `128500f7314cafce`.
- Changed `occ_kernel_dsws_flow.s:980-982`: added a fail-closed `.error` for `CFASSIGN && DSWS2_CONV`.
- Post-edit SHA command: the build profile above with `DSWS2_CONV=0 CFASSIGN=1` printed `cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553`; with `DSWS2_CONV=0 CFASSIGN=0` printed `128500f7314cafce9f1099d6ec6eaa2c348c406f77f07c16c79f7dfbddf73c9b`.
- Guard command: the same profile with `DSWS2_CONV=1 CFASSIGN=1` failed exactly with `occ_kernel_dsws_flow.s:981:3: error: CFASSIGN with DSWS2_CONV is not validated: conversion changes roles while counter-free cohorts are keyed by fixed wid. Use CFASSIGN=0 for Phase B bring-up.`

Task B - blocked, no unsafe wiring added

- Exact source mapping disproved the proposed insertion point: `occ_kernel_dsws_flow.s:4074` is the repeated decentralized-assignment peek, and all retry/boundary/window failures branch to `.Lflow_feedmt_sleep` (`:4080`, `:4105`, `:4111`, `:4159`, `:4222`, `:4348`). `.Lflow_feedmt_sleep` then sleeps and branches to `.Lflow_loop` (`:4842-4860`). It is a per-retry/coast park site, not a per-super-tile terminal bail.
- Macro census command `for m in occ_sample try_gate conv_apply conv_dec_floor reserve_try epoch_mark gq_reset gq_bump gq_read; do grep -nE "^[[:space:]]+$m\\b" ...; done` output: `occ_sample=0`, `try_gate=0`, `conv_apply=0`, `conv_dec_floor=1`, `reserve_try=1`, `epoch_mark=0`, `gq_reset=0`, `gq_bump=0`, `gq_read=0`. The lone `conv_dec_floor` and `reserve_try` uses are inside `conv_apply`.
- Open question: provide a source-verified once-per-super-tile follower rendezvous/terminal-bail site for the flow pipeline before adding any decision call. Wiring at feedmt would violate the no-new-traffic-in-coast/bail-spin rule.

Task C - blocked, no unsafe commit added

- `conv_apply` is defined at `occ_kernel_dsws_flow.s:2101-2159`, but has zero external invocations. It requires a terminal-bail ordering before a per-super-tile quiesce bump; the live flow has no such bump. The only active `QUIESCE_CNT_OFF` increment is the dispatch-retirement barrier at `:4973-4984`, after `ROLE_RETIRE` broadcast, not a super-tile boundary.
- No role writeback or conversion commit was added because doing so before the existing retry/loop path would make role population mutate mid-super-tile.

Task D - blocked, missing prerequisite state machine

- The required mutable role slots and controller state are only declarations/macros: `NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT/GATE_OFF/VRESV_OFF/SEGCNT_OFF` at `:381-386`; snapshots and `QUIESCE_CNT_OFF` at `:760-763`. The flow initializer at `:3009-3040` initializes heads, optional retirement counter, and ROLE mailboxes, but not the role-count/gate/reservation/segment controller state.
- `gq_reset`, `gq_bump`, and `gq_read` are definitions at `:1859-1891` with zero invocations. Thus there is no existing flow-kernel gq-based snapshot reconciliation to connect safely. The required DIAG N-1 cross-check cannot be truthfully wired without first supplying that missing per-super-tile protocol.

Task E - partially green; not complete

- Regression SHA gates are green after the kernel edit: A1 CONV=0 is `cac3ff7c2338e73f...`; A0 CONV=0 is `128500f7314cafce...`.
- INERT gate is green only for the selected A0 bring-up profile: `DSWS2_CONV=1 DSWS2_FORCE=0 CFASSIGN=0` assembled to `128500f7314cafce9f1099d6ec6eaa2c348c406f77f07c16c79f7dfbddf73c9b`, and `cmp` printed `A0 CONV=1 inert == CONV=0`. Because Tasks B-C are blocked, no decision can fire in this build; this is the pre-wiring inert proof, not a completed wired-path gate.
- CPU control-model gate: `g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/t && /tmp/t` printed `dsws_ctrl_model: dispatch/cooldown/pool OK`, `entry-contract (Pool-T7 repro) OK`, `envelope invariant OK`, and `ALL PASS`.
- RGA gate is NOT green: `KSRC=occ_kernel_dsws_flow.s ./rga_check.sh phaseb_conv1_a0 ...` failed before stats with `Failed opening file /home/kmbandy/.rga/GPUOpen/rga/rga_cli41-20260722-184125.log for writing: Read-only file system`, followed by `cat: 'rga_out/phaseb_conv1_a0/*stats*.csv': No such file or directory`. No spill count is claimed.
