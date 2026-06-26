# MAD-305 dyn-coop — MORNING PICKUP

## ★ END OF DAY 2026-06-26 — RESUME POINTER (read this first; full detail in KG + SPEC_WAVESPEC.md)
**The day's arc: "the GPU bricks every time we touch it" → a named, moat-bearing architecture (DSWS v2) + the
infra to build it safely.** Three things SHIPPED + CONFIRMED on silicon, then a big architecture reframe:
1. **POOLTERM** — fixed the pool≥2 teardown brick (root cause = pool=1-only count terminal; fix = compute exits
   on the feed's per-WG terminal-ti broadcast). pool=2 now retires clean, fence FIRED, NO brick. KG `dac0bb8c`/`0a2cea44`.
2. **Compositor-safe chunking** (`occ_dispatch.cpp` `ML8_COOP_CHUNK`) — bounded dispatches + yield between →
   real-shape ml8 GEMM runs IMPERCEPTIBLY on the R9700 that drives the desktop. kmbandy: "I didn't even notice
   it was running." The "perf bricks the desktop" wall is GONE. It's a PRODUCT requirement (single-GPU users),
   not a workaround. KG `21827908`. (occ_dispatch rebuilt 12:38; POOLTERM bin = occ_coop_2x4_p1_r2_b1_d1_POOLTERM_gd.bin.)
3. **The architecture: DSWS v2 = Adaptive Wave-Role Economy** (KG `63583120`, SPEC_WAVESPEC.md "DSWS v2" section).
   Corrects the morning's POOLTERM occupancy framing AND the v1 "split-K for occupancy." MEASURED (RESULT_WGGEMM.md):
   occupancy is NOT the wall (flat 4→8 waves); the wall is the **VALU issue port** (~31 non-WMMA : 32 WMMA → 52%
   of 307 TF ceiling). DSWS = ONE kernel that re-balances its mix of {compute(fat) / A-feed(lean) / B-feed(lean)}
   waves to the bottleneck at runtime, dyn-VGPR making the VGPR budget FUNGIBLE across roles (1 fat ↔ 3-7 lean),
   sensed by the ring counters we already have. Split-K = the headroom-creator, not occupancy. The MOAT is the
   **runtime adaptivity** (CUTLASS warp-spec + setmaxnreg already does static lean/fat on NVIDIA — verify the
   adaptive part is unclaimed).

**IMMEDIATE NEXT (post-compact, agreed with kmbandy):**
1. **Deep-research pass** — NVIDIA/AMD/arXiv: is anyone doing runtime-adaptive warp/wave-role rebalancing (vs
   CUTLASS STATIC warp-spec)? + mine CUTLASS `setmaxnreg` mechanics to copy not reinvent. Borrow if it exists;
   if not, we're first on the adaptive bit. (kmbandy: "happy to just borrow.")
2. **ml8-shape ground truth** — add a fast per-chunk hang-abort (current 25s → ~1.5s) so the pool sweep can't
   re-brick at the oversubscription point; then measure on `down` (M=2048 K=9216 N=2560): raw-fp8 baseline TF +
   `--att` issue-mix. The square-shape 161 TF / 31:32 ratio do NOT transfer; the 307/272 ceiling + issue-port
   mechanism do.
3. **DSWS v2 brainstorm → spec → plan → build** on the cooperative substrate.

**STANDING SAFETY (hard-won today, ~5 bricks): a GPU brick is a BUG not a tax.** Only sub-second bounded
dispatches are safe on the display GPU; the chunking enforces that. One gated dispatch at a time, every run
streams, never `--gl2c`. Do NOT move displays to the other GPU (eGPU 6900XT can't init pre-login AND it's not
representative of the single-GPU user we ship to — the kernel must coexist with the compositor, period).

---

## 2026-06-26 UPDATE — pool≥2 TEARDOWN BRICK: ✅ FIXED + CONFIRMED ON SILICON (no brick).
**RESULT (`coop_dyn_pool2_poolterm.log`, pool=2, both shapes):** `compPh=8` (was 7), `fence=FIRED` (was `--`),
clean teardown no WARN, `oracle CLEAN ok=256 bad=0`, exit 0, dmesg clean, **user-confirmed NO brick.** First
clean pool=2 dyn-VGPR cooperative run — correct AND terminates AND GPU survives. Blocker CLOSED. KG `0a2cea44`.
NEXT (gated, one dispatch at a time): R0 occupancy attribution (step pool up one at a time; trim NFV ~96), then
`ML8_P=2` (the real reuse lever). Details below.

### (root-cause + fix detail, for the record)
**The pool≥2 brick is NOT hardware contention and NOT a store-drain problem — both were disproven today.**
- **STOREWAIT refuted on silicon** (`coop_dyn_pool2_storewait.log`): fence still `--`, same late brick. The
  compute wave never reaches `s_endpgm`, so there was no store to drain.
- **§3.3.3.2 contention retracted**: two WGs can be on different SIMDs (each sole grower) yet the fence
  still never fires — placement-independent, so contention can't be it.
- **CONFIRMED CAUSE (in the kernel's own `.Lcompute_loop` comment): the compute terminal is a pool=1-only
  stub.** Compute exits when its tile COUNT (`s57`) hits TOTAL — valid only if ONE WG owns all TOTAL tiles.
  At pool≥2 the WGs SPLIT the tiles via the shared global atomic claim, so each compute gets <TOTAL, the
  count never reaches TOTAL, and after its last real tile the compute waits forever at `.Lwait_epoch` →
  WG never retires → dispatch never hits end-of-pipe → EOP `RELEASE_MEM` never fires → queue never IDLE →
  any reclaim wedges the GPU. Explains pool=1-clean/pool=2-brick, oracle-CLEAN-anyway, and `compPh=7≠8`.
- **FIX LANDED (gated behind `POOLTERM` defsym):** compute now exits on the feed's per-WG terminal broadcast
  — the feed writes `ti≥TOTAL` to LDS `TI_OFF` + bumps epoch before `.Lfeed_exit`; compute checks the RAW
  ti (before the SAFEPROBE clamp) and exits. Per-WG feed→compute rendezvous, correct for any WG count;
  count terminal kept as the pool=1 fast path.
- **Verified offline:** static d0 `POOLTERM=0` BYTE-IDENTICAL to `.clean_bins` (1716B); `DYNVGPR=1
  POOLTERM=1` assembles, disasm shows `s_cmp_ge_u32 s17,s11 → .Lcompute_exit` at the compute-loop top.
- **Run-ready staged bin:** `occ_coop_2x4_p1_r2_b1_d1_POOLTERM_gd.bin` (3136B; DYNVGPR=1 POOLTERM=1
  SAFEPROBE=1 DIAG=1). STOREWAIT is refuted — leave it OFF.
- **NEXT = ONE gated pool=2 run** with the POOLTERM bin (`cp occ_coop_2x4_p1_r2_b1_d1_POOLTERM_gd.bin
  occ_coop_2x4_p1_r2_b1_d1_gd.bin` then the usual gated `--mbml8coop` invocation). Predict `compPh=8`,
  `fence=FIRED`, clean teardown, NO brick. If fence still `--` → handshake hole (e.g. compute parked at
  `.Lwait_prod` when feed exits) → offline trace. After clean: R0 occupancy attribution, then `ML8_P=2`
  (the real reuse lever).
- **STANDING SAFETY, CORRECTED: a GPU brick is a BUG, not an "accepted tax."** Freeze dyn dispatch whenever
  the build is known to leave the queue non-idle. One gated dispatch at a time, no sweeps, every run streams.

---

## 2026-06-25 UPDATE — ROOT CAUSE FOUND & FIXED (offline). Awaiting SUPERVISED GPU rerun.
The "rendezvous deadlock" AND the "dead marks" were **ONE bug**: an **out-of-range (OOR) VGPR**
in the hand-written asm, NOT a dyn-vs-barrier HW incompatibility.
- gfx1201 dyn-armed waves LAUNCH with **only one 16-VGPR block backed = v0..v15**; RSRC1.VGPRS is
  ignored in dyn mode. The init/rendezvous + compute epoch-wait run PRE-grow (lean-16).
- The LDS macros used v27/v28/v29 and `mark` used v30/v31 → all OOR pre-grow. RDNA4 OOR rules:
  OOR LDS-addr aliases v0 → INITFLAG spin reads wrong addr, never sees 0xACED → **infinite spin =
  the deadlock**; OOR mem-source → atomics write nothing → **dead marks**. (Codex + RDNA4 ISA + disasm.)
- **FIX landed:** gated all pre-grow LDS/atomic temps to v11(addr)/v14(data) under DYNVGPR; static keeps
  v27..v31 verbatim. Static d0 **byte-identical** (1716B); dyn kernel has **zero** v27-v31 refs; d1 rebuilt
  instrumented DIAG (3248B). KG decision id 17f209af.
- **NEXT (supervised):** rerun fixed instrumented d1 at pool=1. SUCCESS = feed reaches claim (occ5>0),
  phases advance (occ6/occ7>0), marks (occ23..27) populate. Would mean multi-wave dyn coop is VIABLE.
- **Universal rule for dyn-VGPR hand-asm:** NO VGPR > v15 as a source/LDS-addr before the first s_alloc_vgpr.

---
## (original 2026-06-24 night note below — superseded by the root-cause above)

## TL;DR
The dyn-VGPR cooperative GEMM **deadlocks (not faults)** in its pre-grow init region,
**even at pool=1** (single 2-wave WG). Removing the `s_barrier` (BUSYWAIT) did NOT fix it.
Address clamps (SAFEPROBE) DID kill the earlier page-fault — so brick #1's fault and the
current hang are two separate failure modes; clamps peeled off the fault, exposing the hang.

## MORNING TASK #1 (offline, do FIRST): fix the broken `mark` instrumentation
- Symptom: ALL `mark`-based markers read 0, but real atomics work (live=1, maxlive=1,
  occ[2]=timer=542230338 set). occ[2] (timer, line 198) is AFTER mark 92 (line 190) in
  program order, so the leader PROVABLY passed mark 92 — yet occ[23]=0 ⇒ mark is dead.
- Root-cause lead (high confidence): the `mark` macro gates on **v2** (`v_cmp_eq 0, v2`);
  the WORKING atomics (admission/timer) gate on **v0**. Marks dead + v0-atomics alive ⇒
  **v2 is corrupted (nonzero for lane 0)** by the time marks run.
- ACTION: disasm occ_coop_2x4_p1_r2_b1_d1_gd.o; trace v2 from prologue `v_and_b32 v2,31,v0`
  to the first `mark`; find what clobbers v2 (suspect a recent edit reusing v2). Fix, rebuild,
  confirm static d0 stays byte-identical to .clean_bins, then the marks/stream are trustworthy.

## What the RELIABLE signals proved (real atomics, not marks)
- pool=1 run (coop_dyn_pool1.log): leader admitted (live=1), passed the timer (occ[2] set),
  NEVER reached claim (occ[5]=0). Wedged in [LDS-init / rendezvous / role-split / pre-claim].
- pool=64 run (coop_dyn_busywait.log + coop_dyn_initmark.log): live=64, all else 0. Same wedge.
- So: NOT multi-WG co-residency (pool=1 wedges too). NOT the s_barrier specifically
  (BUSYWAIT removed it, still hangs). It's the dyn pre-grow init rendezvous itself.

## Death certs (all recoverable MODE1 — GPU recovers to Hyprland safe-mode, no hard reboot needed)
- This session's coop hangs = ring gfx_0.0.0 timeout, fence frozen, MES can't evict,
  NO VM fault → pure DEADLOCK (matches wavespec BRICK #4 KG note).
- Earlier brick #1 (this morning) = TCP/permission PAGE FAULT at buffer edge → fixed by clamps.

## Instrumentation built this session (all gated, static d0 byte-identical to proven-green)
- occ STREAMING: harness snapshots occ→disk every 200ms during poll (ML8_COOP_STREAM=1 or any
  oracle). Survives MODE1. THIS IS THE KEY CAPABILITY — captures the wedge frame-by-frame.
- SAFEPROBE=1 (kernel): ti clamp [0,TOTAL-1] + v8/v9/v10 vaddr clamps → provably no OOB
  (Codex-reviewed). Killed the page-fault mode.
- BUSYWAIT=1 (kernel): replaced the one s_barrier with an LDS INITFLAG busy-wait. Did NOT fix
  the hang (so the rendezvous deadlocks regardless of barrier-vs-busywait).
- 512MB guard padding + C-tail canary + per-frag badmap (harness).
- DIAG fine markers occ[14..27] (BROKEN — see task #1).

## The strategic fork (deferred to morning)
Multi-wave dyn coop deadlocks at the cross-wave init rendezvous even at pool=1 — re-confirms
the 2026-06-22 96-agent deep-research verdict (multi-wave dyn fights the HW; ISA-supported dyn
expression = SINGLE-WAVE register-blocked L4 / task #324, zero cross-wave rendezvous).
User chose: fix instrumentation + rerun in the morning to learn more before deciding pivot.

## Build commands (P=1 FM=2 FN=4 RINGD=2 oracle)
- static d0 (must stay byte-identical to .clean_bins/occ_coop_2x4_p1_r2_b1_d0_gd.bin):
  clang ... -Wa,-defsym,FM=2,FN=4,P=1,RINGD=2,BATCH=1,DYNVGPR=0 -c occ_kernel_coop.s
- SAFE+instrumented dyn d1:
  clang ... DYNVGPR=1,DIAG=1,SAFEPROBE=1,BUSYWAIT=1 -c occ_kernel_coop.s -o ...d1_gd.o
- run: ML8_P=1 ML8_POOL=1 ML8_DYN=1 ML8_ORACLE_ONLY=1 ML8_COOP_NOFENCE=1 ML8_ONLY=down
       ML8_COOP_PAD_MB=512 ML8_COOP_STREAM=1 timeout 40 ./occ_dispatch --mbml8coop > LOG 2>&1

## SAFETY STANDING RULES (unchanged)
Gate EVERY GPU dispatch; warn first. R9700 brick → MODE1 → Hyprland safe-mode (recoverable,
user has been rebooting by choice). Bricks are accepted FOR DATA — every run must stream to disk.
NEVER --gl2c. Bounds gate + clamps + padding stay on. static-cooperative oracle = GREEN (the win).
