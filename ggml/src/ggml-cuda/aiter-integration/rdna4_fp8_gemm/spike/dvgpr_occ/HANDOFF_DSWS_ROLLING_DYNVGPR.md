# HANDOFF — DSWS rolling dyn-VGPR (the sum-envelope grow economy)

**Written 2026-07-02** for a fresh session that will write the detailed spec + plan + implementation.
This doc gives you the **architecture** (agreed with kmbandy) and the full **state of the world** so you
can start cold. It deliberately stops short of the detailed transcription spec — that is YOUR first task.

Everything lives in `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/` unless noted.
The kernel under work is **`occ_kernel_dsws.s`** (the "v2 substrate", `--dsws2` path). Do **not** touch
`occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.*` — they are shared/not-ours siblings (leave dirty,
never stage).

---

## 0. TL;DR — what to do next

1. Read this doc, then the referenced repo docs (§7) and the SDD ledger (`.superpowers/sdd/progress.md`).
2. Write the **detailed design spec** for the rolling dyn-VGPR grow economy (§4 is the architecture; you
   turn it into exact LDS offsets, register assignments, macro definitions, asm placement, CPU-model
   interfaces). Use the brainstorming→writing-plans flow. Save to `docs/superpowers/specs/`.
3. Then plan (`docs/superpowers/plans/`) and implement, offline gates first, one supervised GPU oracle run.

**The one-line problem:** the DSWS compute waves currently grow to peak (`NFV=112`) via a bare
`s_alloc_vgpr` SCC-retry with **no sum-envelope and no stagger**, so all `NCOMP` compute waves grow in
lockstep and collide on the shared dyn-VGPR pool → intermittent forward-progress hang (ISA §3.3.3.2).
**The fix:** route every compute grow through the shared `vgpr_reserved` sum-envelope + stagger the
grows, so peaks never all coincide and freed budget flows to whatever role the bottleneck needs.

---

## 1. Where we are (git + disk state, 2026-07-02)

- **HEAD = `a7212a605`** on branch `feat/dsws-phaseb-conversion` (base `66eec7f54`).
- Relevant commits (newest first):
  - `a7212a605` docs(dsws): correct entry != re-dispatch in pool-economy spec + plan
  - `b01c722df` **fix(dsws): route first-time wave entry through _alloc/_init** — the Pool-T7 *brick* fix
    (see §3). Message overstates it as "the Pool-T7 brick" fix: it fixed the desktop-**brick** failure mode
    but NOT the residual hang. Leave the commit; the record is corrected in the ledger + KG.
  - `3dbb8c985` feat(dsws): pool-economy offline machinery (Tasks 1-6)
  - `5421751` feat(dsws): CPU model for Phase-B snapshot/quiesce
- **Uncommitted OURS (working tree):**
  - `occ_kernel_dsws.s` — the **wedge-frame DIAG instrumentation** (see §6). Byte-identity verified: at
    `DSWS2_CONV=0` and at `DSWS2_CONV=1/DIAG=0` the `.text` sha is IDENTICAL to HEAD (`e5ec5e50` /
    `e296b846`). The markers only materialize at `CONV=1 && DIAG=1`. **Safe to commit.**
  - `test_dsws_quiesce_race.cpp` (untracked) — the offline `std::thread` QUIESCE/epoch race harness (§5).
  - `lean_bounds_sim.cpp` (untracked) — pre-existing, not ours, ignore.
- **Installed bins:** all three `occ_dsws2_{4c2a2b,6c1a1b,2c3a3b}_gd.bin` restored to the **safe CONV=0
  4840B** build (`e5ec5e50` / `bb8cb230` / `066b73a4`). No CONV=1 bin is installed (footgun rule).
- The 4 not-ours siblings are dirty; never stage them.

---

## 2. What DSWS is (one paragraph, so you have the frame)

**Dynamic-Split Wave-Specialization** — one persistent-wave fp8 GEMM kernel on gfx1201 (RDNA4/R9700,
wave32) that re-balances its mix of **{fat compute / lean A-feed / lean B-feed}** wave roles to the
runtime bottleneck, using raw-PM4-armed **dyn-VGPR** (`s_alloc_vgpr`, which HIP cannot express — the moat)
to make the per-SIMD VGPR budget **fungible across roles**. One WG = `WAVES` waves: wid0 = pinned
"claimer" (super-tile broadcaster + B-feed-class), the rest are followers in three roles. Coordination is
**barrier-free** (no `s_barrier`) via LDS atomics + epoch polling. Split-K decomposes K into `SEGK`-sized
segments accumulated into C via `global_atomic_add_f32` (ksi-independent C term). Role mix is written
`{NCOMP,NAFEED,NBFEED}` (e.g. `4c2a2b` = 4 compute, 2 A-feed, 2 B-feed). Phase-A = static substrate
(`DSWS2_CONV=0`, GPU-green). Phase-B = the adaptive conversion economy (`DSWS2_CONV=1`).

Read `MAD305_DSWS_MASTER.md` for the full vision; `SPEC_DSWS_SUBSTRATE_V2.md` for the substrate;
`SPEC_DSWS_PHASEB_CONVERSION.md` for the conversion economy.

---

## 3. How we got here — the debugging arc (so you don't re-walk it)

1. **Pool-T7 first GPU gate (CONV=1, conversions dormant) BRICKED** — MODE1 reset, rebooted the desktop
   (gfx1201 shares the graphics ring with Hyprland). CONV=0 was green across all mixes.
2. **Entry-fix (`b01c722df`)** — root-caused offline: the CONV=1 **seed block routed first-time wave entry
   through the `.Ldispatch` re-dispatch trampoline**, which lands on `_follow` and skips `_alloc`/`_init`
   (the `s_alloc_vgpr 32` handshake, the `INITFLAG==0xACED` rendezvous, and `s35=0`). Re-pointed the seed
   arms at the full role entry (`.Lcompute/.Lafeed/.Lbfeed`). This **fixed the desktop-BRICK failure mode**
   — confirmed on silicon (re-attempt #2: dmesg silent, chunk watchdog self-aborted cleanly at ~24.8s).
3. **BUT a residual non-bricking hang remained** — chunk 1 (8 super-tiles) completes, the *next* dispatch
   stalls at its first super-tile (`occ[20]` frozen, `occ[0]=1` claimer live). CONV=0 control ran clean
   (`ok=1536 bad=0`), so CONV=1-dormant has a real liveness regression.
4. **Offline analysis RULED OUT the protocol logic:** a faithful `std::thread` model of the claimer+
   followers QUIESCE/epoch handshake (`test_dsws_quiesce_race.cpp`) → **0/400 stalls** (seq_cst *and*
   relaxed). Codex static review (task-mr3u2ig0) independently found no source-level race. Snapshot LDS
   offsets don't collide; snapshot values are constant in the dormant case (so `s50 == CONV=0`);
   `lds_put/get/fetch_add` all `s_wait_dscnt` (ordering HW-enforced). Logic is sound.
5. **SKIPQ bisection BRICKED (do NOT repeat):** an uncommitted `DSWS2_DBG_SKIPQ` knob forced `s51=1` to
   advance on the snapshot sentinels alone. Dispatched → hard hang (~8 min) → brick. **Disabling the
   quiesce gate removes the chunk-watchdog's clean self-abort** and re-introduces the straggler bug the
   gate prevents. **Never disable the quiesce gate on the compositor card.**
6. **HEISENBUG → root cause:** adding the wedge-frame DIAG instrumentation (extra occ stores in the
   claimer poll loop + per-role markers) shifted timing and the CONV=1/DIAG=1 bin ran **CLEAN**
   (`ok=1536 bad=0`, all 4 chunks, dmesg silent) where the stock bin hung. A timing-sensitive race that
   instrumentation perturbs away ⇒ **not a logic bug**. kmbandy confirmed: it's the **known multi-grower
   dyn-VGPR collision** — all `NCOMP` compute waves grow to 112 in lockstep and collide.

**Root cause (CONFIRMED, KG `21cc0d63`):** `occ_kernel_dsws.s` `.Lcompute_grow` is a **bare
`s_alloc_vgpr NFV` SCC-retry with no sum-envelope reservation and no stagger**. The `vgpr_reserved`/`VRESV`
envelope exists but only gates **role-conversion** grows, not the per-rowblk compute grow. All compute
waves issue `s_alloc_vgpr` simultaneously → ISA §3.3.3.2 (HW deadlock-avoidance guarantees only *one*
wave/SIMD can reach max) → intermittent hang, worsened by the shared B-ring (`RINGD=2`) interaction.

---

## 4. THE ARCHITECTURE (the fix) — build the detailed spec from this

The design kmbandy and I converged on. **One pool, one law, split-K churn, stagger for saturation +
safety + fungibility.** This *unifies* the per-compute-wave grow with the role-conversion economy: they
are the same pool discipline, not two systems.

### 4.1 One pool, one invariant
A single per-SIMD VGPR budget `BUDGET`. The only governing law:
```
Σ instantaneous allocations over all resident waves  ≤  BUDGET   (at every instant)
```
Every grow — a compute wave taking a kseg, OR a feed wave converting to compute — reserves against the
**same** `vgpr_reserved` counter. VGPR is fungible without regard to who grows next. That fungibility IS
the adaptive substrate.

### 4.2 Real split-K (n_kseg > 1) makes the pool churn
Old full-K behavior: a compute wave grabs its fp32 accumulators and camps at peak for ~95% of the K-loop
(square wave — this is the `MAD305_DSWS_MASTER.md:78-81` "staggering buys nothing" case). With **real
split-K** each compute wave: **grow → compute ONE kseg → flush that partial to C via
`global_atomic_add_f32` → shrink to lean IMMEDIATELY → loop**. No accumulators persist across a wave's
ksegs (the accumulate-into-C path split-K already uses across waves handles the sum), so peak is a *brief*
window, duty drops, and the pool has genuine free headroom cycling through it. Split-K is **already
plumbed** in the substrate: `n_kseg` is derived in-kernel from `KT` and `KSEG_STEPS=SEGK/16`; `DECODE_STI`
splits `sti → (t, ksi)` and handles `n_kseg=1` for free (`shift=0, mask=0`). We've only ever run
`n_kseg=1` (degenerate high-duty). The real thing runs `n_kseg > 1`.

### 4.3 Every grow through the shared sum-envelope (the collision fix)
Replace the bare `.Lcompute_grow` SCC-retry with the **enveloped reserve→grow→…→shrink→release**, the
same primitive `conv_apply` already uses for role conversion:
```
reserve : old = atomic_add(vgpr_reserved, NFV-VLEAN); if old+(NFV-VLEAN) > BUDGET { atomic_sub; spin-at-lean; retry }
grow    : s_alloc_vgpr NFV        (SCC-retry stays as the hardware backstop)
burst   : one kseg of WMMA
flush   : global_atomic_add_f32 partials into C
shrink  : s_alloc_vgpr VLEAN
release : atomic_sub(vgpr_reserved, NFV-VLEAN)
```
Because the budget can never be exceeded, the "all NCOMP grow at once" collision is **unreachable by
construction** — only as many waves as fit can be at peak; the rest spin at lean until budget frees.
This is exactly the ISA §3.3.3.2 "software forward-progress scheme." Reserve/release run at lean-32
(v≤15, OOR-safe) — before the grow and after the shrink.

### 4.4 The stagger — three jobs at once
Phase-offset the compute waves' grow windows (lock-free LDS phase tokens, reuse the prod/cons counter
style — NO `s_barrier`) so:
1. **Compute stays saturated** — there is always *some* compute wave at peak doing WMMA.
2. **The collision can't recur** — grows don't all issue at the same instant (the envelope is the hard
   floor; the stagger is the scheduling that keeps peaks spread).
3. **Feed fungibility** — the immediate per-kseg release frees budget so **feed waves can spawn/grow the
   moment the ring counters say feed is the bottleneck.** The freed VGPR flows to wherever the current
   bottleneck is, per-moment — the adaptive core of DSWS.

**Master knob = burst length** (ksegs held at peak before shrink) and **stagger period / concurrent-peak
count**. These are the R3 sweep axes (`SPEC_WAVESPEC.md` R3). Short burst → more churn/occupancy but more
grow/shrink overhead; long burst → amortized overhead but higher duty. Optimum = shortest burst that keeps
the WMMA pipe full.

### 4.5 Instrumentation baked in from the start (kmbandy mandate)
"Build the real thing with the real instrumentation so if/when it breaks, the data pinpoints it — not a
blind brick." On top of the wedge-frame markers already in the tree (§6), add grow-envelope telemetry to
host-streamed occ slots: `vgpr_reserved` high-water, per-wave reserve-spin count (permit-wait),
grow/shrink/release event counts, per-role epoch. A wedge becomes a readout (permit-starvation vs
elsewhere), not a mystery. All DIAG-gated so `DIAG=0` stays byte-identical.

### 4.6 Validation ladder
CPU-model the envelope invariant (Σ ≤ BUDGET, no lost reservation, forward-progress under thread race —
extend `dsws_ctrl_model.cpp` + `test_dsws_ctrl_model.cpp`) → assemble matrix + RGA 0-spill → CONV=0 /
DIAG=0 byte-identity → **one** safeguarded GPU oracle run at `n_kseg>1` (`ok=1536 bad=0`) → then sweep
concurrent-peak × burst × stagger-period for occupancy/throughput.

---

## 5. How this lines up with the existing repo design (reconciliation)

kmbandy asked explicitly whether this matches what's already written. It does, **with one correction and
one addition** we made this session:

- **`SPEC_WAVESPEC.md` EXTENSION (2026-06-25), "rolling dyn-VGPR / traveling peak" (KG `1fdd5784`)** — the
  governing sum-envelope rule (Σ instantaneous ≤ B), the trapezoid grow→peak→shrink, the "three jobs"
  (occupancy, brick-avoidance, §3.3.3.2-unreachable). This IS our design; we're realizing it on the DSWS
  claim-based split-K substrate. Build order R0–R4 there.
- **`MAD305_DSWS_MASTER.md:78-81`** — "Split-K is the headroom-creator; split-K creates brief peak windows
  = the room to reallocate VGPR between roles." Matches. **Correction:** its "staggering buys nothing"
  clause describes **full-K / n_kseg=1** (95% duty square wave). With real split-K the peak is brief and
  staggering both avoids the collision AND enables the fungibility. kmbandy's framing: the split-K/dyn-VGPR
  combo is for *whatever's needed at that instant* — compute-per-kseg OR role conversion — not one or the
  other. It's one pool.
- **`vgpr_reserved`/`VRESV` sum-envelope** (`SPEC_DSWS_SUBSTRATE_V2.md:65,165`, `SPEC_DSWS_PHASEB_
  CONVERSION.md:95,130`) — already built, currently applied only to role-conversion grows. **Addition:**
  extend the *same* envelope to gate the per-rowblk/per-kseg compute grow (§4.3). Same counter, same
  primitive — this is the unification, not a new mechanism.
- **`HYBRID_DESIGN.md`** — the grow-ordering rule (grow before loading operands; the DEFERGROW hazard),
  the Σ≤~1152/SIMD budget, RGA livereg. Keep the grow-ordering rule; it's orthogonal and still applies.

---

## 6. What's built & tested this session (all offline unless noted)

- **Entry-fix (`b01c722df`, committed)** — first-entry through `_alloc/_init`. Fixed the desktop-brick;
  confirmed on silicon.
- **CPU model (committed at `5421751`, extended offline)** — `dsws_ctrl_model.cpp` +
  `test_dsws_ctrl_model.cpp`: watermark/epoch/gate/reserve + snapshot/quiesce + entry-contract repro. All
  pass.
- **QUIESCE race harness (`test_dsws_quiesce_race.cpp`, uncommitted)** — `std::thread` claimer+7-follower
  model, 400 trials × {seq_cst, relaxed}, **0 stalls** — ruled out the protocol race.
- **Wedge-frame DIAG instrumentation (`occ_kernel_dsws.s`, uncommitted, byte-identity verified)** — an
  `epoch_mark` macro + a claimer wait-loop snapshot of all advance-gate counters, routed into the host's
  existing `ML8_COOP_STREAM` occ slots. Stream field map: `cons`=ROWBLK_DONE, `tiles`=AROW_DONE,
  `compPh`=BFRAG_DONE, `comp:dsB`=QUIESCE_CNT, `feed:tr/pub/rawTi`=ROWBLK/AROW/BFRAG_NEXT,
  `roles[C/A/B]`=per-role last-bumped epoch. RGA 0-spill, max-live 84.
- **GPU (supervised, kmbandy-greenlit):** the instrumented CONV=1/DIAG=1 4c2a2b bin ran **CLEAN**
  (`ok=1536 bad=0`, occ[0]=0, occ[20]=33, dmesg silent) — the heisenbug that pinned the root cause.
  Bins restored to safe CONV=0 immediately after.

---

## 7. References

**Repo docs (this dir unless noted):**
- `MAD305_DSWS_MASTER.md` — the vision, the issue-port wall, split-K framing.
- `SPEC_DSWS_SUBSTRATE_V2.md` — the claim-based split-K substrate + role economy.
- `SPEC_DSWS_PHASEB_CONVERSION.md` — the conversion economy + `vgpr_reserved` envelope.
- `SPEC_WAVESPEC.md` §EXTENSION — the rolling dyn-VGPR "traveling peak" design (R0–R4). **Primary.**
- `HYBRID_DESIGN.md` — grow-ordering rule, Σ≤pool budget, RGA gates.
- `RESULT_DSWS.md` — the substrate's GPU history (the feed-scheduling starvation class + K-outer fix).
- `docs/superpowers/specs/2026-07-01-dsws-phaseb-pool-economy-design.md` + the matching plan — the
  pool-economy universal-dispatch design (Tasks 1-6 offline built, GPU gates T7-T10).
- `.superpowers/sdd/progress.md` — the SDD ledger; the full blow-by-blow incl the brick/SKIPQ/heisenbug.

**KG (mneme) notes — search these:**
- `1fdd5784` rolling dyn-VGPR / traveling peak (THE architecture).
- `21cc0d63` root cause confirmed = multi-grower collision (this session).
- `d1a6d529` correction: entry-fix fixed the brick, not the residual hang.
- `2009aa16` dyn-VGPR OOR rule (no v>15 pre-grow) — already satisfied in the kernel.
- `98614f24` / `1cef3378` / `a241842c` ISA §3.3.3.2 multi-grower deadlock (authoritative).
- `11d86889` / `7c697c80` never dispatch a dyn kernel above the live SQ_DYN_VGPR cap (reboot lesson).

---

## 8. Safety & operating rules (HARD — kmbandy standing orders)

- **No GPU isolation exists.** gfx1201 drives the desktop; a hang can MODE1-reset it. **Bricks are
  accepted** ("if it bricks, it bricks — we reboot and go from there") **but only with safeguards.**
- **kmbandy greenlights EVERY GPU dispatch individually.** Never auto-dispatch.
- **The safeguarded, proven-non-bricking dispatch config** (keep ALL of these):
  ```
  ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 \
    DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS2_NKSEG=<N> \
    timeout 30 ./occ_dispatch --dsws2 4c2a2b
  ```
  The chunk watchdog (`ML8_COOP_CHUNK_MAXS=0.75`) self-aborts a stalled chunk cleanly; `ML8_COOP_STREAM=1`
  streams occ frames to disk so the wedge frame survives a hard hang.
- **NEVER disable the quiesce gate** (no `SKIPQ`-style force `s51=1`) — it removes the clean self-abort
  and bricks.
- **Always rebuild+install the CONV=1 bin only at dispatch time, and restore the safe CONV=0 4840B bin
  immediately after, unconditionally** (footgun removal — see `scratchpad/fire_wedge_diag.sh` for the
  pattern). Never leave a CONV=1 bin installed.
- **`NFV=112 < 128` (the default `SQ_DYN_VGPR` cap)** — do NOT exceed 128 VGPR without the umr flip
  (sudo, operator-only). Keep grows ≤128.
- **Never modify** `occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.*` — shared/not-ours. Never stage
  them.
- **Commit to git ONLY when kmbandy asks.** Never `--gl2c`.

---

## 9. Your first tasks (next session)

1. **Spec** the rolling dyn-VGPR grow economy from §4: exact LDS slot(s) for the compute grow envelope
   (reuse `VRESV_OFF` or a sibling), the reserve/release macro, the stagger phase-token mechanism, where
   they plug into `.Lcompute_claim`/`.Lcompute_grow`/`.Lcompute_shrink`, the per-kseg flush structure
   (largely already there — each super-tile's ksi flushes to C), and the DIAG telemetry slots. Decide:
   concurrent-peak count (start 1-2, sweep), burst length, stagger period. Save to
   `docs/superpowers/specs/`. Self-review, then have kmbandy read it.
2. **Plan** (`docs/superpowers/plans/`) — offline tasks (CPU model envelope invariant → asm → assemble/RGA
   → byte-identity) then the supervised GPU oracle gate at `n_kseg>1`.
3. **Build**, offline gates first. One greenlit GPU oracle run. Then the occupancy sweep.

**Open question to resolve in the spec:** at `n_kseg>1`, confirm each kseg's partial flushes to C
immediately (atomic-add accumulate — it already does across super-tiles) so a compute wave truly shrinks
to lean between segments (true trapezoid), vs any path where accumulators pin the peak. kmbandy confirmed
the intent is per-kseg flush → immediate release.
