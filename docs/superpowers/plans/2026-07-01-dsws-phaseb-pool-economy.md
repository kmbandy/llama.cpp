# DSWS Phase-B Pool Economy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a converted wave execute its new role's code, and turn the fixed
8-wave partition into a lean-start grow-into-budget pool that rebalances its
{compute, A-feed, B-feed} mix at runtime.

**Architecture:** One scalar (`s59`)-driven dispatcher unifies entry and
re-dispatch (lands on each role's `_follow`, scalar-only). Every wave already
launches lean-32 (compute grows to `NFV` per-rowblk on demand and shrinks back —
Phase-A behavior); "seeding" a role is therefore just setting `s59` in the
existing lean partition arms, with **no launch-time grow**. The only
`s_alloc_vgpr` GROW in the whole design remains `conv_apply`'s, already audited.
`BUDGET` is retuned to the real per-SIMD ceiling so feed→compute grows can
succeed (bidirectional balancing within the launched mix). A cooldown `K` damps
thrash; a `DSWS2_FORCE` hook gives a deterministic first GPU proof. All offline
tasks precede three SUPERVISED GPU gates. Scaling the *launched* wave count
above the current mix sum (the 12–16-wave "bigger pool") is a follow-on once
conversions are silicon-proven — it needs an `occ_dispatch` (not-ours) dims
change and is out of scope here.

**Tech Stack:** Hand-written RDNA4/gfx1201 (wave32) assembly (`occ_kernel_dsws.s`),
a C++ CPU control model (`dsws_ctrl_model.cpp` + gtest-free asserts), the
`build_dsws.sh` / `rga_check.sh` offline gates, and the `occ_dispatch` PM4
harness for supervised GPU runs. Base: committed HEAD `f6dda4ccf`, branch
`feat/dsws-phaseb-conversion`.

## Global Constraints

- A GPU brick is a **BUG**, never a tax. Any hang/DMESG-fault/oracle-`bad>0`/DIAG-mismatch = **full STOP + report + bisect**; NEVER auto-advance to the next variant.
- **kmbandy greenlights EVERY GPU dispatch individually.** Tasks 7–9 each STOP for explicit go before touching the GPU.
- Supervised GPU dispatch env only: `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, `timeout 30`. **NEVER pass `--gl2c`.** `ML8_COOP_STREAM=1` always.
- **No `s_barrier`** anywhere (mixed dyn-VGPR + `s_barrier` hard-deadlocks). LDS-atomic busy-wait only.
- `occ_kernel_coop.s` is **NEVER modified** — known-good reference. All work is additive in `occ_kernel_dsws.s`.
- **All new code gated under `.if DSWS2_CONV`.** With `DSWS2_CONV=0` the assembled object MUST be byte-identical (sha256) to before the change. This is a per-task gate.
- **OOR-poison rule (SPEC §4, #1 brick risk):** any VGPR temp with index > v15 read *before* an `s_alloc_vgpr` GROW completes is poison. Every pre-grow LDS/atomic temp must be ≤ v15; scalar temps ≤ s65.
- **Reserved-register contract:** `occ_sample` clobbers s60/s61(+dst s55/s56); `try_gate` clobbers s62–s65 + v5/v6/v7; `reserve_try` clobbers s62/s63; `conv_dec_floor` clobbers s52/s53/s65 + v5/v6/v7; `conv_apply` clobbers s52/s53/s54. Persistent Phase-B scalars s57(dir)/s58(s_win)/s59(role) sit outside all clobbers. Whole-kernel-live: s67(mask)/s68(shift)/s69(chunkHi, claimer)/s24(wid)/s35(last-epoch). **s66 (n_kseg) is dead after the prologue** and is the designated free persistent scalar for the cooldown counter.
- Config held at G=6, SEGK=64, FM=2, FN=4. `NFV=112`, `VLEAN=32`, conversion delta `NFV−VLEAN=80`. Role slot ids: `NCOMP_SLOT=24 / NAFEED_SLOT=28 / NBFEED_SLOT=32`.
- **Commit to git only when kmbandy explicitly asks.** Otherwise leave the tree dirty and report (overrides the per-task commit step below — do the `git add`/`commit` step only on request).
- Pre-existing dirty files (`occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.*`, `mt_pagedattn*`) are **not ours** — never stage them.
- Shell is **fish** — run any `read <<<` / `set -- $var` snippet via `bash script.sh`, not directly.
- Round-table discipline: implement → adversarial review → kmbandy greenlights each GPU dispatch.

---

## File Structure

- **`dsws_ctrl_model.cpp`** + **`test_dsws_ctrl_model.cpp`** (modify) — CPU source of truth. Add: `role_dispatch()` (slot id → role enum), `cooldown_step()` (per-wave counter), `seed_mix()` + pool invariant `pool_fits_lean()`, and a `quiesce_ready` variant parameterized on `N_POOL`. Tests transcribe the asm's decisions.
- **`occ_kernel_dsws.s`** (modify, additive, all under `.if DSWS2_CONV`) — the `.Ldispatch` trampoline; the 3 tail-branch replacements; the `s59` seed assignment in the existing lean partition arms (no launch grow); the cooldown gate on the decision; the `DSWS2_FORCE` hook; new defsyms `CONV_COOLDOWN / DSWS2_FORCE / DSWS2_FORCE_WID / DSWS2_FORCE_DIR / DSWS2_FORCE_EPOCH`, and a `BUDGET` retune knob. **No `N_POOL`/`SEED_*` defsyms** — pool size == launched mix sum (`WAVES`); seed == launch partition.
- **`build_dsws.sh`** (modify) — thread `CONV_COOLDOWN/BUDGET` and the `DSWS2_FORCE*` defsyms through `mk2()`; keep no-conversion bins buildable.
- **`rga_check.sh`** (already `KSRC`-overridable) — used as-is for the live-VGPR/spill gate.
- **`scratchpad/run_pool_gates.sh`** (create, GPU tasks only) — supervised-gate driver (env above, `timeout 30`, dmesg check, abort-on-anomaly), modeled on the Phase-A `run_mix_gates.sh`.

---

## Task 1: CPU control model — dispatch, cooldown, seed/pool invariants, N_POOL quiesce

**Files:**
- Modify: `dsws_ctrl_model.cpp`, `test_dsws_ctrl_model.cpp`

**Interfaces:**
- Produces: `enum Role { COMPUTE, AFEED, BFEED }`; `Role role_dispatch(uint32_t slot_id)` (24→COMPUTE, 28→AFEED, 32→BFEED); `uint32_t cooldown_step(uint32_t cd)` (saturating `cd?cd-1:0`); `bool in_cooldown(uint32_t cd)`; `bool pool_fits_lean(uint32_t n_pool, uint32_t vlean, uint32_t budget)` (`n_pool*vlean <= budget`); `bool quiesce_ready_pool(uint32_t quiesce_cnt, uint32_t n_pool)` (`quiesce_cnt >= n_pool-1`). These mirror the asm exactly.

- [ ] **Step 1: Write the failing tests** (append to `test_dsws_ctrl_model.cpp`, inside `main()` before the final PASS print):

```cpp
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd <dvgpr_occ> && g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/test_dsws_ctrl 2>&1 | head`
Expected: FAIL — `role_dispatch`/`cooldown_step`/`pool_fits_lean`/`quiesce_ready_pool` not declared.

- [ ] **Step 3: Implement in `dsws_ctrl_model.cpp`**

```cpp
enum Role { COMPUTE, AFEED, BFEED };
inline Role role_dispatch(uint32_t slot_id) {
    return slot_id == 24 ? COMPUTE : (slot_id == 28 ? AFEED : BFEED);
}
inline uint32_t cooldown_step(uint32_t cd) { return cd ? cd - 1 : 0; }
inline bool in_cooldown(uint32_t cd) { return cd > 0; }
inline bool pool_fits_lean(uint32_t n_pool, uint32_t vlean, uint32_t budget) {
    return (uint64_t)n_pool * vlean <= budget;
}
inline bool quiesce_ready_pool(uint32_t quiesce_cnt, uint32_t n_pool) {
    return quiesce_cnt >= n_pool - 1;
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd <dvgpr_occ> && g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/test_dsws_ctrl && /tmp/test_dsws_ctrl`
Expected: `dsws_ctrl_model: dispatch/cooldown/pool OK` then `dsws_ctrl_model: ALL PASS`.

- [ ] **Step 5: Commit** (*only if kmbandy asks*): `feat(dsws): CPU model for dispatch/cooldown/pool invariants`

---

## Task 2: Seed `s59` in the lean partition arms + BUDGET retune (no launch grow)

**Files:**
- Modify: `occ_kernel_dsws.s` (entry role branch region ~L597–629; defsym/BUDGET block ~L128–131 / ~L403); `build_dsws.sh` (`mk2()`)

**Interfaces:**
- Consumes: `NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT`, `VLEAN`, `WAVES`.
- Produces: every non-claimer wave records its launch role slot id in `s59` before it reaches the dispatcher. `BUDGET` overridable via `-defsym`. Consumed by Tasks 3–5.
- **Correction vs spec §3/§6 (kmbandy-approved 2026-07-01):** every wave already launches lean-32 (compute grows to `NFV` per-rowblk, not at init). Seeding is `s59`-assignment only — **no launch-time `s_alloc_vgpr`**, so no new OOR grow site. `N_POOL`/`SEED_*` are dropped (pool size == `WAVES` == launched mix sum; seed == the existing wid partition).

- [ ] **Step 1: BUDGET retune knob + no-parking invariant** (near the existing `BUDGET` defsym ~L403). The default stays the launch-footprint (Phase-A conservation); a `-defsym BUDGET=` override supplies real per-SIMD headroom for feed→compute grows. Add the compile-time no-parking assert:

```asm
// compile-time no-parking invariant: every launched wave must fit lean at once
.if (WAVES * VLEAN) > BUDGET
  .error "WAVES*VLEAN exceeds BUDGET — pool cannot stay all-lean (parking is out of scope)"
.endif
```

(`BUDGET` itself already exists behind `.ifndef BUDGET` at ~L403 — do not redefine it; only add the invariant, placed after both `WAVES` and `BUDGET` are set.)

- [ ] **Step 2: Seed `s59` in the existing partition arms.** In the entry role branch (~L625–629), which already routes wid→`.Lbfeed`/`.Lafeed`/`.Lcompute`, add — under `.if DSWS2_CONV` — a one-instruction `s59` write on each arm so the dispatcher/re-dispatch can read the role. Do **not** restructure the branch and do **not** add any `s_alloc_vgpr`. Keep the `.else` (CONV=0) path byte-identical. Structure (`wid`=s24):

```asm
.if DSWS2_CONV
    s_cmp_lt_u32 s24, NBFEED
    s_cbranch_scc1 .Lseed_bfeed
    s_cmp_lt_u32 s24, (NBFEED+NAFEED)
    s_cbranch_scc1 .Lseed_afeed
    s_mov_b32 s59, NCOMP_SLOT
    s_branch .Lcompute                     // Task 3 retargets these three to .Ldispatch
.Lseed_afeed:
    s_mov_b32 s59, NAFEED_SLOT
    s_branch .Lafeed
.Lseed_bfeed:
    s_mov_b32 s59, NBFEED_SLOT
    s_branch .Lbfeed
.else
    s_cmp_lt_u32 s24, NBFEED
    s_cbranch_scc1 .Lbfeed
    s_cmp_lt_u32 s24, (NBFEED+NAFEED)
    s_cbranch_scc1 .Lafeed
    s_branch .Lcompute
.endif
```

**Verify (name in report):** `s59` is written on every non-claimer arm before any `_follow`; no `s_alloc_vgpr` added; `s24`(wid) intact; the CONV=0 `.else` arm is character-identical to the pre-edit branch.

- [ ] **Step 3: Thread `BUDGET` (and later `CONV_COOLDOWN`, `DSWS2_FORCE*`) through `build_dsws.sh mk2()`** — add `-defsym,BUDGET=${BUDGET:-...}` passthrough (default = the current launch-footprint expression), leaving existing calls unchanged.

- [ ] **Step 4: Assemble + byte-identity + RGA.** Run:

```bash
cd <dvgpr_occ>; L=/opt/rocm/llvm/bin
# byte-identity at CONV=0 (baseline = same defsyms assembled from HEAD f6dda4ccf):
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DSWS2=1 -Wa,-defsym,DSWS2_CONV=0 \
  -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 -Wa,-defsym,SAFEPROBE=1 \
  -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 -c occ_kernel_dsws.s -o /tmp/c0_after.o \
  && sha256sum /tmp/c0_after.o
# CONV=1 assembles across all three mixes:
for MIX in "4 2 2" "6 1 1" "2 3 3"; do read NC NA NB <<< "$MIX"
  $L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DSWS2=1 -Wa,-defsym,DSWS2_CONV=1 \
    -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=1 \
    -Wa,-defsym,NCOMP=$NC -Wa,-defsym,NAFEED=$NA -Wa,-defsym,NBFEED=$NB -c occ_kernel_dsws.s -o /tmp/t2.o \
    2>/tmp/t2.err && echo "$MIX ASSEMBLE_OK" || { echo "$MIX FAIL"; sed -n '1,15p' /tmp/t2.err; }
done
```
*(Run via `bash script.sh` — fish won't `read <<<`.)* Expected: CONV=0 sha256 == the HEAD baseline sha256 (assemble HEAD's `occ_kernel_dsws.s` with the identical CONV=0 line to get the baseline); all three `ASSEMBLE_OK`. Then `KSRC=occ_kernel_dsws.s bash rga_check.sh t2_pool DSWS2=1 DSWS2_CONV=1 NCOMP=4 NAFEED=2 NBFEED=2` → `SGPR_SPILLS=0 VGPR_SPILLS=0`.

- [ ] **Step 5: Commit** (*only if kmbandy asks*): `feat(dsws): seed s59 in lean partition arms + BUDGET headroom knob`

---

## Task 3: The `.Ldispatch` trampoline + tail-branch replacements

**Files:**
- Modify: `occ_kernel_dsws.s` (add `.Ldispatch`; replace tail branches at L907, L951, L1097; flip Task-2 seed arms to `.Ldispatch`)

**Interfaces:**
- Consumes: `s59` (seeded in Task 2, flipped by `conv_apply`), the `_follow` labels `.Lcompute_follow`/`.Lafeed_follow`/`.Lbfeed_follow`.
- Produces: universal per-epoch role dispatch. Every non-claimer wave routes to the `_follow` of the role `s59` names.

- [ ] **Step 1: Add the trampoline** (under `.if DSWS2_CONV`, placed after the three role bodies so all `_follow` labels are in scope):

```asm
.if DSWS2_CONV
.Ldispatch:                                  // scalar-only, wave-uniform; s59 = current role slot id
    s_cmp_eq_u32 s59, NCOMP_SLOT
    s_cbranch_scc1 .Lcompute_follow
    s_cmp_eq_u32 s59, NAFEED_SLOT
    s_cbranch_scc1 .Lafeed_follow
    s_branch .Lbfeed_follow
.endif
```

- [ ] **Step 2: Replace the three tail branches** — under `.if DSWS2_CONV` route to `.Ldispatch`; `.else` keep the verbatim Phase-A branch (byte-identity). At L907 (bfeed), L951 (afeed), and L1097 (compute, after `.Lcmp_quiesce`):

```asm
.if DSWS2_CONV
    s_branch .Ldispatch
.else
    s_branch .Lbfeed_follow      // (afeed: .Lafeed_follow ; compute: .Lcompute_follow)
.endif
```

- [ ] **Step 3: Flip the Task-2 seed arms** from the temporary `.Lcompute`/`.Lafeed`/`.Lbfeed` targets to `.Ldispatch` (three `s_branch .Ldispatch`), so entry uses the same dispatcher. **Verify (name in report):** `.Ldispatch` lands on `_follow` (NOT `_alloc`/`_init`), and `s35` (last-epoch) is preserved across the trampoline so a re-dispatched wave waits for the next epoch.

- [ ] **Step 4: Assemble + byte-identity + RGA** — same commands as Task 2 Step 4. Expected: CONV=0 sha256 unchanged from baseline; 3× `ASSEMBLE_OK`; RGA 0 spills. Additionally grep the object disassembly is not required, but confirm no `s_barrier` was introduced: `grep -c s_barrier occ_kernel_dsws.s` → unchanged from HEAD.

- [ ] **Step 5: Commit** (*only if kmbandy asks*): `feat(dsws): universal s59 dispatch trampoline (entry + re-dispatch)`

---

## Task 4: Cooldown `K` gate on the conversion decision

**Files:**
- Modify: `occ_kernel_dsws.s` (decision blocks at the three bails ~L897–902 bfeed, L941–946 afeed, L1071–1091 compute; seed init in Task 2)

**Interfaces:**
- Consumes: `s66` (designated free persistent scalar, dead after prologue), `CONV_COOLDOWN` defsym.
- Produces: a wave skips the watermark decision while `s66 > 0`; on a committed conversion `s66` resets to `CONV_COOLDOWN`; `s66` decrements once per epoch.

- [ ] **Step 1: Add the defsym** (with the DSWS2_CONV defsyms):

```asm
.ifndef CONV_COOLDOWN
  .set CONV_COOLDOWN, 0            // default 0 = spec-faithful (no cooldown); storm sets 0; >0 damps thrash
.endif
```

- [ ] **Step 2: Init `s66=0` at seed** (Task 2 entry, before `.Ldispatch`): one `s_mov_b32 s66, 0` on the common seed path so every wave starts un-cooled. **Verify** all pre-existing `s66` refs are prologue-only (`grep -n '\bs66\b' occ_kernel_dsws.s` — confirm none live past the shift/mask derivation) before repurposing it.

- [ ] **Step 3: Gate each decision + decrement.** At each bail, before `occ_sample`, add (under `.if DSWS2_CONV`): if `in_cooldown` skip straight to the `QUIESCE_CNT` bump; else run the decision. Always decrement once per epoch. Pattern (compute bail shown; mirror at afeed/bfeed, branching to the role's `_quiesce` label):

```asm
.if CONV_COOLDOWN > 0
    s_cmp_eq_u32 s66, 0
    s_cbranch_scc0 .Lcmp_cooldn        // s66>0 -> in cooldown, skip decision
.endif
    // ... existing occ_sample + watermark_decision + try_gate + conv_apply ...
    // inside conv_apply commit path (on \won), reset cooldown:  s_mov_b32 s66, CONV_COOLDOWN
.Lcmp_cooldn:
.if CONV_COOLDOWN > 0
    s_cmp_eq_u32 s66, 0                 // decrement once per epoch (saturating)
    s_cbranch_scc1 .Lcmp_quiesce
    s_sub_i32 s66, s66, 1
.endif
.Lcmp_quiesce:
    // ... existing QUIESCE_CNT bump ...
```

The cooldown reset on commit is added inside `conv_apply`'s win path (after `s59` flip, `s_mov_b32 s66, CONV_COOLDOWN`), gated `.if CONV_COOLDOWN > 0`. **Verify:** with `CONV_COOLDOWN=0` (default) none of this code emits bytes → byte-identity vs Task 3 preserved.

- [ ] **Step 4: Assemble both K settings + CPU model.** Assemble CONV=1 with `CONV_COOLDOWN=0` (expect byte-identical to Task 3) and with `CONV_COOLDOWN=4` (expect `ASSEMBLE_OK`, RGA 0 spills). Re-run `test_dsws_ctrl_model` → `ALL PASS`.

- [ ] **Step 5: Commit** (*only if kmbandy asks*): `feat(dsws): conversion cooldown K (default 0, tunable damper)`

---

## Task 5: `DSWS2_FORCE` deterministic bring-up hook

**Files:**
- Modify: `occ_kernel_dsws.s` (decision blocks); defsym block; `build_dsws.sh`

**Interfaces:**
- Consumes: `s24` (wid), `s35`/epoch, the `conv_apply` commit.
- Produces: with `DSWS2_FORCE=1`, exactly the wave `DSWS2_FORCE_WID` converts direction `DSWS2_FORCE_DIR` at epoch `DSWS2_FORCE_EPOCH`, watermarks bypassed; every other wave never converts.

- [ ] **Step 1: Add defsyms:**

```asm
.ifndef DSWS2_FORCE
  .set DSWS2_FORCE, 0
.endif
.ifndef DSWS2_FORCE_WID
  .set DSWS2_FORCE_WID, 0
.endif
.ifndef DSWS2_FORCE_DIR
  .set DSWS2_FORCE_DIR, 0            // 0/1 = compute->A/B ; 2/3 = A/B->compute
.endif
.ifndef DSWS2_FORCE_EPOCH
  .set DSWS2_FORCE_EPOCH, 1
.endif
```

- [ ] **Step 2: Add the forced path** at each bail, gated `.if DSWS2_CONV && DSWS2_FORCE`, taken INSTEAD of the watermark decision: convert iff `s24 == DSWS2_FORCE_WID` AND current epoch (`s35`) `== DSWS2_FORCE_EPOCH` AND this bail's role matches the source of `DSWS2_FORCE_DIR`; then `try_gate DSWS2_FORCE_DIR` + `conv_apply` with the matching src/dst slots. Bypass the watermark compare entirely. Only the site whose dir source matches emits the `conv_apply` (the dir is a compile-time immediate — a `.if DSWS2_FORCE_DIR == N` selects the site). **Verify:** with `DSWS2_FORCE=0` no bytes emit → byte-identity preserved.

- [ ] **Step 3: Thread the four `DSWS2_FORCE*` defsyms through `build_dsws.sh mk2()`.**

- [ ] **Step 4: Assemble matrix.** Assemble CONV=1 `DSWS2_FORCE=0` (byte-identical to Task 4 at K=0) and `DSWS2_FORCE=1 DSWS2_FORCE_WID=4 DSWS2_FORCE_DIR=0 DSWS2_FORCE_EPOCH=2` → `ASSEMBLE_OK`, RGA 0 spills. (For `4c2a2b` the partition is wids 0–1 B-feed, 2–3 A-feed, 4–7 compute; wid 4 is a compute seed, so `DIR=0` = compute→B-feed exercises a shrink at a known wave/epoch.)

- [ ] **Step 5: Commit** (*only if kmbandy asks*): `feat(dsws): DSWS2_FORCE deterministic bring-up hook`

---

## Task 6: Offline integration gate (consolidation)

**Files:**
- Modify: none (verification-only task); optionally `build_dsws.sh` (add pool bins to the default build set)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: the green offline package kmbandy sees before greenlighting Task 7.

- [ ] **Step 1: Full assemble matrix.** For each mix × {CONV=0, CONV=1/K=0, CONV=1/K=4, CONV=1/FORCE=1}: assemble → `ASSEMBLE_OK`. CONV=0 sha256 == HEAD baseline for every mix.

- [ ] **Step 2: RGA live-VGPR within BUDGET.** For each CONV=1 mix at the intended `BUDGET`: `KSRC=occ_kernel_dsws.s bash rga_check.sh pool_<mix> ...` → `SGPR_SPILLS=0 VGPR_SPILLS=0` and reported max-VGPR ≤ the per-wave `NFV`. The `WAVES*VLEAN ≤ BUDGET` no-parking invariant is proven at assemble time by the `.error` guard (Task 2).

- [ ] **Step 3: CPU model** — `test_dsws_ctrl_model` → `ALL PASS`.

- [ ] **Step 4: Dry-print** — `DSWS2_DRYRUN=1 DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS2_NKSEG=1 ./occ_dispatch --dsws2 4c2a2b 2>&1 | grep -iE "NCOMP|n_kseg|tier|REFUSE"` → prints the mix, no `REFUSE`.

- [ ] **Step 5: Write the offline-green summary** to `.superpowers/sdd/progress.md` and STOP for kmbandy greenlight. Commit the whole offline series (*only if kmbandy asks*).

---

## Task 7: [SUPERVISED GPU] CONV=1 inert-safe re-baseline gate (conversions dormant)

**Files:**
- Create: `scratchpad/run_pool_gates.sh`

**Interfaces:**
- Consumes: the CONV=1 bins with conversions suppressed.
- Produces: the first silicon proof that the Phase-B machinery (seed `s59`, `.Ldispatch`, snapshot/quiesce sentinels — none of which has ever run on GPU) is inert-safe: with conversions dormant it reproduces the Phase-A green. Isolates "did the machinery regress the substrate" from "does conversion work" (Task 8).

- [ ] **Step 1: Build the dormant bins** — CONV=1 with watermarks unreachable so no ticket ever fires: `DSWS2_FORCE=0` and set the watermark thresholds to make `watermark_decision` never trigger (e.g. `CTRL_LOW=0` so `occ_X < 0` is never true, and `CTRL_HIGH_*` ≥ the ring depth). Confirm from the disassembly/DIAG that zero conversions occur. All three mixes, 0-spill.

- [ ] **Step 2: STOP — request greenlight.** Present the offline-green package (Task 6) and state plainly this is the first-ever CONV=1 GPU run; conversions are provably dormant.

- [ ] **Step 3: Run the re-baseline sweep** — the Phase-A driver pattern in `scratchpad/run_pool_gates.sh`: 3 mixes × {n_kseg 1, 8}, env `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, `timeout 30`, dmesg check, abort-on-anomaly. Expected per gate: `dsws2 oracle CLEAN`, `ok=1536 bad=0`, `occ[0]=0`, exit 0, DIAG `occ[29]` == agree (0 mismatch — the snapshot/quiesce cross-check's first silicon check), dmesg silent.

- [ ] **Step 4: On ANY anomaly** (bad>0, 124/hang, dmesg fault, occ[29] mismatch): **full STOP + bisect**; do not run Task 8. A failure here is a machinery-regression, isolated from conversion behavior.

- [ ] **Step 5: Record** the milestone in `.superpowers/sdd/progress.md` and a KG note.

---

## Task 8: [SUPERVISED GPU] Force-convert gate

**Files:**
- Modify: `scratchpad/run_pool_gates.sh` (force params)

**Interfaces:**
- Consumes: `DSWS2_FORCE` bins from Task 5.
- Produces: the deterministic first proof that a converted wave runs its new role's code bit-exact.

- [ ] **Step 1: Build the forced bin** — `DSWS2_CONV=1 DSWS2_FORCE=1 DSWS2_FORCE_WID=4 DSWS2_FORCE_DIR=0 DSWS2_FORCE_EPOCH=2`, one mix (`4c2a2b`; wid 4 = a compute seed), 0-spill confirmed.

- [ ] **Step 2: STOP — request greenlight.** Present the Task 7 inert-safe clean result + the exact forced-conversion parameters. Do not proceed without explicit go.

- [ ] **Step 3: Run the forced gate** — env `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, `timeout 30`, `DSWS2_NKSEG` 1 then 8. Expected per run: `dsws2 oracle CLEAN`, `ok=1536 bad=0`, `occ[0]=0`, exit 0, DIAG `occ[29]` == agree (0 mismatch), dmesg silent.

- [ ] **Step 4: On ANY anomaly** (bad>0, 124/hang, dmesg fault, occ[29] mismatch): **full STOP + bisect**; do not run Task 9.

- [ ] **Step 5: Record** the milestone in `.superpowers/sdd/progress.md` and a KG note.

---

## Task 9: [SUPERVISED GPU] Dynamic-mix gate (watermark-driven conversions)

**Files:** Modify: `scratchpad/run_pool_gates.sh` (watermark params)

**Interfaces:** Consumes: the K=0 CONV=1 pool bins. Produces: proof the watermark balancer moves roles while the oracle stays clean.

- [ ] **Step 1: Build** the CONV=1 `DSWS2_FORCE=0 CONV_COOLDOWN=0` pool bins for all three mixes, 0-spill.

- [ ] **Step 2: STOP — request greenlight.** Present the Task 8 clean result + the watermark settings that will actually fire conversions.

- [ ] **Step 3: Run the dynamic sweep** — 3 mixes × {n_kseg 1, 8}, env + `timeout 30` + dmesg check, abort-on-anomaly. Expected per gate: `oracle CLEAN`, `bad=0`, `occ[0]=0`, `occ[29]` agree, exit 0.

- [ ] **Step 4: On ANY anomaly: full STOP + bisect;** do not run Task 10.

- [ ] **Step 5: Record** milestone + KG note.

---

## Task 10: [SUPERVISED GPU] Storm gate (race-hunt)

**Files:** Modify: `scratchpad/run_pool_gates.sh` (storm params)

**Interfaces:** Consumes: CONV=1 pool bins. Produces: the lock-free race-hunt at maximal conversion frequency.

- [ ] **Step 1: Build** CONV=1 with tight watermarks + `EPOCH_SHIFT=0` + `CONV_COOLDOWN=0`.

- [ ] **Step 2: STOP — request greenlight.**

- [ ] **Step 3: Run** each mix ×10 repeats, env + `timeout 30` + dmesg check. Expected every repeat: `oracle CLEAN`, `bad=0`, `occ[0]=0`, `occ[29]` agree.

- [ ] **Step 4: On ANY anomaly: full STOP + bisect.**

- [ ] **Step 5: Record** the final milestone; dispatch the whole-branch review (subagent-driven-development final review) and then superpowers:finishing-a-development-branch.

---
