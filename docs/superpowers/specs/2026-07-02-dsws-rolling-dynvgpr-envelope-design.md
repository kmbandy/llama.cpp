# DSWS rolling dyn-VGPR — the sum-envelope grow economy (detailed design)

**Date:** 2026-07-02
**Status:** design, pre-implementation
**Kernel:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_dsws.s` (`--dsws2` path)
**Predecessors:** `HANDOFF_DSWS_ROLLING_DYNVGPR.md` §4 (approved architecture), `SPEC_WAVESPEC.md` §EXTENSION (R0–R4 traveling-peak),
`SPEC_DSWS_SUBSTRATE_V2.md` / `SPEC_DSWS_PHASEB_CONVERSION.md` (`vgpr_reserved`/`VRESV` envelope, `conv_apply`).

This spec turns the handoff §4 architecture into exact LDS offsets, register/scratch assignments, macro definitions, asm
placement, host-streamed DIAG slots, and CPU-model interfaces. It is written to be implemented directly.

---

## 0. One-paragraph problem & fix

`occ_kernel_dsws.s:1158` `.Lcompute_grow` is a **bare `s_alloc_vgpr NFV` SCC-retry with no sum-envelope reservation and no
stagger**. All `NCOMP` compute waves claim a rowblk and issue `s_alloc_vgpr NFV=112` in lockstep, colliding on the shared
per-SIMD dyn-VGPR pool → intermittent forward-progress hang (ISA §3.3.3.2: HW deadlock-avoidance guarantees only *one*
wave/SIMD can reach max allocation). Confirmed root cause (KG `21cc0d63`); the heisenbug (adding DIAG made it run clean)
proves a timing race, not a logic bug. **Fix:** route every per-rowblk compute grow through the existing `vgpr_reserved`
sum-envelope (`reserve_try`, `VRESV_OFF=52`) with a spin-at-lean retry, so at most `PEAK_CONC` waves hold peak at once and
the collision is **unreachable by construction**. Then (a later validation rung) add a lock-free stagger so peaks stay
phase-spread and freed budget flows to whichever role is the current bottleneck.

---

## 1. Goals / non-goals

**Goals**
- G1. Make the multi-grower collision unreachable: enforce `Σ instantaneous VGPR allocations ≤ BUDGET` at every instant,
  gating the *per-rowblk compute burst grow* through the same `vgpr_reserved` counter role-conversion already uses.
- G2. Realize the rolling trapezoid `lean → reserve → grow → WMMA burst → flush → shrink → release` per rowblk/kseg, with
  real split-K (`n_kseg > 1`) so the peak is a brief window and the pool genuinely churns.
- G3. Bake in envelope telemetry (DIAG-gated) so a wedge is a readout (permit-starvation vs elsewhere), not a mystery.
- G4. Keep the door open to a lock-free stagger (phase-spread peaks; feed-fungibility timing) as a tuning layer on top.
- G5. **Byte-identity:** every new instruction sits behind a new assembler symbol; with the new features off, the `.text`
  is bit-for-bit identical to HEAD at `DSWS2_CONV=0` (`e5ec5e50`) and `DSWS2_CONV=1/DIAG=0` (`e296b846`).

**Non-goals (YAGNI)**
- N1. No wave parking / dynamic launch-count change. `WAVES` is fixed; every wave fits lean at once (`WAVES*VLEAN ≤ BUDGET`).
- N2. No `s_barrier` anywhere. Coordination stays LDS-atomic + epoch/phase polling.
- N3. No `NFV > 128` (stays under the default `SQ_DYN_VGPR` cap; no operator umr flip).
- N4. The role-conversion **watermark controller** (occ_sample → try_gate → conv_apply) is unchanged in *policy*. This spec
  only changes what conversion *reserves* against the envelope (see §4.4), and only under the new envelope mode.

---

## 2. The invariant (the one law)

Let `B = BUDGET` (per-SIMD VGPR sum-envelope ceiling), `V_peak = NFV = 112`, `V_lean = VLEAN = 32`, `Δ = V_peak − V_lean = 80`.

```
vgpr_reserved  =  Σ over resident waves of (that wave's current booked allocation)
INVARIANT:     vgpr_reserved ≤ BUDGET     at every instant
```

**Accounting model (this is the crux — it inverts the static-fat init).** Today `VRESV_OFF` is initialized to
`NCOMP*NFV + (NAFEED+NBFEED)*VLEAN` — i.e. it books **all compute waves at full peak permanently** (the static substrate,
where compute waves camp fat for ~95% of K). That leaves **zero** headroom for a transient per-burst reserve, so it cannot
express the rolling model. The rolling model is the inverse:

- **Every wave is `VLEAN` at rest** (compute, A-feed, B-feed, claimer). Role identity = *which loop the wave runs*, not a
  permanent VGPR footprint.
- The reservation counter books **transient burst peaks only.** Init `vgpr_reserved = WAVES*VLEAN` (everyone lean).
- A compute wave taking a rowblk/kseg books `+Δ` (reserve → grow), and releases `−Δ` on shrink. Net zero per rowblk.
- `BUDGET = WAVES*VLEAN + PEAK_CONC*Δ`, where `PEAK_CONC` = the number of concurrent compute peaks the SIMD VGPR file
  admits. At most `PEAK_CONC` waves can hold peak simultaneously; the rest spin at lean until budget frees.

**Concurrent-peak count** (WAVESPEC §EXTENSION worked example): a symmetric trapezoid averages `≈ V_peak/2`, so resident
peaks `≈ B/(V_peak/2) = 2×` the all-at-peak count. `PEAK_CONC` is the sweep knob (R3); start at 2 (proven-safe: 2 peaks =
`2*112 + 6*32 = 416 ≤ B`), sweep up while `Σ ≤ B` holds with slack ≥ one grow-step (the brick guard).

Reconciliation with the existing conversion envelope (§4.4): under envelope mode a feed↔compute **conversion books no peak
VGPR** — both source and dest roles are `VLEAN` at rest, so the flip is a zero-delta reservation. Only the *burst* books
`+Δ`. This folds the conversion reserve into the per-burst reserve — the unification the handoff calls for — and supersedes
`SPEC_DSWS_PHASEB_CONVERSION.md`'s `+(NFV−VLEAN)` feed→compute conversion reserve *when `DSWS2_ENVELOPE=1`*.

---

## 3. New assembler symbols (all default to the byte-identical value)

```asm
.ifndef DSWS2_ENVELOPE
  .set DSWS2_ENVELOPE, 0        // 1 = route the per-rowblk compute burst grow through the vgpr_reserved sum-envelope.
.endif                          //     0 = HEAD behavior (bare .Lcompute_grow) -> .text byte-identical.
.ifndef PEAK_CONC
  .set PEAK_CONC, 2             // concurrent compute peaks the budget admits (R3 sweep axis). Only used when ENVELOPE=1.
.endif
.ifndef DSWS2_STAGGER
  .set DSWS2_STAGGER, 0        // 1 = enable the lock-free phase-token stagger (R3 refinement on top of ENVELOPE).
.endif
.ifndef STAGGER_PERIOD
  .set STAGGER_PERIOD, NCOMP   // phase slots in the stagger ring (R3 sweep axis). Only used when STAGGER=1.
.endif
```

`BUDGET` (already a `-defsym`, `occ_kernel_dsws.s:441`) gains an envelope-mode default:

```asm
.ifndef BUDGET
.if DSWS2_ENVELOPE
  .set BUDGET, (WAVES*VLEAN + PEAK_CONC*(NFV-VLEAN))   // rolling: lean floor + concurrent-peak headroom
.else
  .set BUDGET, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)     // static-fat (HEAD) — unchanged
.endif
.endif
```

Compile-time guards (extend the existing `.error` block at `:447`):
```asm
.if DSWS2_ENVELOPE
  .if (WAVES*VLEAN + (NFV-VLEAN)) > BUDGET
    .error "ENVELOPE: BUDGET admits < 1 concurrent peak — forward progress impossible"
  .endif
.endif
```

`DSWS2_NKSEG` is **host-side only** (dispatch env; already plumbed) — it sets `KT`, and the kernel derives
`n_kseg = KT >> NKSEG_SHIFT` in the prologue (`:621`). No kernel symbol; split-K churn is turned on purely by dispatching
with `DSWS2_NKSEG=<N>` (N>1). `n_kseg=1` remains the degenerate high-duty case (handled free by `DECODE_STI`).

---

## 4. Mechanism

### 4.1 Envelope the per-rowblk compute grow (the correctness fix)

**Init.** Replace the `VRESV_OFF` seed (`occ_kernel_dsws.s:744`) under envelope mode:
```asm
.if DSWS2_ENVELOPE
    lds_put VRESV_OFF, (WAVES*VLEAN)                     // rolling: everyone lean; counter books transient peaks
.else
    lds_put VRESV_OFF, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)   // HEAD (unchanged)
.endif
```

**Reserve before grow, release after shrink.** The current claim loop (`:1151–1226`):
```asm
.Lcompute_claim:
    lds_fetch_add s33, ROWBLK_NEXT_OFF, 1
    s_cmp_ge_u32 s33, G
    s_cbranch_scc1 .Lcompute_drained
.if DYNVGPR
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
    // >>> NEW (ENVELOPE): reserve +Δ against vgpr_reserved; spin-at-lean-retry on over-budget <<<
.if DSWS2_ENVELOPE
.Lcompute_reserve:
    reserve_try +(NFV-VLEAN), s54          // s54 = won (1 = booked, ≤BUDGET); reserve_try scratch s62/s63
    s_cmp_eq_u32 s54, 0
    s_cbranch_scc0 .Lcompute_grow          // won -> grow
    s_sleep SLEEPN                          // over budget -> back off AT LEAN (reserve_try already undid its add)
    s_branch .Lcompute_reserve
.endif
.Lcompute_grow:
    s_alloc_vgpr NFV                        // SCC-retry stays as the HW backstop
    s_cbranch_scc0 .Lcompute_grow
.endif
    ...zero ACC / WMMA burst over KSEG_STEPS / flush to C via global_atomic_add_f32...
    s_wait_storecnt 0x0                      // atomic-adds READ ACC -> drain before shrink frees ACC
.if DYNVGPR
.Lcompute_shrink:
    s_alloc_vgpr 32
    s_cbranch_scc0 .Lcompute_shrink
    // >>> NEW (ENVELOPE): release −Δ; wave is lean-32 here so v-temps ≤v15 (OOR-safe) <<<
.if DSWS2_ENVELOPE
    lds_fetch_add s54, VRESV_OFF, -(NFV-VLEAN)   // mirror of the reserve; shrink already committed
.endif
.endif
    lds_inc ROWBLK_DONE_OFF
    s_branch .Lcompute_claim
```

**Why the collision is now unreachable.** `reserve_try` (`:497`) atomically adds `+Δ`, validates `prev+Δ ≤ BUDGET`, and
UNDOES on failure — the LDS atomic serializes concurrent grows. With `BUDGET = WAVES*VLEAN + PEAK_CONC*Δ`, at most
`PEAK_CONC` waves can be past the reserve gate (holding peak) at once; every other claimant spins in `.Lcompute_reserve` at
lean-32. "All `NCOMP` grow at once" cannot occur. This is exactly the ISA §3.3.3.2 "software scheme for forward progress":
reserve-slack is the guarantee.

**Forward progress.** `PEAK_CONC ≥ 1` (guarded at assemble time) ⇒ at least one waiter always fits. Each burst is bounded
(`KSEG_STEPS` WMMA steps) and unconditionally reaches `shrink → release`, monotonically freeing `Δ`, so a waiter always
eventually proceeds. No wave holds peak indefinitely. (Validated by the CPU race-model, §7.)

**OOR safety (SPEC 4 #1 brick rule).** The reserve runs while the wave is lean-32 (pre-grow); `reserve_try`'s only vector
temps are the `lds_fetch_add` v11/v14 pair (`RP_A`/`RP_D`, ≤v15) — same as `conv_apply`'s proven pre-grow window. The
release runs *after* the shrink, wave back at lean-32, again ≤v15. No VGPR > v15 is sourced before a grow.

**Scratch register budget at the claim loop.** At `.Lcompute_claim`, the conversion scratch band `s52–s65` is dead (conv
only runs at `.Lcompute_drained`). Assignment: `s54` = reserve `won`; `s62/s63` = `reserve_try` internal scratch (already
its documented scratch). Live-across-burst regs untouched: `s33` (rowblk), `s38` (C tile-term), `s35` (epoch),
`s19/s30/s31` (mblk/tcol/ksi), `s67/s68` (mask/shift), `s6/s7`/`s28/s29` (C base). No new persistent register.

**Transient-blip note.** A failed `reserve_try` does add-then-undo, a momentary over-count another wave could observe and
spuriously back off from. This is not a correctness violation (no one ever exceeds `BUDGET`); worst case is one extra
`s_sleep SLEEPN` backoff. The monotonic shrink-frees-budget property rules out livelock (proven in the CPU race-model). We
keep the proven `reserve_try` primitive rather than introduce a peek-without-commit variant.

### 4.2 Real split-K churn (`n_kseg > 1`)

The per-rowblk `grow → WMMA(KSEG_STEPS) → flush → shrink` is **already the trapezoid** — each rowblk covers one `SEGK`
segment (`KSEG_STEPS = SEGK/16` k-steps) and flushes fp32 partials to C via `global_atomic_add_f32`
(`:1205–1219`, `scope:SCOPE_DEV`). With `n_kseg > 1`, `DECODE_STI` splits `sti → (t, ksi)` so a compute wave processes one
`ksi`'s `SEGK`-segment per super-tile; partials from all `ksi` accumulate into the **same** ksi-independent C cell
(`s38 = ti*(G*FM*FN*1024)`, ksi-independent — `:1147`). **No accumulator persists across a wave's ksegs** (the accumulate-
into-C path handles the sum), so the peak is brief and duty drops as `n_kseg` rises. Split-K is fully plumbed; the only
change to exercise it is dispatching with `DSWS2_NKSEG=N` (N>1). **Open question resolved:** per-kseg flush → immediate
release is confirmed the intent and matches the existing structure — the wave shrinks to lean between rowblks/ksegs; no
path pins the peak across ksegs.

**Master knob = burst length.** Here burst length = `KSEG_STEPS` (WMMA steps a wave holds peak per rowblk) × rowblks held
before shrink (currently 1 — shrink every rowblk). Short burst → more churn/occupancy, more grow/shrink overhead; long
burst → amortized overhead, higher duty. Optimum = shortest burst that keeps the WMMA pipe full. `SEGK` (⇒ `KSEG_STEPS`)
and `n_kseg` are the R3 sweep axes alongside `PEAK_CONC`.

### 4.3 The stagger (R3 refinement — phase-spread peaks + feed fungibility)

The envelope alone yields **emergent demand-driven staggering**: with `PEAK_CONC < NCOMP`, waves take turns at peak. The
explicit stagger *coordinates* those turns to (1) keep compute saturated (always `≈PEAK_CONC` peaks active and phase-
spread), (2) keep peaks from bunching, and (3) time the per-burst release so **feed waves can grow the moment the ring
counters say feed is the bottleneck** — freed VGPR flows to the current bottleneck, per-moment (the adaptive core).

**Mechanism (lock-free, no `s_barrier`).** A ring of `STAGGER_PERIOD` phase slots in a new LDS word `STAGGER_TOK_OFF`
(§5). Reuse the prod/cons counter style (`lds_fetch_add` + compare). Before the reserve, a compute wave claims the next
phase slot (`lds_fetch_add STAGGER_TOK_OFF, 1`, mod `STAGGER_PERIOD`) and only proceeds to `.Lcompute_reserve` when its
slot is "open" (its phase's turn), else `s_sleep SLEEPN` and re-poll. The envelope remains the hard floor; the stagger is
scheduling on top (it can never admit more than `BUDGET` allows). Gated under `DSWS2_STAGGER`; `STAGGER=0` ⇒ pure emergent
envelope staggering, byte-identical-when-off.

**Whether explicit stagger beats emergent** is an empirical R3 question (measure WMMA-pipe saturation and feed-wait % both
ways). The spec defines the mechanism; the sweep decides if it earns its bytes.

### 4.4 Conversion-reserve reconciliation (envelope mode only)

When `DSWS2_ENVELOPE=1`, a feed↔compute conversion is a **zero-delta** reservation (both roles lean at rest). Concretely,
under envelope mode the four `conv_apply` call sites pass `delta = 0` for the feed→compute grow direction, and the
compute→feed direction likewise (the wave is already lean; the burst reserve is what books peak). The `s_alloc_vgpr`
in `conv_apply` for feed→compute becomes `s_alloc_vgpr 32` (stay lean — the burst grows later, enveloped), not `NFV`.

**Scoping:** the first GPU correctness gate runs with **conversions dormant** (the exact CONV=1-dormant / CONV=0 configs
already on silicon), so this reconciliation is not exercised at the first gate — the envelope-on-the-burst fix stands
alone. The `conv_apply` delta change lands as a scoped follow-up (its own CPU-model gate) before conversions go live with
the envelope. This is test *ordering*, not scope-cutting: the full design includes both.

### 4.5 Instrumentation baked in (DIAG-gated, byte-identical when `DIAG=0`)

Add envelope telemetry to host-streamed occ slots at the claimer advance-gate wedge frame (`:893` `.if DIAG` block, next
to the existing snapshot dump), and at the reserve/release sites. All under `.if DSWS2_ENVELOPE && DIAG`, lane-0-masked,
`global_store_b32 v4, …, s[0:1] offset:<slot>` (the existing idiom):

| datum | meaning | occ slot (byte off) |
|---|---|---|
| `vgpr_reserved` high-water | peak `Σ` observed — headroom check vs `BUDGET` | occ[8] = 32 |
| reserve-spin count (per wave, saturating) | permit-starvation depth — "wedge = permit wait" vs elsewhere | occ[9] = 36 |
| grow events | bursts entered | occ[12] = 48 |
| shrink/release events | bursts completed (grow==release ⇒ no leak) | occ[13] = 52 |
| `PEAK_CONC` echo | budget config readback | occ[14] = 56 |

(Free slots verified against the used set {24,28,40,44,60,76,84,88,104,108,112,116}; occ[8/9/12/13/14] are unused. The
existing wedge-frame markers — ROWBLK/AROW/BFRAG DONE/NEXT, QUIESCE, per-role epochs — stay as-is.) A wedge then reads out
*directly* whether it is permit-starvation (high reserve-spin, `vgpr_reserved` pinned at `BUDGET`) or elsewhere.

---

## 5. LDS layout delta

Control region is `0..255` below the resident region (`BRES_OFF=256`). Current occupancy: control words `0..68`,
`SNAP_BASE=72` (u32[6] → 72..95), `QUIESCE_CNT_OFF=96`. One new word:

```asm
.set STAGGER_TOK_OFF, 100        // u32 phase-ring counter (only touched when DSWS2_STAGGER=1)
```

Extend the existing overlap guard (`:184`) to include `STAGGER_TOK_OFF < BRES_OFF`. `vgpr_reserved` reuses the existing
`VRESV_OFF=52` (no new slot). No resident-region repoint (all new state stays in the `0..255` control gap), preserving the
`DSWS2_CONV=0` byte-identity.

---

## 6. Assembler-symbol / register / offset summary (implementation checklist)

- **New symbols:** `DSWS2_ENVELOPE` (0), `PEAK_CONC` (2), `DSWS2_STAGGER` (0), `STAGGER_PERIOD` (`NCOMP`). `BUDGET` gains an
  envelope-mode default. All default to the byte-identical value.
- **New LDS:** `STAGGER_TOK_OFF=100` (u32). Reuse `VRESV_OFF=52`.
- **New labels:** `.Lcompute_reserve` (spin-retry reserve loop). Optional `.Lcompute_phase` (stagger poll) under STAGGER.
- **Registers:** `s54` = reserve `won`; `s62/s63` = `reserve_try` scratch (unchanged); no new persistent SGPR/VGPR. Reserve
  runs pre-grow (lean, ≤v15); release runs post-shrink (lean, ≤v15).
- **Edited sites:** `VRESV_OFF` init (`:744`); `.Lcompute_claim/_grow` (`:1151–1161`); `.Lcompute_shrink` (`:1221–1224`);
  `BUDGET` block (`:441`); the `.error` guard (`:447`); the DIAG wedge frame (`:893`); (envelope-mode) the four
  `conv_apply` deltas (`:1010/1019/1084/1093` grows; `:1253/1257/1267/1274` shrinks) — follow-up, §4.4.

---

## 7. CPU-model interfaces (extend `dsws_ctrl_model.cpp` + `test_dsws_ctrl_model.cpp`)

The existing `reserve_grow(resv, delta, budget) -> bool` (`dsws_ctrl_model.cpp:47`) already models the atomic
add/validate/undo. Add, matching the asm exactly:

1. **Signed / spin reserve** — a compute-burst reserve helper: reserve `+Δ`; on fail, back off and retry (models
   `.Lcompute_reserve`). Pure function over a shared `std::atomic<uint32_t>` reservation + `budget`.
2. **Envelope invariant test** — assert `vgpr_reserved ≤ BUDGET` holds across every interleaving; assert grow-count ==
   release-count at quiescence (no lost/leaked reservation); assert `vgpr_reserved` returns to `WAVES*VLEAN` when all
   bursts complete.
3. **Forward-progress race test** — `std::thread` model of `NCOMP` compute waves each looping {claim rowblk → spin-reserve
   → burst (bounded) → release}, `PEAK_CONC ∈ {1,2,…}`, `{seq_cst, relaxed}`, N trials. Watchdog trips iff any wave fails
   to complete its rowblks (reproduces a permit-starvation hang on CPU). Target: **0 stalls** for all `PEAK_CONC ≥ 1`
   (mirrors the existing `test_dsws_quiesce_race.cpp` structure and its 0/400 result).

---

## 8. Validation ladder (test ordering — not scope-cutting)

1. **CPU model** — envelope invariant + forward-progress race (§7) green; existing model tests still pass.
2. **Assemble matrix** — `{ENVELOPE,STAGGER} × {CONV 0/1} × {DIAG 0/1}` assemble clean; RGA 0-spill; max-live ≤ prior.
3. **Byte-identity** — `ENVELOPE=0 && STAGGER=0`: `.text` sha == HEAD at `CONV=0` (`e5ec5e50`, 4840B) and
   `CONV=1/DIAG=0` (`e296b846`). Non-negotiable gate.
4. **One supervised GPU oracle run** — `ENVELOPE=1`, conversions dormant, **`n_kseg > 1`**, `PEAK_CONC=2`, the safeguarded
   config (§9). Target `ok=1536 bad=0`, dmesg silent. kmbandy greenlights it individually.
5. **Sweep** — `PEAK_CONC × SEGK/n_kseg × (STAGGER on/off, STAGGER_PERIOD)`; rocprof + RGA each cell; watch resident-wave
   count, WMMA-pipe %, feed-wait %, `vgpr_reserved` high-water (must keep slack ≥ one grow-step — the brick guard).

---

## 9. Safety & operating rules (HARD — kmbandy standing orders)

- **No GPU isolation.** gfx1201 drives the desktop; a hang can MODE1-reset it. Bricks are accepted **only with safeguards.**
- **kmbandy greenlights EVERY GPU dispatch individually.** Never auto-dispatch.
- **Safeguarded, proven-non-bricking dispatch config** (keep all of it; only `DSWS2_NKSEG` and the envelope build change):
  ```
  ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 \
    DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS2_NKSEG=<N>  \
    timeout 30 ./occ_dispatch --dsws2 4c2a2b
  ```
- **NEVER disable the quiesce gate** (no `SKIPQ`/force `s51=1`) — removes the clean chunk-watchdog self-abort → bricks.
- **Build the `ENVELOPE=1` bin only at dispatch time; restore the safe `CONV=0` 4840B/`e5ec5e50` bin immediately after,
  unconditionally** (footgun removal — see `scratchpad/fire_wedge_diag.sh`). Never leave a fat/`ENVELOPE` bin installed.
- **`NFV=112 < 128`** (default `SQ_DYN_VGPR` cap) — never exceed 128 VGPR without the operator-only umr flip.
- **Never modify / stage** `occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.*` (shared/not-ours). Never pass `--gl2c`.
- **Commit to git ONLY when kmbandy asks.**

---

## 10. References

- `HANDOFF_DSWS_ROLLING_DYNVGPR.md` §4 — approved architecture (this spec's source).
- `SPEC_WAVESPEC.md` §EXTENSION — traveling-peak, sum-envelope reasoning, R0–R4 build order.
- `SPEC_DSWS_SUBSTRATE_V2.md` / `SPEC_DSWS_PHASEB_CONVERSION.md` — `vgpr_reserved`/`VRESV`, `conv_apply`, `reserve_try`.
- `MAD305_DSWS_MASTER.md:78–81` — split-K headroom framing (its "staggering buys nothing" clause describes `n_kseg=1`;
  corrected here for real split-K).
- KG: `1fdd5784` (traveling peak), `21cc0d63` (root cause = multi-grower collision), `98614f24`/`1cef3378`/`a241842c`
  (ISA §3.3.3.2), `2009aa16` (dyn-VGPR OOR rule), `d1a6d529` (entry-fix ≠ residual-hang fix).
