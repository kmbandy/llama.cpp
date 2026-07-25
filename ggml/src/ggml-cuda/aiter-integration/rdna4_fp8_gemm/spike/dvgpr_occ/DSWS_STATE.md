# DSWS — CURRENT OPERATING STATE (single source of truth)

> **This is NOT a journal.** It is the canonical *current configuration* + *load-bearing decisions*.
> Read it FIRST every session, restate it, and get "confirmed" BEFORE building or running anything.
> Update it as the LAST action of every session. The narrative lives in `DSWS_TESTING_LOG.md`.
> Last updated: **2026-07-17** (after GROUPS>1 fix verified).

---

## DEFINITIVE ARCHITECTURE — the river / tier system (kmbandy's unified model, 2026-07-17, verbatim intent)

> **Running water with flow control.** Nothing ever stops moving. Each wave runs a TIER LADDER every loop:
> it does the most productive thing available; if that path is blocked, it drops to the next tier and retries.
> Every producer/consumer role goes to the **next available wave** — no fixed owner, no blocking read.

**The lifecycle:**
1. **Grab work** — the *next available wave* grabs a super-tile off the global pool, brings it to the WG; it
   splits into K-slices. *(= intra-WG decentralized ASSIGN — the one tier still centralized on wid0; BUILD THIS.)*
2. **Stagger** launches compute waves one-at-a-time, in sequence, as fast as free VGPR allows.
3. As many waves as possible **grow to peak** (dyn-VGPR grow + fungible physical budget; grow-fail→coast is the
   only throttle).
4. The instant a wave **starts to shrink**, the **baton** pokes the *next available* compute wave — "grab a
   K-slice, grow now" — so the peak TRAVELS (always ≥1 wave at peak, continuous compute).
5. **Banking** accumulates each K-slice's partial on-chip (per-WG LDS banks).
6. When a tile is fully reduced, the *next available wave* **carries it off to DRAM** (one C-store).
7. **Tier ladder** (per wave, each iteration, first-available wins): deliver → grow/compute → assign → feed/stage → coast.

**Build status of each tier:** assign = **centralized on wid0 → decentralize (Fork A, NEXT)**; grow-turn (baton) =
built; banking/dyn-VGPR/stagger/GROUPS = built; deliver = already next-available (TILEDONE completer).

**⚠️ NAME MAP (two meanings bit us 2026-07-17 — use these):**
- **"lazy carry-off"** = kmbandy's *"deep-J delivers to DRAM"* = the decentralized delivery tier = code's **TILEDONE completer**.
- **`JDEPTH`** (code knob) = "a wave holds J K-slices in registers before one write" = flush-frequency optimization. NARROWER; not the delivery tier.
- **Banked** = on-chip LDS accumulation (§ GLOSSARY). NOT deep-J.

---

## RECONCILE (open conflicts — none blocking right now)

1. **DECENTASN — RESOLVED (2026-07-17).** The afternoon's confusion was a terminology collision: kmbandy read
   "banked" as "deep-J" and understood DECENTASN as "up-front decentralized accounting so no wave blocks on
   assignment." Ground truth: **global `DECENTASN=1` was refuted on silicon** (`391c7530`: global claim ⊥ per-WG
   banked reduce); every verified run (deep-J, baton, GROUPS>1) is `DECENTASN=0` coordinator. **Resolution
   (kmbandy's authorized call):** keep the intent (no assignment block) via **intra-WG decentralized assign**
   (Fork A) — WG owns whole tiles, decentralize the within-WG producer. See DECISION JOURNAL. Global DECENTASN dead.

---

## GLOSSARY (terms that have drifted — define them so they never mislead again)
- **Banked** = combine split-K partials in **LDS on-chip** (per-workgroup), one C-store per tile. `WOFLUSH=0 BANKZERO=1`.
  It is the reduce METHOD — **NOT deep-J**. It's what you're left with for split-K once the flush is killed.
- **Deep-J (`JDEPTH`)** = one wave holds J K-segments in **registers**, flushes once. Flush-amortization. Orthogonal to banked.
- **The flush (`WOFLUSH=1`)** = each segment atomic-adds to C in **DRAM** (cross-workgroup). KILLED (97% of clock).
- **Split-K partials combine in exactly 3 places:** DRAM (flush, killed) / LDS (banked, same-WG only) / registers (full-K, no split-K).

## DECISION JOURNAL (who decided what, and was it authorized — append-only; do NOT silently attribute a decision to kmbandy)
| date | decision | authorization | status |
|---|---|---|---|
| 2026-07-13 | Kill the flush (`WOFLUSH=1` → off) — it was 97% of the clock | **kmbandy, explicit** | standing |
| 2026-07-15 | Use **banked** (LDS reduce) for split-K | was silently attributed to kmbandy; **RE-CONFIRMED by kmbandy 2026-07-17** as part of the intra-WG decision (banked is correct given split-K is kept + flush is killed) | authorized |
| 2026-07-16 | Global **`DECENTASN=1`** (any wave claims any super-tile) | designed, then **REFUTED on silicon** (global claim scatters a tile's slices cross-WG ⊥ per-WG LDS reduce) | **dead-as-global** |
| 2026-07-16 | Deep-J on the coordinator (`DECENTASN=0`) | measured, work-exact, 5.2→22 TF fed | verified |
| 2026-07-16 | Baton = **pure notification** (poke-at-shrink + wake-sleeper) | **kmbandy, explicit + verbatim** | verified-built |
| 2026-07-17 | GROUPS>1 fix (ACC_N<G decouple), 3 bugs | kmbandy directed "just fix it" | verified full-oracle |
| 2026-07-17 | **Accounting + reduce = intra-WG decentralized assign + banked LDS combine** (Fork A). WG owns whole tiles (banked-valid); decentralize *which wave in the WG* assigns (no single-wid0 block). Global DECENTASN stays dead; Thread B parked. | **kmbandy, EXPLICIT (2026-07-17)** | **AUTHORIZED — the direction** |

**RESOLVED (kmbandy, 2026-07-17): (a) intra-WG decentralized assign + banked.** Each workgroup still claims WHOLE tiles
(so its per-WG LDS combine is valid); the within-WG super-tile assignment is decentralized so no single wave (wid0) is
the producer bottleneck and no wave blocks waiting for assignment. **Global `DECENTASN=1` stays dead** (silicon-refuted).
Thread B (full-K/no-split-K) parked.

### ⭐⭐⭐ CANONICAL CONFIG — **J=1 (JDEPTH=1) IS LOCKED** — RUN VIA `./dsws.sh` — kmbandy 2026-07-18 ⭐⭐⭐
> **JDEPTH=1. DEEP-J (JDEPTH>1) IS RETIRED. DO NOT RE-ENABLE IT. DO NOT DROP THIS.**
> Deep-J in this kernel = ONE carrier wave computes J K-slices SERIALLY in its own registers (walking pool slots, WAITING
> for each to stage = `jwait`), flush once. It **LOSES** on 2s clock-committed runs: **J=1 = 9.5 TF vs J=2 = 8.7 TF @ product 256**.
> The CORRECT design (kmbandy's) is **J=1 BANKED**: each wave computes ONE slice → `ds_add_f32` into the shared LDS bank → the
> **TILEDONE completer** (a free wave) does the ONE C-store. PARALLEL compute, DECOUPLED delivery — that is the design, and it is
> what wins. (The 2026-07-16 "higher-J = better" numbers were ALL sub-0.5s idle-clock; the 2s-committed runs REFUTE them.)

**Build baked into `./dsws.sh` (do NOT re-derive):**
```
DECENTASN=1 STAGGER=1 BATONGATE=1 BANKZERO=1 WOFLUSH=0 JDEPTH=1 FM=1 G=6 ACC_N=6 POOL_N=2 SEGK=64 WAVES=30 MSDRAIN=1 RBU=1
```
Host geom: `FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=2 DSWS2_SEGK=64` ; shape (attn_q):
`DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64`. `./dsws.sh` = fed run; `./dsws.sh correct` = bounded-K full stride=1 oracle.
Correctness ✅ (SEGK=64 J=1 is the simplest/safest banked path; B-addr 64-bit fix IN). **DYNVGPR=1 default.**

**THROUGHPUT MAP (measured 2s clock-committed, 2026-07-18):** SEGK is THE lever (bigger SEGK = fewer flushes, no carrier cost) —
**J=1/SEGK=256/ACC_N=3 = 9.5 TF** > J=1/SEGK=64/ACC_N=6 = 7.5 TF, still climbing (SEGK=256 needs a correctness gate). **G is NOT a
lever** (flat/negative). **The traveling-peak BATON has NO regime** (grow-fail=0 at G=6 AND G=18) — retired. **The WALL is
FEED/STAGING** (`nothing-staged` ~95-98% of coast) — the next lever. **MEASUREMENT RULE: NEVER quote TF from a run <2s; feed to
≥2s clock-committed via BIG M (many tiles, low RAM), NOT deep-K (OOMs ~30GB at 2s).**

**⭐ 2026-07-18 — 64-BIT B-ADDRESS FIX (deep-K wrong-C):** deep-K feeding (`DSWS2_K` large, single chunk — the CORRECT way to add
work; reps trip the fat-carrier deadman) exposed a 32-bit integer overflow in the shuffled-B segment offset (`ksi*KSEG_STEPS*s10`
via low-32 `s_mul_i32` + a single `s_addc_u32 …,0`). Overflows 2^32 at **n_kseg ≥ 32768** (B K-offset ≥ 4GiB) → wrong C
(work-exact). Fixed with `s_mul_hi_u32` + carry at all 3 B-stage sites (BSTAGE/BSTAGE_R×2). **Real ml8 shapes (K≤4096) NEVER hit
it — always correct;** only artificial deep-K feeding reached it. Found by Codex, numerically reproduced (relerr=1.375001431).
Validated: deep-K K=2097152 was bad=24→**now bad=0**, work-exact, clock-committed 1.29s TF=7.7. See DSWS_TESTING_LOG.md 2026-07-18.
FOLLOW-UP: A-address has the same 32-bit pattern (safe until K>~7.4M at Mo=576) — port the fix to ASTAGE before feeding that deep.

**FED RUN (`./dsws.sh`, 576×4096×262144, n_kseg=4096):** oracle CLEAN (sampled), computed=9437184 EXACT, occ[0]=0. **BUT span
only ~0.17s (1 chunk) — NOT true steady state (<5s; feed harder next time: more reps/tiles).** Directional finding MATCHES the
prior KG headline: **`door4 grow-fail=0` (VGPR budget never binds at G=6), `occ[88] jwait=127M` (carriers STAGE-STARVED, fat-
waiting for feed), `occ[98] baton=0` (INERT — no budget valley → baton==river), TF=7.3.** ⇒ the decentralized assign is CORRECT
+ fed, but the wall is STILL **stage-wait**, not assign/budget — the baton's value REQUIRES **binding G (G>12)**, which needs
SEGK=32 (LDS) i.e. the G=18/GROUPS=3 path. So the SEGK=64/G=6 canonical proves CORRECTNESS but CANNOT exercise the baton;
binding-G at a real shape without WOFLUSH remains the open architectural question (see the deep-J-costs / duty-cycle KG notes).

---
### TASK-45 STATUS (2026-07-18 FIXED) — ✅ decentralized assign + deep-J + GROUPS>1 is CORRECT on silicon
The wrong-C is FIXED. Two `gpt-5.6-sol` (independent, verify-at-file:line) found the bugs my static self-review missed —
both FEED-side (I was fixated on the bank lifecycle):
1. **FEED-ABA** (`.Lflow_da_stamp`): published `SL_STI` AFTER resetting the feed claim-counters (`SL_BFNEXT`/`SL_ARNEXT`);
   the feed reads `SL_STI` post-claim with no `SL_GEN` recheck, so a feeder delayed across a `g→g+POOL_N` reuse claimed
   the new gen's reset counter but staged with the OLD STI → wrong K-segment (work-exact, ~2.3×). FIX: publish `SL_STI`
   BEFORE the counter resets (`SL_GEN` still last). → bad 3814→200 (95%).
2. **COLD-START gen alias**: `SL_GEN` init=0 ALIASED first real generation 0 → the feed's `SL_GEN==cursor` gate passed on
   the UNSTAMPED slot 0 → double-staging → over-large (the residual race). FIX: init `SL_GEN`=`0xFFFFFFFF` sentinel under
   `.if DECENTASN` (coordinator keeps 0, byte-identical). → bad 200→0.

**CONFIRMED CLEAN (full stride=1 oracle, x2 each):** DECENTASN=1 J=1 GROUPS=1 (`c996bb73`) bad=0; **DECENTASN=1 J=2 GROUPS=3
(the TARGET, `29239903`) ok=27648 bad=0 max_rel=0, computed=221184 EXACT, occ[0]=0.** DECENTASN=0 byte-id `02faf45a` intact.
Staged bin = `29239903`. **NEXT: this was UNFED (K=1024 bounded correctness). The remaining goal is the FED deep-K run
(Gate D) to measure the actual traveling-peak / stagger+baton engagement** (does the assign-bound lift, budget bind, baton fire).

---
### TASK-45 STATUS (2026-07-18 LATE) — BUILT + review-clean, but SILICON found a wrong-C; localized to the coupled-cursor CORE
**BRING-UP FAILED (oracle BAD).** After all 4 Codex fixes + re-review-clean, the greenlit deep-J J=2 GROUPS=3 bring-up ran
clean/work-exact (computed=221184) but oracle **bad=3857 max_rel=0.19**. Full stop, offline root-cause via a DIAGNOSTIC LADDER
(all same bounded shape K=1024 n_kseg=32 MTL=6 NTL=64 CHUNK=96 full-oracle):
| config | bin | oracle |
|---|---|---|
| coord DECENTASN=0 J=2 GROUPS=3 | 677dc1b2 | **CLEAN bad=0** → shared deep-J×GROUPS compute is CORRECT |
| DECENTASN=1 J=2 GROUPS=3 | 93927e2e | bad=3857 max_rel=0.19 |
| DECENTASN=1 J=1 GROUPS=3 | fc2a017a | bad=4196 max_rel=2.22 → **not deep-J** |
| DECENTASN=1 J=1 GROUPS=1 (G=6) | d69dd3fa | bad=3814/9216 max_rel=2.29, **computed EXACT** → **CORE broken** |

**ROOT CAUSE localized:** a DOUBLE-COUNT race in the CORE coupled cursor (not deep-J, not GROUPS). Signature: work-EXACT (every
segment computed once) but ~41% of C frags ~2.3× = a bank carrying RESIDUAL (tile computes into not-zeroed banks / a prior
flush lands after the zero / C-store reads mid-accumulation or fires twice / cross-wave LDS ordering gap). **Statically it looks
impossible** (tile boundary is drain-gated `DRAIN==ASSIGN` ⇒ all J=1 flushes drained since RBDONE++ is post-`s_wait_dscnt`, then
`zero_banks` fences, then `DA_ZDONE` release) — so it's a subtle decentralized race the coordinator's single-writer sidesteps.
**IN FLIGHT:** independent `gpt-5.6-sol` Codex review (codex task, withheld my hypothesis) on the tiny J=1 GROUPS=1 core surface
to pinpoint the double-count mechanism. KG `3fff2c1a` (ladder), `1112b22d`. DO NOT ship; the coupled cursor is wrong-C at the
simplest config. (Codex model default IS gpt-5.6-sol per ~/.codex/config.toml; now pinned explicitly.)

---
### TASK-45 STATUS (2026-07-18) — decentralized assign + DEEP-J + GROUPS>1 BUILT (the REAL target); awaiting pre-silicon gate
The target kernel = decentralized assign + **deep-J** + **GROUPS>1** + stagger + baton, banked. deep-J is a REQUIREMENT
(kmbandy, 2026-07-18: "not a choice — it's the entire design"), not optional. The morning "drain-gate peek-first" fix is
SUBSUMED by this rewrite.

**WHY it was more than a flag (the real blocker):** deep-J's carrier walks consecutive pool POSITIONS (`slot=cursor mod
POOL_N`, `.Lflow_jloop` ~2892) trusting each carries the next consecutive ksi. The decentralized assign grabbed ksi and
reserved the slot as TWO separate atomics → position vs ksi **permute** → the carrier's J-window scatters across unrelated
ksi → **silent wrong-C**. deep-J+DECENTASN had NEVER assembled (the `.error` at line 786 always blocked it), so the design
doc's §6 "just wire the same poison" was a latent break.

**THE FIX (all `.if DECENTASN`):**
1. **COUPLED CURSOR** — ksi is DERIVED from the reservation index (`within = r − DA_BASE`; `ksi = within & mask`;
   `group = within >> shift`), NOT grabbed separately → pool **position == ksi order** → carrier J-window aligned. Removed
   `DA_KSI/DA_NEED/DA_CLAIMING`. New state: `DA_BASE`(140), `DA_TILE`(144), `DA_ZDONE`(=`OP_BASE−4`=508; top bit `ZLOCK`
   = boundary lock). `DA_ZDONE` gates reservations (`r < DA_ZDONE`) so a group's banks are drain-gated + zeroed before its
   ksi are handed out → **GROUPS>1 for free**. NO over-reservation (boundary hits exactly at `ASSIGN==DA_ZDONE`, handled by
   ONE wave via `ZLOCK` CAS, drain-gated non-blocking). Init `DA_BASE = −(GROUPS<<shift)` so the first reserve claims the
   first tile. New labels `.Lflow_da_peek/_boundary/_bnd_tile/_bnd_giok/_bnd_bail/_bnd_term`.
2. **S1 poison SIMPLIFIED** (no encoding widen, no `side_final` change): EVERY slot stamps `RB_PENDING` (arms to 0); non-lead
   slots are turned away from the CLAIM by the pre-grow lead-gate **+ a new post-grow lead RE-CHECK** (`DECENTASN&&J>1`,
   re-reads SL_STI at the re-derived slot), NOT by an `ACC_N` poison. The carrier reaches non-lead slots via its cursor walk
   (bumps `SL_RBDONE`, never `SL_RBNEXT`). Removed the `JDEPTH>1` `.error` guard.

**OFFLINE GATES — ALL PASS:** deep-J J=2 GROUPS=1 (`cebd98dd`) + **deep-J J=2 GROUPS=3 (`57802557`, the target)** + GROUPS=3
J=1 (`8cf2fdfb`) all assemble **0-spill**; **DECENTASN=0 byte-identical `02faf45a`**; hot-loop clean (no new spin/blocking-read,
boundaries bounded, `deadman_check_fat` intact). `J=4 GROUPS=3` fails ONLY on LDS budget (POOL_N=4 at G=18 >64KB) — geometry,
not code. Dead-but-unreachable: `.Lflow_da_rollback`/`.Lflow_da_termslot` (harmless; remove later).

**STAGED ARTIFACT (NOT yet run):** `occ_dsws2_w30_flow_gd.bin` = **`57802557`** at DECENTASN=1 FM=1 **G=18 ACC_N=6 GROUPS=3
POOL_N=2 SEGK=32 JDEPTH=2** STAGGER=1 BATONGATE=1 MSDRAIN=1 RBU=1 WAVES=30.

**PRE-SILICON GATE:** (a) **O1 enumeration** — DONE (`DSWS_O1_ENUMERATION_2026-07-18.md`). (b) **Adversarial Codex review**
(codex session 019f7289) — REFUTED the first build with 4 findings; **core coupled-cursor design CONFIRMED sound** (position==ksi,
boundary election, ASSIGN-freeze, interlock, release-ordering); gaps were consumer-side. **ALL 4 FIXED 2026-07-18** (KG `3d7cd735`):
- **D1** (CRITICAL, wrong-C @ POOL_N=2/J=2): the DECENTASN claim CAS clobbered the jloop cursor `s46`; nothing reloaded it →
  wrong-index carry walk. FIX: reload `s46 = SL_GEN[claimed]` (= reservation index r) before `.Lflow_jloop`; + build guard
  **`POOL_N % JDEPTH == 0`** (kills the lead↔non-lead ABA). ⇒ POOL_N must be a multiple of J now (J=2 → POOL_N=2).
- **A1**: `ZLOCK` → bit 0 (DA_ZDONE always a multiple of n_kseg≥2 → structurally free; no 2³¹ alias); + `n_kseg==1→terminal`.
- **B1**: the redundant bare `deadman_check` in `.Lflow_jwait` → `.if !DECENTASN` (silent fat-retire gone for DECENTASN; coordinator
  bytes preserved). ⚠️ this was the byte-id break — B1 is in the SHARED jwait path; MUST stay `.if !DECENTASN`.
- **C1**: new per-WG `GSTORED` (LDS `OP_BASE−8`); C-store owner bumps it after `s_wait_storecnt`; boundary requires
  `GSTORED >= z>>shift` before `zero_banks` (non-blocking) → banks can't be zeroed while a C-store reads them.

**RE-REVIEW (codex, done):** A1/B1/C1 **CLOSED** (Codex traced each invariant). D1 canonical race repaired, but a residual:
the assembly didn't enforce `JDEPTH | n_kseg` (counterexample J=4 POOL_N=4 n_kseg=2 → a J-window straddles a group → circular
carrier/boundary wait → deadman retire + incomplete). **FIXED**: runtime fail-safe in the peek (`.if JDEPTH>1`):
`(n_kseg & (JDEPTH-1)) != 0 → terminal`. Fail-safe = clean terminal (work-exact-detectable), never wrong-C/wedge. **All 4 addressed.**

**CONSTRAINTS NOW ENFORCED:** `POOL_N % JDEPTH == 0` (build `.error`), `n_kseg % JDEPTH == 0` + `n_kseg >= 2` + pow2 (runtime
fail-safe terminal). **RE-GATED:** DECENTASN=0 byte-identical `02faf45a`; target deep-J J=2 GROUPS=3 = **`93927e2e`** (on disk),
J=2 GROUPS=1 = `48332592`, all 0-spill.

**NEXT = SILICON (awaiting greenlight):** one `./gpu_run.sh`, **full stride=1 oracle**, expect `bad=0` +
`computed == G*MTLsuper*NTL*n_kseg` (work-exact) + `occ[0]=0` + no DMFAT/reset. ⚠️ NEVER `FORENSICS=1`. Host geom MUST match:
DSWS2_G=18 DSWS2_ACC_N=6 FLOW_POOL_N=2 DSWS2_SEGK=32 FLOW_WAVES=30, n_kseg=K/32 a multiple of J=2, FEED deep-K.
Staged: `occ_dsws2_w30_flow_gd.bin=93927e2e`. KG: `6c8a2b03`→`3d7cd735`→`b0a4cc11`. Byte-id anchor `02faf45a`.

---

## CANONICAL BUILD (the current working config — every knob, with why)

```
FM=1  G=<geom>  ACC_N=<geom>  POOL_N=<LDS-legal>  WAVES=30  SEGK=<geom>  \
WOFLUSH=0  BANKZERO=1  JDEPTH=<geom, pow2 | n_kseg>  MSDRAIN=1  \
STAGGER=1  BATONGATE=1  DECENTASN=0  RBU=1  STAGINSTR=1  TFPROBE=1  ./build_flow.sh
```

### LOAD-BEARING KNOBS — do NOT change without logging a decision + reason HERE
| knob | value | why (evidence) |
|---|---|---|
| `WOFLUSH` | **0** | banked LDS reduce (fast/correct). WOFLUSH=1 = the killed flush-wait (~97% of clock, off the table). |
| `BANKZERO` | **1** | pre-zeroed banks → every ksi a pure `ds_add_f32`. Required by the banked completer. |
| `MSDRAIN` | **1** | **MANDATORY for POOL_N>1** — POOL_N>1 completes slots out-of-order; MSDRAIN=1 is the head-gated drain WALK. Without it: silent wrong C (`c6a4ae7c`, bad=5/11→0). |
| `STAGGER`/`BATONGATE` | **1/1** | the traveling-peak baton = **pure notification** (poke-at-shrink + wake-sleeper). `STAGGER=0` MUST stay byte-identical (`22bc8d0d` at the baton geom; `386dc28` at FM=2 G=3). |
| `DECENTASN` | **0** | ⚠️ see RECONCILE #1. Verified path = WG-local coordinator. |
| `GROUPS` (=`G/ACC_N`) | derived | **FIXED 2026-07-17.** `ACC_N<G` decouples compute breadth (G) from LDS banks (ACC_N). GROUPS=1 byte-identical `22bc8d0d`. |
| `JDEPTH` | pow2, divides `n_kseg` | deep-J flush amortization (fed: J1 5.2→J32 22 TF). Under throttled-stagger needs `J≤POOL_N`; **bypassed under BATONGATE=1**. |
| `RBU` | **1** | required under DECENTASN poison; kept 1 (load-batching saturates at 1). |

---

## LOAD-BEARING MECHANISMS / INVARIANTS (how it works — must keep working; not knobs)

- **Lazy role accounting (emergent economy — the "river").** Every wave reads its own `ROLE[wid]` LDS mailbox
  each loop and *simply is* that role; a **stale read = last role = coast**, always valid → **NO blocking
  read anywhere in the hot loop.** The coordinator only *lazily* nudges roles (`COORD_PERIOD` cadence; waves
  coast when idle). Roles EMERGE at runtime (no baked compute/feed mix). Code: `.Lflow_body`/`.Lflow_dispatch`
  (~2691–2714), ROLE mailbox `ROLE_BASE`. **This is the core invariant — never add a blocking read / cap /
  wait / hard partition to the hot loop.**
- **Same-WG-combine constraint (G1).** Split-K partials combine on-chip **SAME-WG only** (LDS banks) OR in
  DRAM cross-WG (WOFLUSH, the slow path). There is NO cheap cross-wave register combine. ⇒ a WG must claim a
  WHOLE tile's `n_kseg` slices to reduce them in its per-WG banks. This is WHY global-`DECENTASN` was refuted
  (it scatters a tile's slices cross-WG) and why the coordinator path is WG-local. Any future decentralization
  must be **intra-WG**.
- **MSDRAIN head-walk drain.** With `POOL_N>1`, slots complete OUT OF ORDER; DRAIN must WALK from the head
  (advance only while `head` slot is fully done), never free a non-head slot early. `MSDRAIN=1` mandatory.
- **TILEDONE per-group completer.** The C-store fires once the whole tile/group is reduced (`TILEDONE[group]`
  reaches `n_kseg*ACC_N`), elected by **first-crosser** (`old<target<=new`) — NOT exact-equality (that dropped
  stores on overshoot; fixed 2026-07-17). Any wave of the tile can own the store.
- **Baton = pure NOTIFICATION.** Poke-at-shrink + wake-sleeper; never a gate/cap/seed. Un-notified waves grow
  on their own (grow-fail→coast is fine). It only fixes TIMING to keep ≥1 wave at peak.
- **Fed-only verdicts (G3).** NO throughput/architecture verdict from <1s steady state or an under-fed shape
  (few tiles/WG). Feed via deep-K. A run that finishes instantly is a bug in the test, not a result.

### Host geometry MUST match the bin or the WG silently never launches (`3c62677a`)
- `DSWS2_G`, `DSWS2_ACC_N`, `FLOW_POOL_N`, `DSWS2_SEGK`, `FLOW_WAVES=30` must equal the build.
- Shape via `DSWS2_ORACLE_MTL`/`_NTL`/`DSWS2_K`: `Mo=(G*16*FM)*oMTL`, `No=(FN*16)*oNTL`, `n_kseg=K/SEGK`.
- **FEED via deep-K** (`DSWS2_K` large); guard ON (`ML8_COOP_CHUNK` bounded, `ML8_COOP_CHUNK_MAXS=3.0`, never CHUNK=0).

---

## CURRENT ARTIFACTS (verified state)

- **On-disk bin:** `248ef859` = G=18 ACC_N=6 GROUPS=3 POOL_N=2 SEGK=32 J=2 (the GROUPS>1-fix proof bin).
- **Byte-identity anchor:** `STAGGER=0` (or GROUPS=1 equiv) → `22bc8d0d` at G=6/ACC_N=6/POOL_N=3/SEGK=64/J=32.
- **3 files uncommitted** (`occ_kernel_dsws_flow.s`, `build_flow.sh`, `occ_dispatch.cpp`). Latch clear.

## DONE / VERIFIED (2026-07-17)
- **GROUPS>1 fully fixed + full-oracle verified** (task #44). Three bugs: (a) bank-residue → group-boundary
  `zero_banks` at `.Lflow_same_tile`; (b) arbitrary-K dense walk → COUNT-based cursor advance; (c) store-election
  overshoot → first-crosser election (`old<target<=new`), GROUPS>1-gated. Proof: G=18 GROUPS=3, full stride=1
  oracle `ok=9216 bad=0`, computed EXACT (was `bad=72`/17856-short). GROUPS=1 byte-identical throughout.
  ⚠️ ALL of today's GROUPS>1 work + fix are on the **`DECENTASN=0` coordinator path** — NOT ported to any
  decentralized assign path (the cursor/boundary fixes live in the wid0 block).

## KNOWN WALL (measured, but on `DECENTASN=0 BATCHASN=0` — see RECONCILE #1 before trusting)
- Even fed at G=18, `door4 grow-fail=0` (budget never binds), `occ[86]` ASSIGN-BOUND ~92%, baton inert.
- Budget-binding is DOWNSTREAM of assignment throughput. **BUT** this was measured single-lazy-coordinator
  (`BATCHASN=0`); re-measure with the actual intended assign path once RECONCILE #1 is settled.

## NEXT (after RECONCILE #1)
- Establish the real assign-throughput baseline on the *correct* assign path (BATCHASN/intra-WG-decentralize).
- Only then: does the budget bind → does the baton engage.

---

## SAFETY FLOOR (never violate — CLAUDE.md + gpu_run.sh)
Dispatch ONLY via `./gpu_run.sh`; ONE per greenlight; changed kernel = ONE bring-up then STOP; hang/DMFAT/
oracle-BAD/INCOMPLETE = full stop; DEADMAN 0.5s NEVER raised; FEED ≥1s steady before any TF verdict; check
`computed == G*MTLsuper*NTL*n_kseg` every run (short = dropped work); **full stride=1 oracle for any correctness
claim — sampled oracle gave false-CLEAN twice (2026-07-17)**.

## UPDATE PROTOCOL
Last action of EVERY session: update the CANONICAL BUILD, LOAD-BEARING KNOBS, CURRENT ARTIFACTS, DONE, and NEXT
sections to match reality, and add/clear RECONCILE items. If a knob changed, log why in the table.
