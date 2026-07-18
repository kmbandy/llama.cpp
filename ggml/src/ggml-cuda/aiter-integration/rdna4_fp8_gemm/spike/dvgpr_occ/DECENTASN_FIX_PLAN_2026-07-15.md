# DECENTASN fix plan — the (next,inflight) single-word pin (Codex gpt-5.6-sol)

**Pick up here after the compact. Read this top-to-bottom, then implement §4 against the current kernel.**
All work in `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`.

---

## 0. TL;DR (one paragraph)

DECENTASN = "assign is a role" — many waves produce super-tiles into a POOL_N-slot ring in parallel, so
production isn't gated by one coordinator. The lock-free **poison-until-staged** claim protocol (Codex
design) killed the earlier deadlock and got arming ~right, but has a residual **~2.3% wrong-C race**. We
localized it *decisively by measurement*: a pinning probe (save `SL_STI` at claim, re-read at completion)
showed **675 "the slot was reused DURING my compute" events ≈ the 192 bad units** — i.e. the invariant
"a won claim pins the slot until it finishes" is **false on silicon**. Codex diagnosed the root cause and
gave a clean fix: **the pin was split across two independently-reset words (`SL_RBNEXT` for the claim,
`SL_RBDONE` for drain authority), so it was never atomic.** The fix folds an **inflight-claim counter into
the SAME word as the claim** (`(next,inflight)` packed in `SL_RBNEXT`); DRAIN requires `inflight==0`;
completion decrements inflight *after* the flush. Now the slot provably cannot reuse between claim and
completion, and it stays wait-free. Codex also caught a **second, independent bug** (feeder decodes
`SL_STI` before atomically claiming `ARNEXT/BFNEXT` → can stage the new gen with old-gen addresses).

**Why we're still on DECENTASN (kmbandy's call, and it's right):** it's the *next* assign-bound lever.
The current binding wall is the per-super-tile round-trip cost (both baseline and DECENTASN are ~97%
starved and the single coordinator is actually 13× faster today). But once the round-trip is cut, publish
rate becomes the wall and DECENTASN pays off — so getting it correct *now*, while the protocol is hot, is
worth it. See §5 for the strategic framing.

---

## 1. Current tree state (as of 2026-07-15 EOD)

- **Knob:** everything is behind `DECENTASN` (default `0`). **Inertness intact: DECENTASN=0 build md5
  `386dc28643ffb58568623ad6d89cfe62`** (FM=2 G=3 ACC_N=3 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=1). Baseline is
  correct on ALL real shapes and is the reference.
- **On-disk bin:** `c9087683754d6e2d870608587571fbac` = the pinning-probe build (DECENTASN=1 FM=1 G=6
  ACC_N=6 POOL_N=4 SEGK=64 STAGINSTR=1 TFPROBE=1). It has the poison protocol + the spin-until-staged
  ablation + the pinning probe (STISAVE / occ[95]). **These last two are EXPERIMENTS — the real fix in §4
  supersedes and removes them.**
- **⚠️ LATCH IS SET** (`.gpu_last_hang`) from the pinning-probe run (12 DMFAT / INVALID). A human clears it
  with `rm .gpu_last_hang` before any dispatch. No brick (occ[0]=0).
- **Oracle / gate shape (Rule 1, real):** `ml8 moe attn_q` **576×4096×2048** (n_kseg=32 pow2). DECENTASN
  geometry FM=1 G=6 ACC_N=6 POOL_N=4 SEGK=64 → super-tile 96×64, MTL=6, NTL=64.
  Baseline at this exact geometry is oracle-CLEAN (control run), so the geometry is fine.
- **Testing rules (kmbandy):** oracle is a REAL shape; every major change gated by the full
  ml8/mlambaformer sweep; log every sweep in `DSWS_TESTING_LOG.md`. KG: bb6bbe09.

## 2. What we proved this session (all by measurement, not argument)

1. **Bug is DECENTASN-specific, not geometry** — baseline (DECENTASN=0) at the identical FM=1 G=6 POOL_N=4
   is oracle CLEAN (bad=0). So compute/WMMA/flush/DECODE are correct.
2. **Not unstaged operands** — spinning until `BFDONE==FN && ARDONE==G` before compute only moved bad
   216→192. Operands ARE staged; C is still wrong.
3. **The pinning invariant is false** — pinning probe: `SL_STI` changed during compute **675×/run ≈ 192
   bad**. The slot is reused mid-compute. THIS is the bug.
4. **`computed ≈ expected`** (right *number* of rowblks) — so it's wrong-data / cross-gen, not gross
   double/missing count.
5. Register-collision bug found+fixed earlier: `CNT_STRADDLE=101` aliased `DM_PROG=101` (deadman progress
   flag) → corrupted every earlier `occ[95]` straddle number. (Now moot; those diags get removed in §4.)
6. **A rare, targeted fat-wait does NOT deadlock** (unlike the every-claim stage-wait) — spun 1290× with
   0 DMFAT one run, 12 DMFAT another (non-deterministic). Reusable finding; not part of the fix.

---

## 3. THE CODEX DIAGNOSIS + FIX (verbatim, gpt-5.6-sol, resumed thread 019f665b)

> **1. Root-cause mechanism**
>
> The observed reuse requires (b); (a) is where the resulting corruption becomes visible. Neither (c) nor (d) can independently cause it.
>
> Under the stated abstract protocol, the probe result is impossible. Let:
>
> ```text
> C_g = successful low RBNEXT CAS claims for generation g
> D_g = RBDONE increments for generation g
> ```
>
> If every claimant increments the same generation it claimed, then always:
>
> ```text
> 0 <= D_g <= C_g <= ACC_N
> ```
>
> A successful but unfinished claim makes `D_g < ACC_N`, so DRAIN cannot pass. Therefore a CAS-to-STI window alone cannot create the first failure.
>
> The silicon result proves that the correspondence has already been broken: at least one `RBDONE++` is landing in a different generation than its claim, or an equivalent extra/wrong-address completion exists.
>
> The concrete self-propagating interleaving is:
>
> ```text
> Generation g:
>     RBNEXT = x
>     RBDONE = ACC_N - 1
>
> W:
>     CAS(RBNEXT, x, x+1) succeeds
>     // W now owns rowblk x, but has not completed it.
>
> S:
>     // S is a stale/misdirected completion whose claim belonged to another gen.
>     fetch_add(slot.RBDONE, 1)
>     // RBDONE becomes ACC_N even though W is outstanding.
>
> D:
>     reads SL_GEN == g
>     reads RBDONE == ACC_N
>     CAS(DRAIN, g, g+1) succeeds
>
> P:
>     eventually reserves g + POOL_N
>     stamps the same physical slot:
>         RBNEXT = RB_PENDING
>         RBDONE = 0
>         STI = sti_new
>         GEN = g + POOL_N
>
> W:
>     reads or has read sti_old
>     computes while P changes STI to sti_new
>     probe observes sti_old != sti_new
>     fetch_add(slot.RBDONE, 1) now increments generation g+POOL_N
> ```
>
> W has now become the stale completion that can prematurely release the newer generation. Once seeded, the error propagates around the ring, explaining repeated reuse events.
>
> Consequently:
>
> - (b), cross-generation or duplicate effective `RBDONE++`, is the necessary enabling failure.
> - (a), reuse between claim/payload/completion, is the corruption window after that failure.
> - (c) increases the chance of addressing a recycled physical slot but cannot violate `D_g <= C_g` by itself.
> - (d) is not the initiating race because the producer writes pending before the new GEN, and feeders cannot arm the new occupant until after GEN publication.
> - Merely checking `SL_GEN` before/after the claim will detect the race but cannot safely recover: after a successful claim, bailing would leak the claim.
>
> There is also an independent feeder-side ABA hazard in the current assembly: it reads and decodes `SL_STI` before `ASTAGE_R`/`BSTAGE_R` atomically claims `ARNEXT`/`BFNEXT`. A delayed feeder can claim a newer generation while retaining the older generation's decoded source addresses. Move the STI read/decode after the successful staging claim. That can explain staged-but-wrong operands, although it does not itself explain the compute-time STI-change probe.
>
> **2. Concrete fix**
>
> Put the pin in the same atomic word as the claim. Encode both `next_rowblk` and `inflight_claims` in `SL_RBNEXT`; stop using the independent `RBDONE` word as the drain authority.
>
> For `ACC_N=6`, eight bits per field are ample:
>
> ```text
> NEXT_MASK       = 0x000000ff
> INFLIGHT_MASK   = 0x0000ff00
> INFLIGHT_ONE    = 0x00000100
>
> A_PENDING       = 0x80000000
> B_PENDING       = 0x40000000
> RB_PENDING      = 0xc0000000
> ```
>
> A staged state is:
>
> ```text
> state = next | (inflight << 8)
> ```
>
> Producer and feeder arming remain essentially unchanged:
>
> ```text
> producer_stamp(r, gi):
>     store(SL_RBNEXT, RB_PENDING)
>     store(SL_RBDONE, 0)              // optional diagnostic only
>     store(BF/AR NEXT/DONE, 0)
>     store(SL_STI, gi)
>     wait_ds()
>     store(SL_GEN, r)                 // last
>     wait_ds()
>
> second_side_final():
>     CAS(SL_RBNEXT, MY_PENDING, 0)    // next=0, inflight=0
>     wait_ds()
> ```
>
> Corrected compute claim:
>
> ```text
> compute_claim(slot):
>     x = load(slot.SL_RBNEXT)
>
>     if x & RB_PENDING:
>         coast
>
>     next = x & NEXT_MASK
>     if next >= ACC_N:
>         coast
>
>     new = x + 1 + INFLIGHT_ONE       // atomically claim k and acquire a pin
>     old = CAS(slot.SL_RBNEXT, x, new)
>     wait_ds()
>
>     if old != x:
>         coast                         // one-shot failure; no obligation
>
>     k = next
>
>     // The successful CAS has already incremented inflight.
>     // This physical occupant cannot drain or be reused now.
>     gen = load(slot.SL_GEN)
>     sti = load(slot.SL_STI)
>     compute_and_flush(sti, k)
> ```
>
> Completion must release the pin last:
>
> ```text
> compute_complete(slot):
>     wait_for_C_flush()
>
>     // Optional diagnostic. It is safe only before releasing inflight.
>     fetch_add(slot.SL_RBDONE, 1)
>     wait_ds()
>
>     old = fetch_add(slot.SL_RBNEXT, -INFLIGHT_ONE)
>     wait_ds()
>
>     assert ((old & INFLIGHT_MASK) != 0)
> ```
>
> No CAS loop is needed for completion. Claim additions and completion subtractions commute atomically, and a claimant necessarily owns one positive inflight unit.
>
> DRAIN now uses only the combined state:
>
> ```text
> drain_advance():
>     d = load(DRAIN_HEAD)
>     if d >= load(STAGE_HEAD):
>         return
>
>     slot = slot(d)
>     if load(slot.SL_GEN) != d:
>         return
>
>     state = load(slot.SL_RBNEXT)
>     if state & RB_PENDING:
>         return
>
>     next     = state & NEXT_MASK
>     inflight = (state & INFLIGHT_MASK) >> 8
>
>     if next != ACC_N:
>         return
>     if inflight != 0:
>         return
>
>     CAS(DRAIN_HEAD, d, d+1)
> ```
>
> A terminal sentinel is:
>
> ```text
> SL_RBNEXT = ACC_N                 // next=N, inflight=0
> SL_RBDONE = ACC_N                 // optional compatibility/diagnostic
> SL_GEN    = r last
> ```
>
> Why this genuinely pins the generation:
>
> - The successful claim CAS changes `(next,inflight)` atomically from `(k,i)` to `(k+1,i+1)`.
> - DRAIN requires `next==ACC_N && inflight==0`.
> - While that claimant is computing, its inflight contribution remains positive.
> - Completion removes that contribution only after the flush.
> - Therefore the producer cannot reuse the slot anywhere between the successful claim CAS and completion.
> - A stale CAS that succeeds after reuse pins the newer generation atomically; it cannot recreate the original two-word gap.
> - Failed claims still coast immediately. There is no downstream wait or spin.
> - `SL_GEN` rechecks may remain diagnostics, but they are no longer part of the correctness mechanism.
>
> I did not edit the assembly: this changes the authoritative drain state and touches claim, completion, sentinel, and drain paths. The workspace is read-only, the file already contains substantial user changes, and the repository-required GitNexus impact tool is unavailable in this session. The assembly changes should follow the pseudocode rather than guessing around those constraints.
>
> **3. Attribution**
>
> The overstrong invariant was in the original protocol design: I treated a claim in `SL_RBNEXT` as though it atomically pinned the generation through a separate `SL_RBDONE` word. It did not encode or enforce that relationship atomically. Once one completion lands in the wrong generation, the protocol has no containment and recursively manufactures more stale completions.
>
> However, an exact execution of the original abstract state machine cannot spontaneously produce the first surplus `RBDONE` increment. Thus the silicon result also proves an implementation-specific seed exists beyond the translation summarized in the prompt: a wrong-slot completion, duplicate completion path, register/address corruption, or an earlier cross-generation operation. The post-grow re-derivation and CAS-to-STI window expose and amplify that seed; they do not generate it from a consistent initial state.
>
> So the responsibility splits cleanly:
>
> - Original protocol defect: the pin was derived across two independently reset atomic words instead of represented atomically.
> - Translation-specific defect still requiring correction: whatever first violates `RBDONE <= successful claims`, plus the confirmed feeder ordering where STI is decoded before the staging claim.
> - The combined `(next,inflight)` state removes the protocol-level hole regardless of which translation path supplied the first bad completion.

---

## 4. IMPLEMENTATION PLAN — map the fix onto the current kernel

> **STATUS (2026-07-15, implemented):** Sites **A, C, D, E, F, G, H, I DONE + offline-verified.** The two
> experiments (spin-until-staged + STISAVE pinning probe) are REMOVED; occ[95]/`CNT_UNSTAGED` repurposed as a
> **pin-accounting canary** (unbalanced-release count; must read 0). Offline gate PASS: DECENTASN=0 →
> `386dc28` byte-identical (inertness intact); DECENTASN=1 assembles (12744B); disasm confirms claim CAS
> `x+0x101`, pending gate `0xC0000000`, next-mask `0xff`, pin release `ds_add_rtn 0xffffff00` + canary
> `0xff00→s102++`, drain gate pending+next≠ACC_N+inflight. Pin release is LDS-only, `s_wait_dscnt 0x0` before
> `s_alloc_vgpr`, C flush drained earlier → safe (rule 5). **On-disk bin = `b3c4a905` (pin-fix build).**
> **Site J (feeder STI-order) DEFERRED to a separate change** — see note under the table. This is a distinct
> second bug that §4-J itself flags "verify independently"; bundling it would break bisection of the pin fix.

Constraints reminder (top-of-file guards already present): `DECENTASN` requires `WOFLUSH=1`, `RBU=1`,
`JDEPTH=1`, `MSSCAN=0`. All edits DECENTASN-gated; keep DECENTASN=0 byte-identical (`386dc28`).

**First, REMOVE the experiments** (they're superseded by the real fix):
- The **spin-until-staged** block after the won claim (`.Lflow_da_reverify` / `.Lflow_da_unstaged` /
  `deadman_check_fat` loop, ~compute claim region).
- The **pinning probe**: `s_mov s[STISAVE], s17` at claim, and the completion re-read + `cnt_inc
  CNT_UNSTAGED` block (~RBDONE++ site). Also `.set STISAVE, 103` and the `CNT_UNSTAGED`/occ[95] host print
  (or repurpose occ[95] as an `inflight!=0 at claim` sanity counter if you want a canary).

**Then apply the (next,inflight) fix:**

| # | site (current approx line) | change |
|---|---|---|
| A | constants (near `.set RB_PENDING`) | add `NEXT_MASK=0xff`, `INFLIGHT_MASK=0xff00`, `INFLIGHT_ONE=0x100`. Keep `RB_PENDING=0xC0000000`. Guard `ACC_N < 0x100` and `POOL_N-1 fits` etc. |
| B | producer stamp (`~3095-3102`) | unchanged: `SL_RBNEXT = RB_PENDING`. |
| C | `side_final` macro (`~897-912`) | the second-side arm `CAS(MY_PENDING → 0)` already sets state=0 = (next=0,inflight=0). Unchanged. |
| D | prologue slot init (`~2314`) | `SL_RBNEXT = RB_PENDING` (already done). |
| E | **compute claim** (`~2600-2640`) | replace `CAS(x, x+1)` with `CAS(x, x + 1 + INFLIGHT_ONE)`. Gate: `if x&RB_PENDING coast; next=x&NEXT_MASK; if next>=ACC_N coast; CAS; if lost coast; k=next`. Read `SL_STI` after the won CAS (fine). **Remove** the spin + pinning probe. `lds_cas_rtn` handles the CAS (runtime addr). |
| F | **completion** (`~2848-2851`, after C flush) | replace/keep `RBDONE++` as diagnostic; ADD the pin release: `lds_fetch_add_r <scr>, &SL_RBNEXT[slot], -INFLIGHT_ONE` (i.e. add `0xFFFFFF00`). Must be AFTER the flush's `s_wait_storecnt`/`s_wait_dscnt`. |
| G | **`drain_advance`** (`~835-868`) | replace the `SL_RBDONE >= ACC_N` gate with: `state=SL_RBNEXT; if state&RB_PENDING stop; next=state&NEXT_MASK; inflight=(state&INFLIGHT_MASK)>>8; if next!=ACC_N stop; if inflight!=0 stop`. Keep the `SL_GEN==DRAIN` gate. (This is the crux — DRAIN authority moves from RBDONE to the combined word.) |
| H | stage-walk (`~3013-3045`) | unchanged: still gates on `(SL_RBNEXT & RB_PENDING)==0`. A freshly-armed slot is `state==0` → pending clear → advance. Good. |
| I | terminal sentinel (`~3131`) | `SL_RBNEXT = ACC_N` (next=ACC_N, inflight=0) — already ACC_N; confirm it reads as next=ACC_N (it does, ACC_N < 0x100). drain passes (next==ACC_N, inflight==0). compute finds next>=ACC_N → coast. ✓ |
| J | **feeder STI-order fix** (ASTAGE_R `~1303`, BSTAGE_R `~1245`) | Codex's 2nd bug: the macros read/decode `SL_STI` (in `.Lflow_feed`, ~line 2931) BEFORE the `ARNEXT/BFNEXT` claim inside the macro. Move the `SL_STI` read + `DECODE_STI` to AFTER the `lds_fetch_add SL_ARNEXT/SL_BFNEXT` claim, so a delayed feeder that claims a newer gen also decodes the newer gen's addresses. Re-verify SL_GEN==cursor after the claim, or re-read STI post-claim. **Do this carefully — separate bug, verify independently.** |

**Watch-outs when implementing:**
- The claim CAS `new = x + 1 + INFLIGHT_ONE`: since `next<ACC_N<0x100` and inflight small, `x+0x101` never
  overflows next into inflight. Fine.
- Completion decrement `-INFLIGHT_ONE` = `+0xFFFFFF00` via `lds_fetch_add`. Confirm the fetch-add wraps
  correctly (u32 add). `assert old & INFLIGHT_MASK != 0` can be a debug `cnt_inc` canary.
- **Order at completion:** flush (atomic_add C) → `s_wait_storecnt 0x0` (drain the C store) → RBDONE++
  (diag) → `SL_RBNEXT -= INFLIGHT_ONE`. The pin release MUST be after the flush is globally visible, else a
  drain could fire and reuse before C landed.
- Keep the deadman sites intact. No new message-bus/store near `s_alloc_vgpr`.

**Verify (offline, Rule 6):** DECENTASN=0 → `386dc28` byte-identical; DECENTASN=1 assembles; disasm-check
the claim/completion/drain use the masks. Then ONE greenlit oracle dispatch on `moe attn_q 576×4096×2048`
FM=1 G=6 POOL_N=4: **predict bad=0, occ[0]=0, occ[91] DMFAT=0**, and any inflight-canary=0. If clean →
the 6-shape pow2 DECENTASN sweep, each logged in `DSWS_TESTING_LOG.md`.

**Exact bring-up commands:**
```
env DECENTASN=1 FM=1 G=6 ACC_N=6 POOL_N=4 WAVES=30 SEGK=64 WOFLUSH=1 STAGINSTR=1 TFPROBE=1 ./build_flow.sh
# inertness: env DECENTASN=0 FM=2 G=3 ACC_N=3 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=1 ./build_flow.sh -> 386dc28
rm .gpu_last_hang    # clear the latch (human OK'd)
./gpu_run.sh decentasn_inflight_fix -- DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_SEGK=64 DSWS2_ACC_N=0 \
  FLOW_WAVES=30 FLOW_POOL_N=4 DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64 DSWS2_K=2048 DSWS2_TARGET_SECS=2 \
  ./occ_dispatch --dsws2
```

---

## 5. STRATEGIC FRAME (don't lose this)

- The bug fix makes DECENTASN *correct*. It does NOT, by itself, make it *faster* — at FM=1 G=6 POOL_N=4
  the single-coordinator baseline is 13× faster and equally starved (97%). **The current binding wall is
  the per-super-tile round-trip cost, not publish rate.** DECENTASN is the *next* lever: it pays off only
  after the round-trip is cut (bigger SEGK/fewer segments, deeper pool, cheaper frontier polling) shifts
  the wall onto publish rate. So: land correctness now (fresh context), then attack the round-trip on the
  correct baseline, then re-measure whether assign becomes the wall.
- Baseline is correct on ALL real shapes; the round-trip/staging cost (the "work item is ~600× smaller
  than its coordination cost", 2026-07-13) is the real throughput target. Also parked: the C-store
  data-width lever (HIPBLASLT_TEARDOWN, 4.1 B/insn vs 16).

## 6. GUARDRAILS (CLAUDE.md — non-negotiable)
- ALWAYS dispatch via `./gpu_run.sh`. ONE dispatch per greenlight, never a batch.
- Changed kernel → ONE bring-up run, then STOP + report.
- Hang/INCOMPLETE/oracle-BAD/DMFAT>0 = FULL STOP → offline root-cause. (DMFAT>0 = INVALID run, latches.)
- NEVER raise DEADMAN_TICKS (0.5s). Nothing new near `s_alloc_vgpr` (stores) or the message bus.
- Max work offline first; predict in writing before silicon.
- The latch is currently SET — clear only when you (human) say so.

## 7. KG pointers (mneme_search "DECENTASN")
7e92918f (bug localized/pinning), ba7d0e35 (reverify ablation), 73368b39 (unstaged diag),
15379b26 (poison impl), a9cfc27e (staged-claim root cause), 0afe9b7d (strategic control finding),
bb6bbe09 (testing rules), b700609a (36.9 ≠ "good" framing). Codex thread: `codex resume 019f665b`.
