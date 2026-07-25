# DECENTASN × banked deep-J — design spec (2026-07-16)

**Status:** design approved (kmbandy, 2026-07-16), pre-implementation. Supersedes the WOFLUSH-bound
DECENTASN pin work. Read `DSWS_STRATEGY_2026-07-15_NIGHT.md` first for the strategic frame.

All work in `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_dsws_flow.s`
(3452 lines) + `occ_dispatch.cpp` (host). Baseline inertness invariant: `DECENTASN=0` → md5 `386dc28`.

---

## 0. The decision

Marry DECENTASN's decentralized (assign-is-a-role) accounting to **deep-J register accumulation + a
single lazy carry-off to DRAM**, i.e. move the assign economy OFF the `WOFLUSH=1`/`JDEPTH=1` substrate and
ONTO the **banked path** (`WOFLUSH=0 BANKZERO=1`) with `J>1`. The night's measurement + first-principles
walk established that the flush is ~97% of the clock and WOFLUSH is the slow way to pay it; the banked
LDS reduce + one C-store per tile is the proven-correct fast path (the baseline coordinator already runs
it clean on every real shape).

**The one structural decision that shapes everything (kmbandy's call):** *retire the (next,inflight) pin;
make TILEDONE the completion authority.* See §3.

---

## 1. The framing (what "banked DECENTASN" IS)

> **Banked DECENTASN = the proven baseline banked completer (TILEDONE-owned single C-store, RBDONE
> head-walk drain + the "new tile needs the pool fully drained" bank-reuse barrier) + DECENTASN's
> decentralized any-wave claim + DECENTASN's `SL_GEN` generation-tag gates. The (next,inflight) pin is
> deleted.**

This is deliberately a *small delta over code that already works*. The baseline banked completer
(`occ_kernel_dsws_flow.s` 2949–3060) is oracle-clean on all real shapes. We are not inventing a new
completion protocol; we are letting **any starved wave** produce/claim into it instead of a single
coordinator, and keeping only the DECENTASN pieces that are orthogonal to completion (the decentralized
stamp + the `SL_GEN` staleness gates).

---

## 2. The stale-guard finding (de-risks Gate 1)

`occ_kernel_dsws_flow.s:697` guards `DECENTASN && !WOFLUSH` with: *"SL_GEN aliases the write-once
store-claim in the non-WOFLUSH completer."* **This premise is stale.** Verified by tracing every `SL_GEN`
reference in the file:

- **Every reader/CAS of `SL_GEN` is `.if DECENTASN`-gated**: feed pick (3095–3102), drain gate (883–886),
  two compute gates (3132, 3329–3331), the stamp release-fence (3233–3234, 3262–3263).
- **The only non-DECENTASN `SL_GEN` references are two resets to 0**: init (2388) and the coordinator
  assign path (2528–2529). The 2529 comment still calls it a "single-winner bank store" store-claim.
- **But the current banked completer elects its store owner via `TILEDONE`** (2961–2970), **not** `SL_GEN`.
  The `SL_GEN` store-claim was superseded by the 2026-07-13 tile-scoped completer and is now dead code.
- Under DECENTASN the coordinator assign path (where 2528 lives) is not even reached — line 2429 branches
  every wave to `.Lflow_body`, away from the privileged-coordinator role.

**Consequence:** there is no live collision. DECENTASN's `SL_GEN` generation-tag spine coexists cleanly
with a banked TILEDONE completer. Guard 697's stated reason no longer holds; it becomes a no-op to remove.

---

## 3. Why retire the pin (the coherence argument)

The (next,inflight) pin packed `inflight_claims` into `SL_RBNEXT[15:8]`; a claim did `+1+INFLIGHT_ONE`,
completion did `-INFLIGHT_ONE` after the flush, and DRAIN gated on `next==ACC_N && inflight==0`. It was
added to make drain-authority atomic with the claim (closing the ~2.4% WOFLUSH straddle). On silicon it
**over-released** (a stray `-INFLIGHT_ONE` underflows an already-0 inflight field → `0x…06` borrow →
manufactured `RB_PENDING` → head-of-line drain stall; measured occ[97]≈800, ~4% of completes). The
claim-persistence diagnostic then proved claims themselves are fine (occ[95]=occ[96]=0) — the pin's only
measurable effect was its own release race.

Retiring it makes both gates line up:

- **Gate 1:** deleting the `inflight` field deletes the entire over-release failure class. Drain reverts
  to the banked model (RBDONE head-walk + tile-level drain barrier), which is proven correct.
- **Gate 2:** guard 706 exists because the J-poison (`SL_RBNEXT=ACC_N` on non-lead slots) makes a slot
  look "exhausted" (`next==ACC_N`) → a **pin-based** drain gate would free it before the J-carrier
  flushes. With drain authority on **TILEDONE** (waits for `n_kseg*ACC_N` segments to actually land, which
  can't happen until every carrier flushes), the J-poison no longer trips premature drain. The collision
  dissolves.

**What the pin protected that we still owe a replacement for:** reuse-during-compute (a slot being
re-stamped/re-fed while a wave is still reading its operands or writing its bank). That is covered WITHOUT
the pin by the combination in §5. This is the crux the spec must prove before silicon.

---

## 4. Gate 1 — decentralized claim onto banked, J=1

Build: `DECENTASN=1 FM=1 G=6 ACC_N=6 POOL_N=<LDS-legal, recheck> SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=1
MSSCAN=0 RBU=1 STAGINSTR=1 TFPROBE=1`. Oracle shape: `ml8 moe attn_q` 576×4096×2048 (n_kseg=32 pow2).

Changes (all `.if DECENTASN`-gated; `DECENTASN=0` must stay byte-identical `386dc28`):

1. **Guards (660–707):**
   - Remove/flip `697` (`DECENTASN && !WOFLUSH`) — stale per §2.
   - Keep `700` (MSSCAN=0) and `703` (RBU=1) as-is.
   - `706` (JDEPTH=1) stays for Gate 1; relaxed in Gate 2.
   - **LDS re-algebra:** banked adds the ACC banks back (`ACC_BASE`/`ACC_STRIDE`, `ACC_N` banks) that
     WOFLUSH omitted. Recompute `LDS_TOTAL_FLOW` and re-check `POOL_N` against the 64 KB cap. WOFLUSH
     DECENTASN ran POOL_N=4; banked may force POOL_N down. Record the actual number; this gates whether
     the pool can still hide the round-trip.

2. **Compute claim (~2653–2746, the DECENTASN claim block):**
   - Keep the `RB_PENDING` poison-until-staged arming (`side_final`, 941; the `A_PENDING/B_PENDING` bits).
     This is orthogonal to the pin and still required so a claim cannot read an unstaged slot.
   - Change the claim CAS from `x → x+1+INFLIGHT_ONE` back to `x → x+1` for `x < ACC_N` (a staged slot's
     `next` field only). A `RB_PENDING`/exhausted/lost slot → COAST (help), never wait — unchanged.
   - Remove all `INFLIGHT_MASK`/`INFLIGHT_ONE` handling from the claim.

3. **Flush (2872–2912, already the `!WOFLUSH` path):** the claimed segment does `ds_add_f32` into bank[r]
   (`acc_base_of` + `v_add_nc_u32 v12,v10,s39`), `s_wait_dscnt 0x0`. BANKZERO=1 → banks pre-zeroed →
   always `ds_add_f32` (no fresh-write branch).

4. **Release / completion (2914–2937):**
   - **Delete** the minimal pin-release block (2921–2937) entirely.
   - Keep `RBDONE++` (2916–2917) — now a *live* drain gate again (it was diagnostic-only under the pin).
     It is bumped AFTER the `ds_add_f32` drains, so RBDONE==ACC_N ⇒ every rowblk's bank write is visible.
   - Keep the TILEDONE bump (2961–2970) and the tile-closer C-store (2988–3034). Any wave of the tile can
     own the store (all ksi share t → same mblk/tcol).

5. **Drain (drain_advance, 866–911):**
   - Keep the DECENTASN `SL_GEN==DRAIN` head gate (883–886) — prevents advancing past a reserved-but-
     unstamped slot.
   - **Replace** the pin branch (887–905: pending / `next==ACC_N` / `inflight==0`) with the baseline
     `SL_RBDONE==ACC_N` head-gate (the `.else` at 906–910). Net: DECENTASN drain = `SL_GEN==DRAIN` AND
     `RBDONE==ACC_N`, then walk.
   - Keep the new-tile bank-reuse barrier (2458–2466: new tile requires `DRAIN >= ASSIGN`, i.e. pool fully
     drained, before `zero_banks`).

6. **Host (`occ_dispatch.cpp`):** relabel/retire the occ[95/96/97] pin-diagnostic prints; add a
   work-exactness readout (`computed` vs `TOTAL_super*ACC_N`) if not already surfaced.

**Gate 1 pass criteria:** oracle bad=0 on `ml8 moe attn_q`; occ[0]=0 (clean retire); no reset; no DMFAT;
`computed == TOTAL_super*ACC_N` (work-exact, fed ≥1s); flush share and TF recorded vs the WOFLUSH 0.4 TF
and the baseline banked number. Then the 6-shape pow2 DECENTASN sweep (one greenlit dispatch each).

---

## 5. The reuse-during-compute correctness argument (the crux — verify offline before silicon)

**Claim:** with the pin gone, no slot is freed (DRAIN advanced past it → producer re-stamps → feeders
overwrite its operands/banks) while any wave is still reading its operands or writing its bank.

**The window to close:** wave W claims rowblk r of slot s (logical index d), reads A/B operands from
`OP_BASE + slot*OPSTRIDE`, does WMMA, `ds_add_f32` into bank[r], `s_wait_dscnt`, then `RBDONE++`. The
danger is slot s being re-stamped for a later generation (d+POOL_N) during W's compute.

**Why each guard holds:**

1. **Stale-operand claim (the original 2.4% straddle):** closed by the `SL_GEN` gate, which W checks at
   the compute pick (3132) and the feed pick (3095) — W only proceeds if `SL_GEN == d`. A re-stamp writes
   `SL_GEN = d+POOL_N` LAST as a release fence (3233/3262), so W either sees the old d (valid, its
   operands are the ones it will read) or the new value (bails). W cannot read half-re-stamped operands
   because operands are written BEFORE the `SL_GEN` fence.
   - **⚠ OPEN sub-question O1:** the `SL_GEN` check at the pick is *before* the WMMA/`ds_add_f32`. Between
     the check and the bank write, can DRAIN advance past d and a re-stamp land? For that, DRAIN must pass
     d, which requires `RBDONE[d]==ACC_N` — but W has claimed r and not yet bumped RBDONE, so RBDONE[d] <
     ACC_N until W finishes. **This is the load-bearing step**: RBDONE is the count of *completed* (bank-
     written) claims; DRAIN gates on RBDONE==ACC_N; therefore DRAIN cannot pass d while W's claim is
     outstanding. Under the pin this was the `inflight` count's job; under banked it is RBDONE's job. The
     spec MUST confirm: is there any claim that bumps `next` (reserving r) but is then abandoned WITHOUT
     bumping RBDONE? (e.g. a grew-but-exhausted path, a deadman force-retire of a fat wave). If yes, RBDONE
     never reaches ACC_N → DRAIN stalls (safe, not a correctness bug — a liveness/incompleteness one). If a
     path lets `next` reach ACC_N while RBDONE < ACC_N permanently, drain wedges. Enumerate every `next++`
     and confirm each is paired with an eventual `RBDONE++` or the slot is otherwise settled.

2. **Bank overwrite by a new tile:** banks are TILE-scoped and zeroed at tile claim (`zero_banks`, 2466)
   only after the new-tile barrier confirms `DRAIN >= ASSIGN` (pool fully drained, 2462–2465). A tile's
   banks cannot be zeroed while any of its segments is still draining, because the drained-pool check
   requires all POOL_N slots retired (RBDONE==ACC_N each) → all bank writes visible.

3. **Out-of-order completion:** DECENTASN claims are out of order (any wave, any staged slot), but DRAIN
   only advances from the HEAD and only while `RBDONE[head]==ACC_N` (the MSDRAIN walk). A non-head slot
   reaching RBDONE==ACC_N first does not advance DRAIN past an incomplete head. So the "free a slot still
   in use" bug the walk was built to prevent (664–669) still cannot occur.

**Verification method (offline, before any dispatch):** enumerate in this doc every `SL_RBNEXT` `next++`
site and every `SL_RBDONE++` site; show the pairing; show the DRAIN gate reads RBDONE; show `SL_GEN`
fence ordering (operands/STI before `SL_GEN`, `SL_GEN` before `ASSIGN_HEAD++`/pick visibility). If O1's
enumeration finds an unpaired `next++`, that is a Gate-1 blocker to resolve before silicon.

---

## 6. Gate 2 — J>1 register-hold + lazy carry

Only after Gate 1 is oracle-clean and work-exact. Build adds `JDEPTH=<2|4|8|16|32>` (pow2, divides
n_kseg=32).

Changes:

1. **Guard 706:** relax — with the pin gone, `SL_RBNEXT` carries `{RB_PENDING bits, next_rowblk}` only.
   The J-poison writes `SL_RBNEXT=ACC_N` on non-lead slots (2513); `next==ACC_N` now means "exhausted →
   coast" to the claim path (intended) and is NO LONGER a drain trigger (drain is RBDONE/TILEDONE-based).
   Confirm the poison value `ACC_N` is distinct from `RB_PENDING` (it is: `ACC_N < 0x100 << 0xC0000000`).

2. **J ownership (already built on the coordinator path, 2505–2514, 2746–2841):** only a group-lead
   segment (`ksi % J == 0`) admits rowblk claims; non-lead segments are poisoned to ACC_N because their
   rowblks are owned by the J-carriers holding ACC across J consecutive ksi. Wire the DECENTASN
   decentralized stamp (3208–3263) to apply the same lead/non-lead poison the coordinator applies at
   2511–2513. The register-hold loop (2746–2841) and the "flush ONCE after J segments" already exist.

3. **Lazy carry:** the "next available wave carries the finished tile to DRAM" IS the existing TILEDONE
   tile-closer — no new mechanism. TILEDONE bumps by `JDEPTH` per flush (2962–2963); the wave that brings
   it to `n_kseg*ACC_N` does the single C-store.

4. **The deep-J invariant (bad=64 lesson, DSWS_STRATEGY §5):** DRAIN MUST NEVER PASS AN UNFLUSHED SEGMENT.
   Under TILEDONE-authority this holds because a tile's C-store waits for `n_kseg*ACC_N` and DRAIN's
   new-tile barrier waits for the pool to drain. Re-verify against the J-carrier that holds ACC across J
   ksi: its intermediate (non-lead) segments never bump RBDONE until the single flush — confirm the head-
   walk drain cannot pass a slot whose lead hasn't flushed. (This is the coordinator path's existing
   invariant; the port must preserve it.)

5. **Stagger interaction:** `JDEPTH>1` pulls in the WAVES≥2*ACC_N / MAXFAT guards (182–192). At ACC_N=6
   that needs WAVES≥12 (we run 30) — fine, but STAGGER + concurrent-fat algebra must be re-checked for the
   DECENTASN wave mix (no privileged coordinator → one more compute-capable wave).

**Gate 2 pass criteria:** oracle bad=0, work-exact, at J∈{2,…,32}; flush share drops ~J-fold vs Gate 1;
TF recorded. Then the pow2 sweep.

---

## 7. Safety + measurement (non-negotiable — CLAUDE.md + DSWS_TESTING_LOG.md)

- One greenlit `./gpu_run.sh` dispatch at a time. A new/changed kernel = ONE bring-up, then STOP + report.
  A sweep is N individual greenlights, never a batch.
- Hang / DMFAT / oracle-BAD / INCOMPLETE = FULL STOP, go offline. Never dispatch onto a wedged queue.
- `DEADMAN_TICKS` stays 0.5s. Never raise it. A false kill means a missing `deadman_progress` site.
- Offline-first: assemble, disasm, static-check, do the LDS/VGPR algebra, and complete the §5 enumeration
  IN WRITING before asking for silicon.
- FEED ≥1s steady state before any TF verdict; never quote TF <1s. Check `computed == TOTAL_super*ACC_N`
  every run — a short count silently drops work and flatters TF.
- Any change that could raise HBM traffic → small chunk (`ML8_COOP_CHUNK<=1024`) + short shape first
  (rule 7: a bandwidth-saturating kernel kills the desktop without bricking the card).
- Baseline inertness: `DECENTASN=0` must stay md5 `386dc28` after every edit.

---

## 8. Open questions / risks (resolve before or during the relevant gate)

- **O1 — RESOLVED (2026-07-16), Gate 1 clean.** Enumerated every `SL_RBNEXT next++`: under DECENTASN the
  only bump is the WON CAS (`.Lflow_claim_execok` → `lds_cas_rtn`); all four abandon paths (post-grow
  catch-up, unstaged, exhausted, lost-CAS) branch to `.Lflow_cmp_tryadv` BEFORE the CAS commits, so a coast
  owes no RBDONE. At J=1 the path from won-CAS to `RBDONE++` is straight-line (decode → `.rept KSEG_STEPS`
  WMMA → banked `ds_add_f32` → `RBDONE++`); the only post-claim abandon branch (`.Lflow_jwait` +
  `deadman_check`) lives inside the `JDEPTH>1` block, compiled out at J=1. ⇒ every won claim bumps exactly
  one RBDONE; RBDONE==ACC_N is a sound drain gate. (The baseline `fetch_add` at 2717 over-bumps then bails,
  but DECENTASN does not use that path.) Gate 2 must re-run this enumeration for the J-carrier (its
  intermediate segments bump RBDONE inside the jloop) — tracked in §6.
- **O2 — ANSWERED (2026-07-16): banked forces POOL_N 4→3.** At FM=1 G=6 ACC_N=6 SEGK=64 the banked ACC
  banks (6×4096=24576B) push POOL_N=4 over the 64KB WGP limit (kernel `.error` at line 597; host FATAL at
  1837 agree). POOL_N=3 fits (55808B). So the bring-up runs POOL_N=3 — a shallower pipeline than WOFLUSH's
  4. Whether 3-deep still hides the ~100µs round-trip (or is assign-bound) is the Gate-1 SILICON question.
- **O3 (Gate 2):** the deep-J register peak vs the dyn-VGPR trapezoid / STAGGER admission — J lengthens the
  VGPR-fat plateau; confirm the traveling-peak still fits the budget at the DECENTASN wave mix.
- **O4 (strategic, tracked not blocking):** Thread B (full-K in registers, no split-K, big tile, 2 WGs/CU)
  may subsume this entire path. If Gate 1 shows banked DECENTASN is still assign-bound (O2), pivot to B
  rather than sinking dispatches into Gate 2.

---

## 9. KG / doc pointers

- Strategy + physics: `DSWS_STRATEGY_2026-07-15_NIGHT.md`; brief `MAD-305-DSWS` (7a328323).
- Run log: `DSWS_TESTING_LOG.md` (append one §4 entry per gated dispatch).
- DECENTASN debug history (KG): 7e92918f, a9cfc27e, 0afe9b7d, fe2eebf1/d2ce95e2 (direction), 15379b26
  (poison protocol). Codex thread: `codex resume 019f665b`.
- Task #42 (this work).
