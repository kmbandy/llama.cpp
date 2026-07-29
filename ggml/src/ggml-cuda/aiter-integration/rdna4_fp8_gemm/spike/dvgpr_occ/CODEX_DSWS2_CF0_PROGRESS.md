# DSWS2 CF0 RETARGET — the P1/P2a/P2b stack rebuilt against `CFASSIGN=0`, 2026-07-24

**Status: ALL GATES PASS. One STOP-grade deviation from the brief, argued below (item 1) — the brief's
nominated `.Lflow_da_rollback` fix would WEDGE the WG, so it is deliberately not used; a stronger
mechanism replaces it.** OFFLINE ONLY: `./gpu_run.sh` and `./occ_dispatch` were never invoked, no GPU
dispatch, no `test_oracle`. Nothing staged (`git add`/`commit`/`stash` never run). Files edited this
session: `occ_kernel_dsws_flow.s` and `build_flow.sh` (one default value). `occ_dispatch.cpp` was
read-only — its working-tree diff is pre-existing dirt from earlier sessions and I never called
Edit/Write on it. `occ_kernel_coop.s` was never opened.

---

## 0. Baselines

| build | profile | `.text` | LDS | sha256 |
|---|---|---|---|---|
| **CF0 baseline** (new) | A1 + `CFASSIGN=0`, `DSWS2_OVERLAP=0 DSWS2_ROLEFLOW=0 DSWS2_PREFETCH=0 DSWS2_RCONV=0` | 32456B | 54784B | `128500f7314cafce9f1099d6ec6eaa2c348c406f77f07c16c79f7dfbddf73c9b` |
| CF1 canonical (unchanged) | A1 + `CFASSIGN=1`, same defsyms 0 | 32324B | 54784B | `cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553` |
| **CF0 ON** | A1 + `CFASSIGN=0 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_ROLEFLOW=1 DSWS2_PREFETCH=1 DSWS2_RCONV=1` | 34072B | 13824B | `e24e3a50cdb6d948205135fb3674e8a112d58f20fed72e5b9284200de6d983ff` |

The CF0 baseline sha was captured from the **pre-edit** source, before any change was made, and the
post-edit source reproduces it exactly. That is what makes it a real inertness gate rather than a
self-fulfilling one.

Full A1 profile used everywhere below:
`STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 DECENTASN=1 SELFSERVE=1 BATCH=1`

---

## 1. What changed (file:line, prose)

All line numbers are **post-edit, live source**.

### A. Guards re-scoped (`occ_kernel_dsws_flow.s:879-946`, `:952-981`, `:1163-1172`)

* `:920-932` — the P2a-rev2 guard `DSWS2_ROLEFLOW && DYNVGPR && !CFASSIGN` is **inverted** to
  `... && CFASSIGN`. `DSWS2_ROLEFLOW` is now a `CFASSIGN=0`-only mechanism. The banner above it records
  *why* the shared-cursor path is the one where the three premises are real, so the inversion reads as a
  decision rather than a knob flip.
* `:934-935` — `BATCH>1` guard kept, but with a **new reason**: the batch-continuation entry
  (`.Lflow_da_ss_batch_next`) re-enters `.Lflow_da_stamp` for `r+i` without passing the new grow site, so
  the commit-only stamp arm would commit an index the wave never grew for. (The old reason — the `s45`
  cohort-end value — no longer exists.)
* `:936-941` — **two new guards**: `DSWS2_ROLEFLOW` requires `DEADMAN=1` (the reversion probe is throttled
  off `s71`, `deadman_check`'s own counter) and `TRACE=0` (`s75` collides with `trace_row`'s never-wired
  `wg_id`). The `TRACE` collision was a latent hole in P2a's own audit — it argued `s75` was free
  *because* `TRACE=0` in the target profiles, but nothing enforced it.
* `:974-979` — **two new `DSWS2_PREFETCH` guards**: requires `CFASSIGN=0` (the error text carries the
  full R1-unique derivation of why the CFASSIGN target degenerates), and requires `DEADMAN=1` (the
  1-in-`DEADMAN_EVERY` bandwidth throttle). The `DA_BASE_OFF` mention was dropped from the `DECENTASN`
  guard because the rebuilt prefetch no longer reads `DA_BASE`.
* `:1163-1172` — `DSWS2_RCONV && !CFASSIGN` **relaxed** to `DSWS2_RCONV && !CFASSIGN && !DSWS2_OVERLAP`.
  The old guard was a scoping statement, not a mechanism dependency: RCONV writes only the wave's own
  role mailbox, and no reservation path (cohort math or shared-cursor CAS) reads a wave's role. The one
  combination genuinely never reasoned through — `CFASSIGN=0` *without* `DSWS2_OVERLAP`, where a converted
  feed wave picks up the ring's real LDS staging job — still errors.
* `:904-912` — `DSWS2_ROLEFLOW_BACK_N` default `16 → 2`; `build_flow.sh` line 19 shell default likewise.
  Rationale in item 3 below.

### B. Reversion trigger + mailbox CAS + hot-path throttle (`:3575-3675`)

`.Lflow_dispatch`'s `DSWS2_ROLEFLOW` arm is rebuilt end to end: throttle (`:3616-3617`), four-clause
reservability probe (`:3620-3637`), tick-counting hysteresis (`:3638-3640`), CAS'd mailbox write
(`:3643`). Details in items 2, 3, 5.

### C. Ring-compute short-circuit (`:3669-3695`)

`.Lflow_compute` under `DSWS2_ROLEFLOW` branches straight to `.Lflow_coast` instead of reading
`DRAIN_HEAD`/`STAGE_HEAD` and attempting the (provably-unwinnable) ring claim. Item 6.

### D. Prefetch retargeted onto the shared cursor (`:4380-4465`)

`.Lflow_feed`'s `DSWS2_PREFETCH` block: throttled on `s71==0`, target derived from `ASSIGN_HEAD` (now the
real reserve cursor), `DA_BASE` read deleted, SAFEPROBE-style `s_min_u32` tile clamp added. Item 4.

### E. Grow-first / reserve-after (`:4726-4780`)

The `s_alloc_vgpr NFV` moves out of `.Lflow_da_stamp` and into `.Lflow_da_realidx`, immediately before the
reservation CAS. New labels `.Lflow_da_cf0_growfail` (`:4777`), `.Lflow_da_cf0_unwind` (`:4773`),
`.Lflow_da_cf0_reserved` (`:4794`). Item 1.

### F. Defensive fat-safe phantom exit (`:4816-4830`)

`.Lflow_da_cf0_phunwind` (`:4826`). Cannot fire at `CFASSIGN=0` (argued in place); exists because if it ever did,
`.Lflow_da_sentinel` exits via `.Lflow_feedmt_sleep`, which never shrinks — a fat wave would coast
forever holding `NFV` VGPRs.

### G. Commit-only stamp arm + STAMP block excluded (`:5076-5101`, `:5162-5217`)

`.Lflow_da_stamp`'s `DSWS2_ROLEFLOW` arm is now pure commit (no grow, no grow-fail branch). The entire
normal STAMP block — including the `lds_put_r SL_RBNEXT, RB_PENDING` poison write — is wrapped in
`.if !(DSWS2_ROLEFLOW && SELFSERVE && DYNVGPR)` and does **not exist** in a ROLEFLOW binary. −372B
measured. Items 1 and 7.

### H. RCONV mailbox CAS (`:5617-5629`)

`lds_put_r s45, ROLE_AFEED` → `lds_cas_rtn s46, s45, ROLE_COMPUTE, ROLE_AFEED`. Item 2.

### I. Register-map comment refresh (`:3237-3243`), ROLEFLOW banner rewrite (`:879-899`).

---

## 2. Per-item arguments (section B of the brief)

### Item 1 — grow-fail. **FIXED, but NOT by the nominated mechanism. Read this one.**

**The brief said:** roll the reservation back via `.Lflow_da_rollback` so another wave can take it.

**Why I did not do that.** `.Lflow_da_rollback` (`:5021`, `.if !CFASSIGN`) is:

```
CAS(ASSIGN_HEAD: r+1 -> r) ; if won -> .Lflow_feedmt_sleep ; if LOST -> fall through to .Lflow_da_sentinel
```

The CAS loses whenever *any* other wave has reserved since — routine at 30 waves against ~`n_kseg`=10
live indices per field. On loss it publishes a **pre-completed sentinel** for `r`. For a phantom (its
designed use, and `.Lflow_da_termslot`'s) that is correct: the index carries no work. For a **real** index
it is not merely a work loss, it is a **guaranteed WG wedge**, and the chain is entirely in-source:

1. Nobody ever computes `(t, group, ksi)` for `r`, so nobody reaches `.Lflow_da_ss_complete` for it.
2. `TILEDONE[group]` is bumped `ACC_N` per reservation (`:5446` per-rowblk, `:4206` at complete) and the
   C-store owner is elected by the crossing of `target = n_kseg * ACC_N` (`:4217-4230`). One missing
   reservation leaves `TILEDONE` permanently at `(n_kseg-1)*ACC_N < target`.
3. So `.Lflow_cstore` never fires for that group → `GSTORED` (`:4324`) is never bumped.
4. `.Lflow_da_boundary`'s C-store gate requires `GSTORED >= z>>shift` (`:4892-4895`) and never passes
   again. `DA_ZDONE` never advances. Every subsequent reservation is refused. The WG deadmans out.

Also worth recording, since the brief's premise 2 asserted otherwise: `.Lflow_da_rollback` is compiled in
at `!CFASSIGN` but **nothing in the file branches to it** — it is preceded by
`s_branch .Lflow_da_terminal` and is unreachable dead code today. That is pre-existing and I left it
alone (deleting it would break the CF0 byte-identity gate).

**What I built instead — GROW-FIRST / RESERVE-AFTER (`:4726-4795`).** On the shared-cursor path the
commit point *is* the reservation CAS. Everything before it is a pure read of the frontier. So the grow
moves to sit immediately before the CAS. A grow-failed wave has then **not taken anything**, which
satisfies every property the brief asked for, by construction rather than by race:

* **no abandon** — there is nothing to abandon; index `r` is still at `ASSIGN_HEAD` and the next wave to
  peek reserves it. This is the brief's "so another wave can take it", achieved with zero CAS exposure.
* **no poison** — control never reaches the STAMP block. Stronger: under `DSWS2_ROLEFLOW` that block is
  **compiled out** (`:5163`), so `lds_put_r SL_RBNEXT, RB_PENDING` is not an instruction in the binary.
  The permanent-poison stall finding 4 traced cannot be created by any instruction that exists.
* **no park** — one bounded attempt, then `.Lflow_feedmt_sleep`, the same bail-and-retry-next-loop target
  every other "cannot proceed this iteration" branch in this function already uses.
* **no held reservation blocking `drain_advance`** — `DRAIN` never waits on this wave at all.

**`deadman_progress` sites (finding 4 / `CLAUDE.md` rule 4).** Two, both narrow and both on paths a wave
can loop on:
* `:4782` grow-fail — a saturated dyn-VGPR budget means *other waves are doing real compute*. Without
  this the watchdog reads "system making progress" as "this wave is wedged" and force-retires it.
* `:4769` reserve-CAS loss — another wave reserved; the WG advanced.

Deliberately **not** blanket-applied to `.Lflow_feedmt_sleep`: that would disable the anti-brick watchdog
for every coasting wave. A genuinely wedged wave is stuck at a frontier gate *before* the grow site and
never reaches either progress site, so it is still killed.

**Fat-window audit (the new risk this introduces).** The wave is fat from `s_alloc_vgpr NFV` (`:4760`) to
the CAS branch (`:4766`) — six instructions, no branches out except the two outcomes:
* CAS won → `.Lflow_da_cf0_reserved` → the ordinary carry-through, which shrinks at
  `.Lflow_da_ss_shrink`.
* CAS lost → `deadman_progress`, `s_wait_storecnt 0x0`, shrink-retry loop (the same idiom as
  `.Lflow_shrink`/`.Lflow_da_ss_shrink`), then `.Lflow_da_peek_retry` — **lean**.

No fat wave escapes into the peek loop, the boundary handler, the terminal drain, or
`.Lflow_feedmt_sleep`. Corollary proof that the grow can never double-fire: `.Lflow_da_realidx` is
reachable only by fallthrough from `.Lflow_da_peek`, whose three entries (`.Lflow_feed_empty`,
`.Lflow_da_peek_retry` after the unwind, `.Lflow_da_boundary`'s advance) are all lean. The one remaining
fat exit — the post-CAS phantom branch — is proved impossible *and* given a shrink anyway (`:4816-4830`).

**Cost, stated honestly.** A lost reservation CAS now wastes one grow/shrink pair (bounded at 8 per
`.Lflow_feed_empty` visit by the pre-existing `s48` retry budget). In exchange the physical
`s_alloc_vgpr` throttle now gates *admission* rather than *post-admission*, which is what
`:1142`'s own guard ("physical `s_alloc_vgpr` grow-fail is the only admission throttle") says it is
supposed to be: a wave reserves only work it can actually compute. This is a perf-regime change I cannot
measure offline and it should be watched on the first supervised run.

### Item 2 — ROLE mailbox race. **FIXED at both new sites, plus the pre-existing RCONV site.**

`ROLE[wid]` has two writers: the wave itself and the terminal `ROLE_RETIRE` broadcast (`.Lflow_drainwait`
`:3548-3552` and `.Lflow_da_alldrained` `:5568-5572`, both of which write **every** wave's slot).

* Reversion site `:3643`: `lds_cas_rtn s46, s45, s34, ROLE_COMPUTE`. Expected-old is `s34` = `cur_role`,
  which is exactly what `.Lflow_body` read one branch earlier (`:3559`), so the only value that can have
  replaced it is `ROLE_RETIRE`.
* RCONV site `:5627`: `lds_cas_rtn s46, s45, ROLE_COMPUTE, ROLE_AFEED`. `.Lflow_coast` is reachable only
  from `.Lflow_compute`/`.Lflow_growfail`, i.e. `ROLE[wid]` was `ROLE_COMPUTE` one branch ago.

Interleaving argument (LDS word ops are atomic, so only two orders exist):
1. CAS succeeds, then the broadcast stores `ROLE_RETIRE` → final value `ROLE_RETIRE`. ✔
2. Broadcast stores `ROLE_RETIRE`, then the CAS finds `!= expected` and **does not write** → final value
   `ROLE_RETIRE`. ✔

The old blind `lds_put_r` had a third order — broadcast, then blind overwrite — which lost the retire.
That order no longer exists. `QUIESCE_CNT` therefore still reaches `WAVES` and the collective exit does
not degrade into the `RETBAR_MAX`/deadman ~18s resident-spin compositor-starve condition.

Note the RCONV site was a **pre-existing** bug, not introduced by P2; it is fixed unconditionally inside
`.if DSWS2_RCONV` (which is 0 in every baseline build, so no gate is perturbed).

### Item 3 — reversion trigger. **FIXED: re-derived from the shared cursor.**

Old trigger `DRAIN_HEAD < STAGE_HEAD` measured the transient between a publisher's
`lds_cmpstore_adv STAGE_HEAD` (`:5296`) and its own `drain_advance` (`:5299`) — a publish artifact, not
available work, because nothing is ever ring-claimable under `DSWS2_OVERLAP` (item 6).

New trigger (`:3620-3637`) evaluates **exactly the gate the shared-cursor reserve path uses to decide it
can reserve**, clause for clause against `.Lflow_da_peek`/`.Lflow_da_realidx` (`:4665-4699`):

| clause | probe | peek's own line |
|---|---|---|
| (a) no boundary in progress | `(DA_ZDONE & ZLOCK) == 0` | `:4666-4668` |
| (b) cursor not at field end | `ASSIGN_HEAD < z` | `:4671-4672` |
| (c) next index is a REAL ksi | `(ASSIGN_HEAD & mask) <= n_kseg-1` | `:4685-4687` |
| (d) reservation window has room | `ASSIGN_HEAD - DRAIN_HEAD < SSWIN` | `:4699-4706` |

All four true ⟺ a reservable index exists ⟺ a compute-role wave would have something to do. That is a
live property of the frontier, not a race artifact. Clause (c) uses the same register-only ALU the peek
uses (no `DA_BASE` read) for the same reason the peek does: `DA_BASE` is always 2^shift-aligned (init
`-(GROUPS<<shift)` at `:3340-3342`, every advance by 2^shift at `:4961`/`:4999`), so `(r-base)&mask == r&mask`
identically.

Terminal is covered for free: `TERMFIX` keeps `ZLOCK` **held** forever once `FLOWTERM` is published
(`:5012-5015`), so clause (a) fails and no wave reverts after terminal.

**Hysteresis kept and made stronger, adaptive, no designated wave.** `s75` counts *ticks*, not passes,
because the probe fires 1-in-`DEADMAN_EVERY`. `DSWS2_ROLEFLOW_BACK_N=2` therefore means **two independent
samples 64 loop iterations apart** — a far better anti-thrash filter than 16 consecutive reads taken
inside the same microsecond, which is what the old design actually sampled. It also keeps the reversion
latency (2×64 = 128 iterations) within 2× of the opposite direction's `DSWS2_RCONV_COAST_N=64`, so the
economy is roughly symmetric instead of ratcheting one way. A skipped (non-tick) pass neither increments
nor resets the counter. Any negative observation resets it to 0 (`.Lflow_rf_noback`, `:3659-3660`). The write
is to the wave's own mailbox only; no wave is designated, nothing cross-wave is published.

### Item 4 — prefetch target. **FIXED: real target, read-only, bounded, clamped, throttled.**

* **Target is now real.** At `CFASSIGN=0`, `ASSIGN_HEAD` is the shared reserve cursor — the next index any
  wave will CAS-claim (`:4700-4702`). `(ASSIGN_HEAD + i) & mask` is therefore the `ksi` of a K-segment
  **no wave has reserved yet**, i.e. by construction not yet resident, and which some wave will self-load
  within the next few reservations. The `DA_BASE` read is deleted (2 LDS reads instead of 3).
* **Read-only.** The complete set of memory-touching instructions in the block is: `lds_get DA_TILE_OFF`,
  `lds_get ASSIGN_HEAD_OFF`, and `PREFETCH_LINES` × `global_load_tr_b64` into `v[16:17]` whose value is
  never read. No `lds_put`, no `lds_put_r`, no `lds_cas_rtn`, no `lds_fetch_add*`, no `global_atomic_*`,
  no `global_store_*`, no `s_sendmsg_rtn`.
* **Bounded.** `PREFETCH_LINES` (4), fully unrolled, no loop, no spin, no wait on any cross-wave signal.
* **Structurally address-safe (the review's recommendation, taken).** Two unconditional clamps:
  `s_min_u32 t, TOTAL-1` (`:4431`) — the SAFEPROBE clamp `DECODE_STI` applies (`:1701`), and the one
  clamp this was the only `tcol` derivation in the file to skip — and `s_and_b32 mask` +
  `s_min_u32 n_kseg-1` on `ksi` (`:4441-4442`). With the tile clamp, `mblk = t/NTL` and
  `tcol = t - mblk*NTL` are in range for *any* value `DA_TILE` could hold, including a torn read. The
  computed B address is a strict subset of the real self-serve access set for every possible value of the
  two racy reads. Worst case: warming a wrong but in-buffer line.
* **Rule-7 throttled (`:4425-4426`).** `PREFETCH_LINES × 32 lanes × 8B = 1KB` per burst. Gated
  1-in-`DEADMAN_EVERY` on `s71==0` — `deadman_check`'s own throttle counter, the same idiom
  `flow_snapshot` and `phist_bump` use. Amortized **16B per wave-visit** instead of 1KB. On the 63/64
  skipped visits the block is one `s_cmp` + one `s_cbranch` and issues no memory op at all. That removes
  the "~1KB × 30 waves × 64 WGs in the coast spin" shape the review flagged.
* **The `s_wait_loadcnt 0x0` is kept, and it is load-bearing** — I re-derived why rather than removing it
  as "a wait on the critical path": `v16:v17` is also `zero_banks`' scratch (`v16..v19`, `:1389-1392`) and
  the C-store's (`v16..v23`, `:4296-4297`). A prefetch load still in flight when this wave later reaches
  `.Lflow_da_bnd_tile` would land on top of `zero_banks`' zero pattern between its `v_mov` and its
  `ds_store` → garbage written into an accumulator bank. Removing the wait would have been a silent
  wrong-C bug.

### Item 5 — hot-path LDS reads. **FIXED, and the pre-existing pair on that path is gone too.**

The file records a 16× regression (97.3 → 5.9 TF, `:4683-4685`) from **one** extra LDS read on the
dispatch path, because at 98% coast the peek *is* the hot path.

* The reversion probe's three `lds_get`s are behind the `s71==0` gate (`:3616-3617`). Cost on 63 of 64 passes:
  one `s_cmp` + one `s_cbranch`, zero LDS traffic.
* The prefetch's two `lds_get`s are behind the same gate (`:4425-4426`).
* **Bonus:** `.Lflow_compute`'s own two `lds_get`s (`DRAIN_HEAD`, `STAGE_HEAD`) are removed entirely under
  `DSWS2_ROLEFLOW` (item 6) — they ran every pass for every compute-role wave and their answer could not
  affect the outcome.

Net LDS reads added to the per-pass dispatch path at `DSWS2_ROLEFLOW=1`: **zero on 63 of 64 passes, and
minus two on all of them.**

### Item 6 — dead ring-compute path. **STILL UNREACHABLE. Reported explicitly, and made structural.**

The brief's premise ("the ring/staged path is reachable at `CFASSIGN=0`") is **false**, and the reason is
`DSWS2_OVERLAP`, not `CFASSIGN`. The proof is exhaustive over the **writers** of `SL_RBNEXT`, which is the
only field the claim gate (`:3824`/`:3827`) tests:

| writer | value written | claim gate result |
|---|---|---|
| cold-start init `:3320` | `RB_PENDING` | rejected (pending bit set) |
| `.Lflow_da_sentinel` `:5035`, `.Lflow_da_ss_decode` `:5261`, `.Lflow_da_termslot` `:5536` | `ACC_N` | rejected (`next >= ACC_N`) |
| `side_final` `:1514`/`:1517` — **the only writer that can produce a claimable value** | `0` | never executes: its only callers are `ASTAGE_R`/`BSTAGE_R`, whose only call sites (`:4528`, `:4531`, `:5707`, `:5716`) sit in `.Lflow_feed` and `.Lflow_coast` **after** the `DSWS2_OVERLAP` branch-aways at `:4467` and `:5646` |
| the grow-fail STAMP block `:5199` | `RB_PENDING` | **compiled out** under `DSWS2_ROLEFLOW` |

So `SL_RBNEXT ∈ {RB_PENDING, ACC_N}` for the entire life of the ON build, and both are rejected. The
`.if DSWS2_OVERLAP` self-load body inside the ring claim has never executed and still cannot.

**What I did about it, rather than leaving never-executed code silently in place:**
1. Deleted the piece that *can* be deleted: the whole grow-fail STAMP block is excluded under
   `DSWS2_ROLEFLOW` (`:5163`), −372B measured. That is also the "no poison" structural guarantee in
   item 1.
2. Made the rest **structurally** unreachable instead of dynamically-always-false: `.Lflow_compute`
   branches straight to `.Lflow_coast` (`:3669-3695`), with the writer table above written out at the
   site. This is not cosmetic — the old behaviour was R1's "OVERLAP *manufactures* the grow-fail events
   ROLEFLOW exists to survive": every time a self-serve publisher opened the `DRAIN<STAGE` transient, a
   compute wave read two LDS words, **grew to NFV**, failed the claim, and **shrank**, and then went to
   `.Lflow_loop` — not to `.Lflow_feed_empty` — so it burned an entire loop iteration without ever
   attempting a reservation. Coasting directly costs nothing and lands the wave where it can reserve.
3. I did **not** delete the ring claim body itself. It cannot be excluded: `.Lflow_cstore`,
   `.Lflow_drain_adv` and `.Lflow_da_ss_complete` live inside the same region and are on the live
   self-serve path. It is needed verbatim for every `DSWS2_ROLEFLOW=0` build, including both byte-identity
   gates.

**This is the honest statement the brief asked for: the ring-compute claim is dead under `DSWS2_OVERLAP`
at any `CFASSIGN`, it stays in the file because `DSWS2_OVERLAP=0` builds need it, and under
`DSWS2_ROLEFLOW` no wave now pays anything to discover that.**

### Item 7 — phase accounting. **FIXED: `PH_GROW` measures the grow again.**

rev2 stamped `s78` *after* the grow, so the grow interval billed to `PH_WORK_WAIT` and `PH_GROW` read ~0 —
the 2026-07-22 mis-billing run backwards. Restored ordering, now split across the two sites the mechanism
occupies:

* `:4758` `phase_stamp s78` — closes the "getting to a reservable index" interval immediately **before**
  the grow.
* `:4762` `phase_stamp s80` — closes **exactly** the `s_wait_storecnt` + `s_alloc_vgpr NFV` interval into
  `PH_GROW`. Tighter than the original, which also swept in `duty_grow`/`fat_inc`.
* `:5093` `phase_stamp s78` at `.Lflow_da_stamp` — closes the reserve-CAS + decode interval into
  `PH_WORK_WAIT`, which is that accumulator's own definition ("all time since the last stamp was spent
  GETTING WORK").

No unstamped region is introduced (`phase_stamp` bills any unstamped interval forward, so a gap is a
mis-bill, not a hole). `PHASEPROBE=0` in the profile, so all three emit zero instructions in the gated
build.

---

## 3. Register / liveness audit

**Persistent registers added by this stack: exactly one, `s75`** (feed→compute reversion hysteresis),
initialised at `:3261`. Its only other occurrence in the file is `trace_row`'s never-wired `wg_id` read,
and a `.error` guard (`:940-941`) now forbids `DSWS2_ROLEFLOW && TRACE` outright instead of relying on the
profile. **`s15` is no longer used at all** in a `DSWS2_ROLEFLOW` build — the rev2 cohort served-mark is
deleted and `s15`'s init is `.if CFASSIGN`-gated (`:3268-3270`). RGA corroborates: SGPR peak **54**, i.e. −1
vs. the CFASSIGN=1 P2a/P2b builds' 55, exactly the `s15`-for-`s75` swap.

**Scratch, by site, with the argument for each:**

| site | writes | argument |
|---|---|---|
| reversion probe `:3616-3660` | `s44, s45, s46` | dead on entry: both successors (`.Lflow_compute` `:3671`, `.Lflow_feed`→`.Lflow_feed_empty`) clobber this range as their first action. Same registers the pre-existing probe used. |
| `lds_cas_rtn` (both mailbox sites) | `s49` (exec save, restored), `v11/v13/v14`, `vcc_lo` | `v11`/`v14` are `RG_A`/`RP_A`/`RG_D`/`RP_D` at `DYNVGPR=1` (`:1245-1248`) — the file's own pre-grow-safe temps; `v13` is used identically by every other `lds_cas_rtn` call site. At `.Lflow_coast` the wave is provably lean (`:5607`: "the failed grow allocated nothing, so we are still lean"). |
| RCONV site `:5627` | `s46` (CAS return, discarded) | dead: `.Lflow_compute`'s `dh`; the coast body reloads `s44/s45/s46` before any use. `s49` is re-saved by the very next instruction — the macro restores it first. |
| prefetch `:4427-4462` | `s16, s18, s19, s20, s25-s29`, `v16:v17` | all freshly written before any read. `v16:v17` is inside the `BSTG` lean window `v[16..31]`, which `ASTAGE_R`/`BSTAGE_R` already use as a global-load destination on lean waves; the in-flight window is closed by the mandatory `s_wait_loadcnt 0x0` (see item 4). Read-only persistent regs touched: `s4/s5` (Bshuf base), `s10-s14` (kernargs), `s11` (TOTAL), `s66/s67`, `s71`, `v9`. |
| grow-first `:4758-4794` | `s45` (r+1), `s47` (CAS return), `s101` (`DM_PROG`), `s87` (`CNT_GROWFAIL`), `s49`+`v11/v13/v14` via the CAS | **`s44` (=r) is never written** — verified instruction by instruction; it must survive to `.Lflow_da_stamp`. `s51` (=z) is dead after this point on the `BATCH=1` path (it is reloaded as `DA_BASE` at `:4806`). `phase_stamp` touches `s62/s63/s64/s77` and emits nothing at `PHASEPROBE=0`; `flow_gauge` touches `s57`/`v3`/`v4` and emits nothing at `FORENSICS=0`; `deadman_progress` is one `s_mov` to `s101`. |
| commit arm `:5088-5101` | counters only (`s104` = `CLAIM_NOPERSIST`) | `fat_inc`/`fat_dec` stay exactly 1:1 (inc here on the commit side, dec at `.Lflow_da_ss_rows_done`), so the unwind path needs no counterpart. `duty_grow` is provably a no-op: guard `:1145` forbids `SELFSERVE && DUTYPROBE`. |

**Untouched throughout:** `s24` (wid), `s34` (cur_role — written only at the two documented role-adopt
points), `s41` (group), `s50` (RCONV counter — deliberately reset at the reversion, not clobbered),
`s66/s67/s68` (count/mask/shift), `s69` (chunkHi), `s70/s71` (deadman, read-only here), `s72/s73` (batch,
`BATCH=1`), `s76` (magic_TOTAL).

**VGPR:** peak **48**, unchanged from every prior pass. Nothing here touches `ACC`/`FA`/`FB`.

---

## 4. Gate outputs (verbatim)

**Gate 1 — CF0 baseline established (pre-edit source), and Gate 2 — inertness (post-edit source).**
Same command; the sha below is the pre-edit capture, reproduced exactly after all edits.

```
=== GATE 1  CF0 baseline (CFASSIGN=0, all new defsyms 0) ===
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32456B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
128500f7314cafce9f1099d6ec6eaa2c348c406f77f07c16c79f7dfbddf73c9b  occ_dsws2_w30_flow_gd.bin
```
**PASS** — with `DSWS2_OVERLAP=0 DSWS2_ROLEFLOW=0 DSWS2_PREFETCH=0 DSWS2_RCONV=0` at `CFASSIGN=0`, the
bin equals the CF0 baseline sha. The changes are inert when off.

**Gate 3 — the old `CFASSIGN=1` path is not perturbed.**
```
=== GATE 3  CF1 canonical (CFASSIGN=1, all new defsyms 0) ===
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match to the canonical `cac3ff7c…`.

**Gate 4 — CF0 ON build assembles, links, 0-spill.**
```
=== GATE 4  CF0 ON build ===
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (34072B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=13824B
flow build done. fail=0
e24e3a50cdb6d948205135fb3674e8a112d58f20fed72e5b9284200de6d983ff  occ_dsws2_w30_flow_gd.bin
```

RGA (`rga_check.sh cf0_on`, linked `.co`, same defsym profile plus `RGADESC=1`, via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin` — purely static, no GPU
dispatch):
```
DEVICE,SCRATCH_MEM,THREADS_PER_WORKGROUP,WAVEFRONT_SIZE,AVAILABLE_LDS_BYTES,USED_LDS_BYTES,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,CL_WORKGROUP_X_DIMENSION,CL_WORKGROUP_Y_DIMENSION,CL_WORKGROUP_Z_DIMENSION,ISA_SIZE
gfx1201,0,N/A,32,65536,65536,106,72,0,256,256,0,N/A,N/A,N/A,28948

--- livereg: Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  54, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** **LDS = 13824B** (from the kernel's own published `.lds_total`
section; RGA's 65536 is the `RGADESC` analysis descriptor's declared maximum, not the real figure).
`.text` 34072B vs the CFASSIGN=1 P2b build's 34260B — net −188B despite four new mechanisms, because the
STAMP-block exclusion (−372B) more than pays for them.

**Gate 5 — host compiles; guards hold.**
```
23 warnings generated.
OK -> ./occ_dispatch [--prong1|--prong2]   (SUPERVISED: raw PM4 on the gfx12 node)
```
`./build.sh` completed, same pre-existing 23 `-Wformat` warnings, 0 errors. `occ_dispatch.cpp` was not
touched. `kOpBase = 512u` and its `static_assert` (`occ_dispatch.cpp:1906-1907`) hold — `OP_BASE` is
unchanged. The `dsws2Overlap`/`ldsBytesRaw` occupancy guard (`:1918-1934`) needs nothing: the host prefers
the kernel-published `.lds` section over its own computation (`:1949-1957`), and that section reads
13824B for the ON build.

**Gate 6 — `.if`/`.endif` nesting.** Full-file balance check (counting `.if`/`.ifdef`/`.ifndef` as
openers, `.endif` as closer): `final depth 0 min 0` — depth reaches exactly 0 at EOF and is never
negative anywhere in the file.

**Gate 7 — the new guards actually fire.** Each of these must FAIL to assemble, and each does, at the
line shown:
```
ROLEFLOW at CFASSIGN=1          -> :932  "DSWS2_ROLEFLOW is now scoped to CFASSIGN=0 ..."
PREFETCH at CFASSIGN=1          -> :975  "DSWS2_PREFETCH is scoped to CFASSIGN=0 ..."
ROLEFLOW with DEADMAN=0         -> :938  "DSWS2_ROLEFLOW requires DEADMAN=1 ..."
ROLEFLOW with BATCH=2           -> :935  "DSWS2_ROLEFLOW's grow-first/reserve-after rebuild is scoped to BATCH=1 ..."
RCONV at CFASSIGN=0, OVERLAP=0  -> :1171 "DSWS2_RCONV at CFASSIGN=0 requires DSWS2_OVERLAP=1 ..."
```

**Gate 8 — disassembly spot-check** (`llvm-objdump -d --mcpu=gfx1201` on the ON `.o`). Verified at the
instruction level: the reversion probe reads LDS `0x1fc` (`DA_ZDONE_OFF=508`), `0x0` (`ASSIGN_HEAD`),
`0x8` (`DRAIN_HEAD`) behind `s_cmp_lg_u32 s71, 0`; the prefetch reads `0x90` (`DA_TILE_OFF=144`) and
`0x0`, applies `s_sub_co_u32 s19, s11, 1` / `s_min_u32 s16, s16, s19`, and issues four
`global_load_tr_b64 v[16:17], v9, s[28:29]`; the grow site emits
`s_wait_storecnt 0x0` / `s_alloc_vgpr 0x50` (`NFV=80`) / `s_cbranch_scc0 →growfail` immediately before
`ds_cmpstore_rtn_b32` on `ASSIGN_HEAD`; the growfail arm is `s_add_co_u32 s87,s87,1` +
`s_mov_b32 s101,1` + branch to `.Lflow_feedmt_sleep`; `.Lflow_da_stamp` is a single
`s_add_co_u32 s104,s104,1` followed by a branch to `.Lflow_da_ss_decode`.

---

## 5. Scope discipline

Edited: `occ_kernel_dsws_flow.s` (all mechanisms) and `build_flow.sh` (one line — the
`DSWS2_ROLEFLOW_BACK_N` shell default, `16 → 2`). `occ_dispatch.cpp` read-only. `occ_kernel_coop.s` never
opened. Nothing staged. Nothing dispatched — `./gpu_run.sh`, `./occ_dispatch` and `test_oracle` were never
invoked. **The last build of this session restores the canonical `CFASSIGN=1` `cac3ff7c…` bin to disk**,
so nothing unexpected is left runnable. The `rga_out/cf0_on` scratch directory from Gate 4's static
analysis was removed afterwards.

---

## 6. STOP items / things to attack

1. **The brief's item-1 mechanism was not used, on purpose.** `.Lflow_da_rollback`'s lost-CAS fallthrough
   wedges the WG for a real index (chain at `:5035` → `:4217-4230` → `:4324` → `:4892-4895`, spelled out
   in item 1). Grow-before-reserve is the substitute. If the reviewer disagrees with the wedge chain,
   that is the single load-bearing argument in this whole document and it should be attacked first.
2. **`.Lflow_da_rollback` is unreachable dead code today** (`:5021`, preceded by
   `s_branch .Lflow_da_terminal`, no branch targets it). Pre-existing; left alone because deleting it
   would break the CF0 byte-identity gate. Worth a follow-up decision.
3. **The ring-compute path remains unreachable** and is not deleted — see item 6 for exactly why it
   cannot be, and for what was deleted instead.
4. **Unmeasurable offline:** moving the physical grow ahead of the reservation CAS changes the VGPR
   admission regime. A lost CAS now costs a grow/shrink pair, and transiently-fat waves could raise the
   observed grow-fail rate. Grow-fail is now harmless by construction, but the *rate* is a perf variable I
   cannot evaluate without silicon. First supervised run should read `CNT_GROWFAIL` (occ[73]) against
   `CLAIM_NOPERSIST` (occ[96]).
5. **Role-population bias.** Under `DSWS2_ROLEFLOW` both roles reach the same reservation peek — the only
   behavioural difference between compute and feed is *who prefetches*. compute→feed takes 64 passes
   (`DSWS2_RCONV_COAST_N`), feed→compute takes 128 (`2 × DEADMAN_EVERY`). The steady state will therefore
   lean feed-heavy, which is arguably correct for the assign-bound regime (slack waves prefetch) but is a
   tuning claim I have not measured.
6. **Work-exactness invariants preserved, and how they would surface a mistake:** `CLAIM_NOPERSIST`
   (occ[96]) increments at exactly one instruction (`:5096`), reachable only after a **won** reservation
   CAS, so it still equals `TOTAL_super` exactly. `CNT_GROWFAIL` (occ[73]) counts grow-fail *events* and
   may exceed the number of distinct indices, unchanged in meaning from rev2. Any loss introduced by this
   diff would surface as an UNDER-count on the host's `computed == G*TOTAL_super` gate (occ[71]).
