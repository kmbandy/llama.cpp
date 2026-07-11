# DSWS — Independent Review (reviewer slug: `fable`, 2026-07-10)

Repo pin `f61ec4f85`, branch `feat/wp-dflash-ds4`. All findings below are backed by evidence I
generated myself (assemble matrix, disassembly, CPU models, host-source reads). Where I rely on
your reported silicon data (Run 5/6/7), I say so and give an offline/host discriminator to confirm.

**Evidence I reproduced first (so you can trust the rest):**
- Assemble matrix (`build_flow.sh`, SEGK=32 POOL_N=1): `ACC_N=6`→16196B, `ACC_N=3`→11624B,
  `ACC_N=2`→10072B, **0 scratch insns** in all three (`llvm-objdump -d | grep -c scratch_ == 0`).
  Byte-for-byte matches your §10 sizes.
- CPU models all pass and reproduce §10 exactly: `quiesce_race` 0/400 both orderings; `envelope_race`
  0/200; `straggler` gate=s50 on permanent block; `ctrl_model` ALL PASS.
- Control-primitive counts in the disasm are **identical across `ACC_N=6/3/2`** (`s_alloc_vgpr`=5,
  `ds_cmpstore`=2, `s_sleep`=3, `s_sendmsg`=3). Group-split changes only addressing immediates, not
  control flow. `s_lshr/lshl_b32 …,28` (the STAMP group pack/unpack) is present at `ACC_N=3/2` and
  **absent at `ACC_N=6`** → the `.if GROUPS>1` gating genuinely drops at GROUPS=1 (byte-identity
  claim's mechanism verified; see F7).

---

## F1 — [HIGH] Run 6's own signature refutes the staging/DRAIN hypotheses and points at the EXIT path — and Run 8 is instrumented to look at the wrong layer

This is my most important finding. The reasoning is entirely source + your reported Run 6 numbers.

**The host completion gate needs BOTH `occ[0]==0` AND `fenceFired`** (`occ_dispatch.cpp:255`:
`admitted && occW[0]==0 && fenceFired && …`). `occ[0]` is `+1 at admit, −1 at exit`
(`occ_dispatch.cpp:239-241`). The EOP fence (`PM4ReleaseMemoryPacket`, `occ_dispatch.cpp:236`) fires
only when every wave has reached `s_endpgm`.

**In the kernel, the `occ[0]−−` happens at `.Lflow_retire` (`occ_kernel_dsws_flow.s:1926`), which is
BEFORE the RETBARRIER spin and BEFORE `s_endpgm` (`:1949`).** The path is
`.Lflow_retire`(occ0−−) → `.Lflow_dead` → RETBARRIER check-in/spin (`:1936-1946`) → `.Lflow_endpgm`.

Therefore Run 6's reported `occ0=0 … fence=--` means: **all 16 waves passed `:1926` (so all 16
retired), but at least one never reached `s_endpgm`.** The only code between those two points is the
RETBARRIER. So Run 6 is a *post-drain exit hang*, not a staging or DRAIN-advance hang.

This **refutes H1 (staging deadlock) and H3 (completer/DRAIN) for Run 6**: if DRAIN had never
advanced, the coordinator would spin `.Lflow_drainwait`→`.Lflow_body`→`.Lflow_loop`
(`:1601-1605,1518`), which *is* deadman-covered, and the retirement would be a deadman drain at
~0.5s — not the reported +0.2s. A +0.2s retirement is *faster* than the 0.5s deadman
(`DEADMAN_TICKS=50M @100MHz`, `:337`), i.e. the **normal** coordinator RETIRE broadcast
(`:1606-1612`) fired, which only happens after `DRAIN >= ASSIGN` (all super-tiles drained). **The
group-split pipeline drained.** The bug is in the shared exit/fence path.

**Consequence for Run 8 (§8.3/§8.7 critique):** `flow_snapshot` is executed **only by the
coordinator** — `:1519-1522` branches every non-coordinator to `.Lflow_body` *before* the snapshot
call at `:1522`. It streams occ[74..80] including `occ[80]=QUIESCE_CNT` (`FDIAG_QUIESCE_OFF=320`,
`:1166`). But **the coordinator is itself one of the 16 waves that retires**; once it hits
`.Lflow_retire` it stops snapshotting. So the *last* streamed frame is a pre-retirement snapshot in
which `barrier(QUIESCE) < WAVES` is **expected and uninformative** — it just means not all waves had
checked in *yet at the instant the coordinator quit reporting*. Run 8's frontier freeze-frame
therefore **structurally cannot observe an exit-barrier / fence hang**, which is exactly the class
F1 argues Run 6 is. You will most likely see `ASSIGN=STAGE=DRAIN` all equal/complete and be no
closer to the verdict — at the cost of a possible brick.

**Offline/host discriminator (cheap, no kernel change):** `occ[0]` and `occ[80]` are plain global
words the *host* can poll every 200ms independent of the (dead) coordinator. Add both to the host
stream line (occ[0] already there; add a raw read of occ[80]=QUIESCE and occ[77..79]). Decision:
- `occ0→0` while `occ80 < WAVES` and stays there → waves are stuck in RETBARRIER **or** never reach
  it → exit-path bug (F4).
- `occ0→0` and `occ80==WAVES` but fence still `--` → all waves *did* `s_endpgm`; the fence itself is
  not firing → CP/EOP-level (F4), not kernel logic.
- `occ0` stalls `>0` with `DRAIN<ASSIGN` → *then* it really is H1/H3 and Run 8's frontier is the
  right tool.
This costs one host-side `printf` and zero GPU risk, and it disambiguates the three hypotheses more
sharply than the coordinator-only frontier freeze-frame can.

**Confidence:** high on "Run 6 is post-drain / exit-path" and on the coordinator-only-snapshot blind
spot (both pure source). Medium on the exact RTC caveat below.

**One caveat you should close first (it is load-bearing for the +0.2s argument):** the whole
"normal-vs-deadman at 0.2s" split assumes the realtime counter is ~100MHz (`:337` asserts it but I
found no measurement). If the RTC is materially faster, 50M ticks could elapse by 0.2s and the +0.2s
retirement could be *deadman-driven* — which would flip the reading back toward H1/H3. **Offline-ish
check:** in any *successful* small dispatch, read a wave's start-RTC and end-RTC (or diff two
`s_sendmsg GET_REALTIME` a known #loops apart) and compare to wall-clock; confirm 100MHz before
trusting deadman timing. This is the single assumption that most changes the diagnosis.

---

## F2 — [MED] The adaptive economy is currently inert: sense/nudge is a no-op, so the built kernel is a *static-mix* flow kernel

`.Lflow_coord_period` (`:1596-1598`) is empty — the comment says "sense/nudge deferred to a later
increment; static launch mix." `ROLE[wid]` is written only at init (`:1474-1486`) and at terminal
RETIRE (`:1607-1611`); nothing ever nudges it at runtime. So in every bin you are assembling and
running today, DSWS's headline mechanism — *runtime* producer:consumer rebalancing — **does not
execute**. What runs is a fixed seed: wid0=BFEED(coordinator), wid1=AFEED, wid2=BFEED, wid≥3=COMPUTE,
with fungibility provided only by the passive `.Lflow_coast`.

This matters for interpreting results: Run 5's "coast-storm" and the group-split hang are being
observed on a kernel whose economy is a constant, not the adaptive controller the design is about.
I'd make the sense/nudge increment land *before* spending more GPU dispatches chasing occupancy —
otherwise you're tuning a lever that isn't connected yet. **Offline experiment:** grep the disasm for
any store to `ROLE_BASE+wid*4` outside init/terminal — there is none (`grep -n 'ROLE_BASE' ` shows
writes only at `:1477-1483` and `:1609`). Confidence: high (source-exact).

Related, the seed has **exactly one dedicated A-feed (wid1)** for G=6 rowblks × FM frags of A per
super-tile, re-staged per group. With sense/nudge inert, A-staging throughput rests on wid1 + compute
coast only. That is a plausible mechanical driver of Run 5's coast-storm (compute repeatedly finds
`DRAIN>=STAGE`, coasts to A). It is a *throughput imbalance*, not a deadlock — but it's why "all
compute coasts" shows up. A two-line fix even without the full controller: seed wid1 **and** wid3 as
AFEED (or bias the coast to A when `ARDONE<G` more aggressively than to B, which `:1897-1900`
currently orders B-first).

---

## F3 — [MED] RETBARRIER is *bounded*, so it cannot itself be the 40s hang — the true suspect is the CP/EOP fence not firing on a bounded/uncoordinated exit

Following F1: with `occ0=0`, every wave is past `:1926` and the only remaining code is RETBARRIER,
which is **bounded** — `RETBAR_MAX=1000000` (`:334`) iterations of `s_sleep SLEEPN(=2)`
(`:1942-1945`). Even at ~128 clocks/`s_sleep 2`, that is ~1.3×10^8 clocks ≈ **<0.2s** per wave to
force-exit via `.Lflow_endpgm`. So kernel logic guarantees every wave `s_endpgm`s within a fraction
of a second of reaching RETBARRIER. **A 40s hang is therefore inconsistent with a kernel-logic spin
once occ0=0** — it implies the EOP fence does not fire even though all waves exit. Your own code
comment names this exact ghost: "the coordinator-broadcast retire fires the fence at 8 waves but not
16" (`:333-335,1933-1934`). That is a **PM4/CP release-mem vs. persistent-WG completion** problem,
not a frontier problem — and again invisible to Run 8's frontier stream.

**The one truly-unbounded, deadman-free spin in the wave path is NOT RETBARRIER — it is the
`s_alloc_vgpr` shrink retry** (`.Lflow_shrink :1628-1630`, `.Lflow_bshrink :1779-1781`,
`.Lflow_tashrink :1829-1831`). These `s_cbranch_scc0`-loop until the shrink succeeds, with no bound
and no deadman. But note they sit in the COMPUTE path *before* `.Lflow_retire`, so a wave stuck there
would leave `occ0 > 0` — inconsistent with Run 6. So they are not Run 6's cause, but they are a
latent unbounded-spin brick-risk independent of group-split. **Recommendation:** give the shrink
loops the same bounded escape as RETBARRIER (counter + fall-through), since a shrink that never
returns SCC=1 has no recovery today. **Offline experiment:** none needed to confirm the risk (it's a
source property); to confirm it's *not* Run 6, the host-poll in F1 (occ0 reaching 0) already does.

**Falsifiable next step for the fence itself (host-only, no new brick surface):** in a *known-good*
small dispatch, instrument the host to log the wall-time delta between `occ0==0` and `fenceFired`. If
that delta is normally ~0 but the failing config shows `occ0==0` with fence never arriving, the
defect is squarely the release-mem/EOP path and the fix is host/PM4-side (e.g. an explicit
`s_sendmsg(MSG_DEALLOC_VGPRS)` before endpgm, or a CP-side completion poll on occ0), not kernel
frontier logic.

---

## F4 — [MED, refuting a worry] STAMP `(group<<28)|sti` packing is safe for every realistic shape; the fragile assumption is the *pre-existing* n_kseg-power-of-two one, not the group bits

§8.6 asks whether `STAMP=(group<<28)|sti` (`:1566-1567`, `STAMP_GSHIFT=28`, `:314`) is collision-free
for all shapes. I re-derived it: `sti=(t<<shift)|ksi` with `t<TOTAL` and `ksi<n_kseg`, so
`sti_max ≈ TOTAL·n_kseg`. For the ml8 `down` shape (M=2048 K=9216 N=2560): MTL≈11, NTL=40 → TOTAL≈440;
n_kseg≈288 → `sti_max ≈ 1.3×10^5`. Even a very large tiling stays far below `2^28 = 2.7×10^8`. So the
group bits `[31:28]` never collide with `sti` for any plausible shape. **This worry is a non-issue —
do not spend effort on it.** (Diag shape: TOTAL=24, n_kseg=64 → sti_max=1535; trivially safe.)

The *real* latent bug in this neighborhood is unrelated to group-split: `DECODE_STI` assumes n_kseg
is a power of two (`shift=s_ff1(n_kseg)`, `mask=n_kseg-1`, `:594-607`, `:38-41`). n_kseg is derived
`KT>>NKSEG_SHIFT` (`:34`), and for `down` that is ~288 = 2^5·9, **not** a power of two → `shift`/`mask`
would silently mis-decode `(t,ksi)`. The diag runs dodge this with `DSWS2_NKSEG=64`. This predates
group-split (it's in the shared decode), but it means the group-split cannot be validated on the
*real* tall-skinny shapes until n_kseg is either forced power-of-two or decoded with a magic-div like
`/NTL` already uses (`:617`). **Offline experiment:** assemble with a defsym'd n_kseg=288 path (or a
tiny CPU model of `DECODE_STI`) and check `ksi = sti & (n_kseg-1)` vs the intended `sti % n_kseg` for
sti spanning a tile boundary — they diverge. Confidence: high on the arithmetic; medium on whether
`down` is ever actually dispatched through this kernel.

---

## F5 — [LOW-MED] The C-store group offset and the coast are correct as emitted (answering §8.6 directly)

- **Per-group C base** `+= group*(ACC_N*FM*FN*1024)` (`:1799-1802`) plus the per-bank store offset
  `r*(FM*FN*1024)` with r∈[0,ACC_N) (`:1804-1819`) yields absolute rowblk `group*ACC_N + r`, which
  tiles `[0,G)` exactly once across the GROUPS passes (G=GROUPS·ACC_N). **Collision-free and complete
  for all shapes.** Matches the A-read `actual_rowblk=group*ACC_N+local` (`:1687-1689`). Confidence:
  high (re-derived from source).
- **Coast safety (§8.6):** the coast is entered while the wave is **lean**, not fat — `.Lflow_compute`
  branches to `.Lflow_coast` at `:1644` (`DRAIN>=STAGE`) *before* the `s_alloc_vgpr NFV` grow at
  `:1653`; the grow-fail coast (`:1883-1886`) is post-*failed*-grow, i.e. "the failed grow allocated
  nothing, so we are still lean." So the coast never runs feed code with a fat allocation live — no
  VGPR aliasing, no OOR temp. (Minor narrative nit: FLOW_ECONOMY_DESIGN.md §"Fungibility" describes a
  *fat* compute wave running lean feed code; the emitted code actually coasts *lean*. Same safety,
  cleaner than advertised — worth aligning the doc so nobody "optimizes" the grow above the coast
  branch and breaks it.) **Offline experiment:** confirmed by branch ordering in the disasm; the grow
  (`s_alloc_vgpr` to NFV) has no dominator over `.Lflow_coast`. Confidence: high.
- **Frontier primitive (§8.6):** ASSIGN/STAGE/DRAIN are already **monotone, never-reset** counters
  advanced by idempotent `ds_cmpstore` (`lds_cmpstore_adv`, `:530-543`) or single-writer `++`
  (ASSIGN, `:1594-1595`); only the *per-slot* counters reset, and only at the coordinator's
  single-writer FREE gate after the prior occupant drained (`:1574-1595`, POOL_N free-gate `:1529`).
  This *is* the coop pattern §8.6 asks about, so no change needed there. The residual §3 exposure is
  **stale reads → extra coasting (perf), never lost updates (correctness)**: every counter mutation
  is a `ds_add_rtn`/`ds_cmpstore` atomic; the torn-read risk on STAMP is already clamped
  (`s_min_u32 s41,GROUPS-1` `:1667`; `DECODE_STI` t-clamp `:614-615`). Confidence: high.

---

## F6 — [LOW] §8.5 unvetted Phase-B code (`occ_kernel_dsws.s`): SENSOR FIX / OCCA_PUB_OFF

I focused my depth on the *flow* kernel (the live problem) and gave Phase-B a targeted pass.
`OCCA_PUB_OFF`/`OCCB_PUB_OFF` (`occ_kernel_dsws.s:210-211`) are **single-writer** (the claimer,
`:1313-1319`) published mid-drain ring peaks, read by followers for next-epoch conversion sense
(`:1463,1543,1783`). Single-writer + a control/hysteresis consumer means the §3 visibility gap
degrades to a slightly-stale nudge input, self-correcting across epochs — **correctness-safe**, and a
sound fix for the "post-drain occ reads ~0" problem it targets. The one thing I'd verify on silicon
is that the neutral seed `2` (`:1139`) actually lands in the CTRL band `[CTRL_LOW,CTRL_HIGH]` before
the first publish, else the first epoch nudges on a phantom. I did not find a correctness defect here.
Confidence: medium (lighter read than the flow kernel).

---

## F7 — [LOW, confirming] Byte-identity gating is real

`ACC_N=6` (GROUPS=1) emits **zero** STAMP group shifts; `ACC_N=3/2` emit the two extra
`s_lshr/lshl …,28` (coordinator pack `:1566`, compute unpack `:1665`). Every group-split edit is
`.if GROUPS>1`-gated (`:1534,1561,1664,1686,1798,1849,1903,1912`), and the control-primitive disasm
counts are identical across ACC_N. So GROUPS=1 producing byte-identical `.text` to the pre-group-split
HEAD is *mechanically* sound. **Caveat on the inference, not the fact:** "GROUPS=1 is byte-identical,
therefore the hang is group-split-specific" is weaker than it sounds. The byte-identity is of the
*code*; the failure (per F1/F3) lives in the *shared* exit/fence path, whose behavior is
timing-sensitive and differs between GROUPS=1 and GROUPS=2 (different super-tile count → different
retire schedule → the "fires at 8 not 16" fence fragility manifests differently) **with identical exit
code**. Note also your own Run 5 (`ACC_N=6`) *also timed out* ("coast-storm") — so the evidence that
GROUPS=1 cleanly *completes* at this exact config is thinner than "byte-identical to known-good"
implies. I would not treat "must be in the group logic" as established.

---

## F8 — [LOW] The CPU race models don't cover the failing kernel

`test_dsws_{ctrl,quiesce,straggler,envelope}` all model the **Phase-B** quiesce/envelope protocol
(`occ_kernel_dsws.s`), not the flow kernel's 3-frontier + POOL_N=1 group-split completion. They pass
(I reran them; §10 reproduced) but say **nothing** about Run 6. There is an offline-model gap exactly
where the live bug is. **Cheap, high-value offline experiment:** write a ~100-line C++ thread model of
the flow loop — WAVES threads, monotone ASSIGN/STAGE/DRAIN with `ds_cmpstore` semantics, per-slot
counters reset by a single-writer coordinator under the POOL_N=1 free-gate, GROUPS>1 super-tile
emission, and the RETBARRIER check-in — under both seq_cst and relaxed. It would let you decide H1/H2/
H3 (and F1/F3) **without a single GPU dispatch**, the same way the Phase-B models retired those dead
ends. This is the "cheaper offline way to decide first" §8.3 asks for.

---

## What I'd actually do next (ranked)

1. **Do not fire Run 8 as specced.** Its coordinator-only frontier freeze-frame cannot see the
   exit/fence layer that F1 localizes, and it risks a brick. First add the **host-side poll of occ[0]
   + occ[80] + occ[77..79]** every 200ms (F1) — pure host change, zero GPU risk — and re-read the
   *existing* Run 6 logs / one greenlit small run through that lens.
2. **Confirm the RTC frequency** (F1 caveat) so the +0.2s "normal-vs-deadman" split is trustworthy.
3. **Land the sense/nudge increment** (F2) before more occupancy dispatches — the adaptive economy is
   currently inert, so occupancy tuning has nothing to rebalance yet (this is also the §5 "does the
   pool bind" question: at static mix it structurally can't).
4. **Bound the `s_alloc_vgpr` shrink spins** (F3) — the only unbounded deadman-free spin left.
5. If the host poll shows `occ0==0, occ80==WAVES, fence==--`, the fix is **PM4/CP-side** (F3), not
   kernel frontier logic — chase `MSG_DEALLOC_VGPRS`/release-mem ordering for persistent WGs.

**On the group-split as the occupancy lever (§5):** the mechanism is sound and B-reuse-via-L2 is a
good instinct, but only `ACC_N=2` actually reaches 2 WG/CU (`ACC_N=3` is still 1 WG/CU per your own
table — it changes nothing occupancy-wise, so debugging the hang at ACC_N=3 buys occupancy=1 either
way). If F1/F3 are right and the blocker is the exit/fence path, that path must be fixed regardless of
ACC_N, and it's cheaper to fix at 2 WG/CU (`ACC_N=2`) directly than to first make ACC_N=3 pass.
