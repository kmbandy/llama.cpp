# ⛔ FULLY SUPERSEDED by `DSWS_BRIEF_2026-08-01_AM.md` (2026-07-31 evening).
> §0 (the bring-up) is **DONE** — it passed correctness and measured **FLAT (+0.44%)**.
> §2's instruction-cut list is **DE-PRIORITISED BY MEASUREMENT**: every geometry lever reaches only
> ~10% of the kernel, and the cuts were inside that 10%. See the 08-01 brief §2 before working it.
> §5's 15.4 TF baseline is **NOT COMPARABLE ACROSS SESSIONS** — the same binary measured 17.5 two days
> later. §3 (the window retraction) and §4 (the hipBLASLt structural diff) still STAND.

---

# DSWS S1 (MAD-305) — MORNING BRIEF, 2026-07-30

**SUPERSEDES `DSWS_BRIEF_2026-07-28_AM.md` entirely.** That brief already carries a superseded banner;
its §0 sweep is half-falsified and its §0.5 headline (7.70 TF) is three revisions stale.

Detail for everything here: `DSWS_TESTING_LOG.md` §21–77 (4,500 lines). KG decisions, newest first:
`baf67773` (tonight's instruction cuts) · `1d54eb1e` (auto-pool) · `313d7c13` (the 5× on MoE shapes) ·
`5b7dfbe6` (HEAD / publisher) · `06f895b5` (15.2 TF, both axes) · `af620a8b` (hipBLASLt ISA diff) ·
`8e201972` (**shapes are inputs, not levers** — read this one).

---

# 0. ★ FIRST THING: BRING UP THE FIVE INSTRUCTION CUTS ★

**Five source edits to `occ_kernel_dsws_flow.s` are IN THE TREE AND HAVE NEVER EXECUTED.** Static review
only. This is a **RULE-2 BRING-UP: ONE dispatch, then STOP AND REPORT**, whatever the number says.

```bash
cd ~/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
WAVES=6 FM=2 FN=4 G=8 ACC_N=4 SEGK=256 ./build_flow.sh     # expect sha 58e965a4… .text 28,852 LDS 34,304
./gpu_run.sh cuts_bringup_waves6_superM256_pool64_nonstd -- \
  DSWS_ALLOW_NONSTD=1 FLOW_WAVES=6 ML8_POOL=64 DSWS2_FLOW=1 \
  DSWS2_FM=2 DSWS2_FN=4 DSWS2_G=8 DSWS2_ACC_N=4 FLOW_POOL_N=1 DSWS2_SEGK=256 SSWIN=32 \
  DSWS2_K=9216 DSWS2_ORACLE_MTL=8 DSWS2_ORACLE_NTL=40 DSWS2_ORACLE_STRIDE=8 \
  DSWS2_TARGET_SECS=1.5 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85 \
  STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```

### THE GATE IS CORRECTNESS, NOT THROUGHPUT
`oracle bad=0` · `computed == 92,160 × reps` WORK-EXACT · `occ[96]` delta **+0** · `occ[0]=0` · canary clean.
**These edits change ADDRESS COMPUTATION for the operands that feed WMMA. A wrong address is a SILENT
WRONG C.** The oracle is the only thing that has ever caught this class here.

### PRE-REGISTERED EXPECTATION — WRITE THE RESULT AGAINST THIS
**Baseline: 15.4 TF** (mean of 4 repeats, ±1.3%, §59). **I expect this to land FLAT, ~15.4.**
Kimi's assessment, and I agree: *"Removing 22% of burst SALU will very likely also measure flat on that
shape."* Reasons — SALU issues on a separate pipe from VALU on RDNA4, and our own NOBLOAD/NOWMMA
ablations already showed the burst body is not the binding constraint on this shape.
**A flat result is a PASS, not a failure**, provided the oracle is clean. The change is justified by the
thesis (one kernel across all shapes — the burst must be lean where the burst *does* bind, i.e. the small
MoE shapes) and by measurement hygiene. **If it comes back materially FASTER, be suspicious and re-check
work-exactness before celebrating.** If SLOWER, that is a real signal worth chasing.

---

# 1. WHAT THE FIVE EDITS ARE

| # | change | sites | effect |
|---|---|---|---|
| 1 | **B ADDR FOLD** — `global_load_tr_b64 …, s[52:53] offset:(ni*256)` replaces an `s_add`/`s_addc` pair | 5 | −248 |
| 2 | **MI=0 FOLD** — at `mi==0` the mul/add/addc chain is a pure copy of `s[56:57]`; load on it directly | 5 | −93 |
| 3 | **MI=1 HOIST** — `mi*s32 + s[56:57]` is loop-invariant; precompute once per rowblk into `s54/s55` | 4 | −46 |
| 4,5 | dead final-step `s52 += s10`, guarded `.if ks < KSEG_STEPS-1` | 2 | −2 |

Slice 2337 → **1948** instrs (−389, −17%); `s_add_co*` 522 → **180**; `.text` 30,844 → **28,852 B**.
**Work invariants IDENTICAL at every step: WMMA 256 · B-loads 124 · A-loads 62 · stores 64 · `ds_add` 68.**

**Only #3 extends liveness.** `s54/s55` now hold the `mi==1` A base from the per-rowblk setup to
`.Lflow_da_ss_rows_done`. There is a **LIVENESS DECLARATION banner in the source** at that point — read it
before touching `s54/s55` in that region.

**The `mi==1` invariant is gated by ONE symbol, `A_MI1_HOISTED`, by design.** It was originally two
independent conditions (`.if FM > 1` at the hoist that WRITES the pair, `.rept FM` at the sites that READ
it); splitting them yields a load from an **uninitialised register — a silent wrong C, not a build error**.
**DO NOT re-split it.** The symbol change was verified a pure refactor: bin byte-identical (`58e965a4…`).

---

# 2. ★ WHERE TO KEEP CUTTING INSTRUCTIONS — THE ORDERED LIST ★

**THE K-STEP LOOP IS DONE.** After these five, the in-k-loop per-step address cost is *exactly* the
irreducible 2-instruction `s52 += s10` advance (`s10 = NT*256` is a **runtime kernarg**, so it can never
become an immediate). Everything below is **per-rowblk or per-claim**, not per-k-step.

### 2.1 THE NEXT REAL SLICE — ~17 instrs/rowblk × (ACC_N−1). BLOCKED ON REGISTERS.
1. **B-base block is fully `s33`-invariant** — the `s20/s21/s25` mul-chain plus the `s52/s53` adds, 8
   instructions. Hoist to decode, re-copy 2 per rowblk → **−6/rowblk**.
2. **A-base is LINEAR in `s33`**: `A_base(r+1) = A_base(r) + FM·s32`, because `s36` increments by exactly
   1. Precompute `FM·s32` once; 2-instruction increment at the loop bottom → **−9/rowblk**.
3. **`s43 = (s41<<2) + TILEDONE_BASE` is `s33`-invariant** → **−2/rowblk**.

**WHY ALL THREE ARE BLOCKED, AND IT IS THE SAME REASON:** they need **3 SGPRs live across the whole
burst**. `s20/s21/s25` look free in-region — **but `drain_advance` writes `s20–s23`** and is invoked
immediately before the label and at every drain site. Making them burst-live recreates the `:1401` hazard
class *exactly* (an SGPR reused through a `.set` alias corrupting LIVE state, not just a counter).
`s58/s59` can carry 2 of the 3 at FM≤2 but not FM>2.
**Kimi described these and STOPPED rather than applying them. That was the right call — do not undo it
casually.** Unblocking needs a genuine free-register audit across the burst, not an assumption.

### 2.2 THE STRUCTURAL ONE — `global_load_tr_b128`
Would **halve B-loads 124 → 62.** `occ_kernel_btr128.s` already proves the two-adjacent-frag semantics on
this hardware. **Blocked because it changes `KDBUF_LPT`'s wait watermark (`:1267`) and the `bcnt`
accounting.** This is a **design conversation, not a slice** — it touches the load/wait pipeline.

### 2.3 CONFIRMED DEAD ENDS — DO NOT SPEND TIME HERE
- **The accumulator / `JDEPTH`.** `ds_add` is **68 instructions = 3.3%** of the slice. Eliminating it
  entirely buys ~0.3 on the ratio. **We set out to attack this and the measurement says it is noise.**
  The ACC is *already register-resident* within a burst; the only gap to hipBLASLt is flush *frequency*.
- **Predication.** The burst is fully-unrolled straight-line `.rept` — **there is no branch in it to
  convert.** hipBLASLt's `v_cndmask` is remainder/tail handling inside a **rolled** loop; our
  `KSEG_STEPS` is compile-time so there *is* no tail. Our real branches (role loop, claim, coast) are
  control flow where predication would execute **both** sides — strictly worse for spin/coast, and it
  "would poke the river for nothing."
- **`s54/s55` for anything else.** They are now live. See §1.

### 2.4 THE BIG UNMEASURED BUCKET
The ~180 remaining `s_add_co*` plus 244 `v_mov`, 139 `s_mov`, 133 `s_wait_dscnt`, 79 `s_and` in the slice
are **per-claim coordination**: `lds_*` macro expansion (exec saves, lane-0 broadcast) and probes, much of
it compiled out at `FORENSICS=0`. **Nobody has profiled the coordination path for instruction economy** —
all of tonight's work was the compute burst. That is the next frontier, and it is where the "one adaptive
kernel" cost actually lives.

---

# 3. ⛔ RETRACTION — THE MEASUREMENT WINDOW WAS NEVER THE INNER LOOP

**`first-WMMA → last-WMMA` IS NOT A K-STEP LOOP ON THIS KERNEL.** It spans **60 labels**, including
`.Lflow_jwait`, `.Lflow_bankwr`, `.Lbaton_norm`, `.Lflow_da_ss_complete`, `.Lflow_cstore`,
`.Lflow_drain_adv` — ring compute plus the entire JDEPTH retire/drain/C-store path plus decode.

**PROOF:** the actual burst source contains **10 `v_mov` and 1 `s_wait_dscnt`**. The window reported
**244 and 133**.

**THEREFORE: "we are 3.7× worse than hipBLASLt on non-WMMA per WMMA" IS WITHDRAWN.** Their 2.19 is a
genuine 511-instruction inner loop (`MT128x128x32`, 160 WMMA). Our 8.13 → 6.62 is a metered slice of most
of the coordination machinery. **The two were never comparable.** The instruction cuts are still real —
they removed genuine per-k-step work — but the ratio they moved is not the quantity 2.19 measures.
**To compare like-for-like we need a window that is provably only the k-step loop. We do not have one.**

---

# 4. THE hipBLASLt STRUCTURAL DIFF — THIS PART STANDS

Extraction recipe (the `.co` is a **CCOB**, zstd-compressed offload bundle — `file` says "data"):
```bash
clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
  --input=/opt/rocm/lib/hipblaslt/library/TensileLibrary_B8B8_..._gfx1201.co --output=fp8.co
```
→ 38 MB ELF, **446 fp8 kernels for our exact arch**. (That library is `bf8_bf8` = e5m2; **we emit
`fp8_fp8` = e4m3.** Same WMMA shape and cost. `B8F8`/`F8B8` libraries exist for an exact-format match.)

| | hipBLASLt (446 kernels) | DSWS |
|---|---|---|
| VGPR | min 56 · **median 254** · max 256 | **48 peak-live** of 256 |
| LDS | min 1,638 · **median 6,400** · max 32,768 | **34,304** |
| WGs/CU | ~4 | **1** |
| spills | 0 | 0 |

**AN EXACT INVERSION OF OUR DESIGN.** They fill the register file and barely touch LDS; we leave 81% of
the register file idle and fill LDS. **One root cause, three symptoms:** accumulators in LDS →
`ds_add`+wait per flush → LDS is the occupancy limiter at 1 WG/CU → the register file sits unused.
This is the `_GSU1_` thesis, previously inferred from kernel *names*, **now measured from their binary**.

**AND THE DISPERSION IS THE PRODUCT ARGUMENT, NOT A FOOTNOTE.** Their non-WMMA:WMMA spans **2.19 → 88.0,
median 12.93** across 446 hand-tuned static kernels selected per shape. **Their median is WORSE than
ours.** We are not trying to beat them on every shape — we are trying to beat them on *most* while being
far more consistent, which a per-shape lookup table structurally cannot do. **That is what adaptivity and
dyn-VGPR are for. Do not let any optimisation trade adaptivity for peak on one shape.**

---

# 5. CONFIG OF RECORD, AND THE REAL-SHAPE PICTURE

**Best measured (single shape, `ml8_dense_ffn_down` M2048 N2560 K9216): 15.4 TF**
`WAVES=6 FM=2 FN=4 G=8 ACC_N=4 SEGK=256`, `ML8_POOL=64`, superM=256, GROUPS=2. Mean of 4 repeats, ±1.3%.

**Both axes have interior optima** (§45–46): WAVES 5/6/7/8/16 → 14.9/**15.2**/14.6/13.8/10.2;
superM 128/256/512 → 12.7/**15.2**/13.6. The optima are real but **shallow** — 5/6/7 differ by 3–5%,
which is near the run-to-run band. **The dominant effect is FEW waves vs MANY, not the precise value.**

**Spread is a run-level lottery, not a config property (§59).** Four repeats of an identical bin gave
3.5% / 3.7% / 3.2% / **35.8%** — and the wide one's *mean barely moved*. It is a handful of outlier reps,
not a second regime. **Do not read a wide spread as a bad config.**

### THE 30-SHAPE PICTURE — AND THE 5× THAT MATTERS MORE THAN 15.4
Full gate at the WAVES=6 config: **30 PASS / 0 FAIL / 3 UNSUPPORTED** (`best15_4.json/.txt`).
Correctness is solid everywhere. Throughput is not: ml8 dense M2048 5.1–14.6, but **mlmf MoE experts
(≈56% of GEMM time) 0.97** and **ml8 MoE M64 0.09–0.61**.

**`--pool-auto` then bought geomean 1.37×, 12 shapes >1.5×, ZERO regressions** (`poolauto.json/.txt`),
with the mlmf experts going **0.96 → 1.78**. Rule: `ML8_POOL = min(64, TOTAL_super/10)`.
**On the worst shapes, pool 64 → 8 alone was 5× (0.101 → 0.5).** The runtime there is ~100% per-dispatch
fixed cost that **scales with WORKGROUP COUNT, not work** — launching 64 CUs for an 8-tile problem means
most WGs ramp up, find nothing, and retire, and *that ramp is the runtime*.
**superM and pool INTERACT** (§66): superM alone is flat, pool alone 2.17×, both 5×. **The auto-pool sweep
captured maybe half of what is there — per-shape superM is a second unclaimed ~2×**, but superM is a
BUILD-TIME defsym, so it needs one bin per superM class and a rebuild between groups.

3 UNSUPPORTED (`mlmf_router_MLP`, `router_out`, `routerout_ML8PAD`): `n_kseg=1<2` at SEGK=256, ZLOCK needs
≥2 → rerun those with `--segk 128`.

---

# 6. WHAT IS MEASURED AND EXCLUDED — DO NOT RE-CHASE

By direct instrument on the real shape: **boundary coordination 1.31% of wave-time** · **memory bandwidth
(22% of peak, NOT pinned — 1.8× spread)** · **load-issue rate (3× spread)** · **poll-pass cost 7.9%** ·
cursor CAS · SSWIN window · C-store gate (both now exactly 0) · **the publisher itself** (539 vs 560
advances/chunk at WAVES=6 vs 16 — it keeps up).

**`HEAD` (wave live → first work) is 19.9% of wave-time**, 1.5 ms, paid fresh every dispatch — the largest
single identified cost. It is **not sleep-dominated** (`SLEEPN` 1/2/4/8 all flat) and **cannot be
amortised on a fixed shape** (the problem is already one chunk). **The lever is to SHRINK it, and nobody
knows what is in it.** `PHIST` is the WRONG instrument (it is a bail-door histogram, ~220% overhead, and
once ran a 2.46 s chunk against a 0.75 s cap).

**`coast-frac` is NOT a throughput proxy** (§56, §58): SEGK=64 coasts 27.6% at 6.6 TF; SEGK=256 coasts
52.1% at 15.2. It moved *opposite* to throughput across the whole sweep, and it falls monotonically with
`SLEEPN` while TF does not move at all — it is a **poll-count artifact**. `door1 = 100% of coast` has been
quoted all project as the problem statement. **It is not.**

---

# 7. TRAPS, IN THE ORDER THEY BIT ME

1. **SHAPES ARE INPUTS, NOT LEVERS** (KG `8e201972`). I ran M=8192/16384/32768 and logged 18.6 TF as a
   "NEW BEST". Retracted — those are cubes I invented. **A fixed per-dispatch cost makes TF a function of
   problem size, so shape-growth ALWAYS looks like progress.** THE TELL: *if the config is unchanged and
   only the input grew, no lever was pulled.*
2. **A probe whose emission rate scales with the population under test cannot measure a population-size
   effect.** `BNDTIME` stamps every losing pass; at WAVES=16 there are 10.9× more, so it appeared to show
   the publisher slowing 4.3×. `ADVPROBE` alone: **167.4 vs 172.6 — flat.** Check the instrument's own
   scaling first.
3. **`pgrep -f <pattern>` self-matches your own watcher.** I reported a finished sweep as "still running"
   for ~18 min and held the card idle. Use `ps` with a non-self pattern, and treat *"the output file is
   already written"* as stronger evidence than any process check.
4. **A zero from a broken instrument is not a measurement.** Five parse bugs in the hipBLASLt work, each
   caught only because "0 WMMA in 4.97M lines" is implausible. And a *win* can be a bug too: whole-body
   counting said we beat the vendor 6.5× — that was a window mismatch.
5. **Empty output ≠ verified.** An awk range with a bad end-bound returned nothing and would have read as
   "s54/s55 verified dead". Re-bound it before believing it.
6. Five models died to CONTROLS this week, never to fits: feed-only, coordination-only, traffic-only, and
   two rate models. **Design the falsifier, not the next data point.**

---

# 8. HANDOFF ROUTE — USE `pi_handoff`, NOT opencode

```
mcp__mad-lab-memory__pi_handoff(
  provider="fireworks", model="accounts/fireworks/models/kimi-k3", thinking="high",
  tools="read,glob,grep,list,edit", exclude_tools="bash,write,task,webfetch,websearch",
  cwd=<dvgpr_occ>, machine="mad-lab-main")
```
**GIVE IT EDIT ACCESS** so it implements directly — do not have it emit blocks for hand-application.
**Keep `bash` OFF** (git hazard on a shared tree; a previous agent ran `git add` over in-flight work) and
**keep all GPU work in the interactive session.**

Kimi's review tonight was worth it: it produced a macro-by-macro register table proving the liveness
argument, found the one real fragility (now fixed), correctly declined predication with a better reason
than mine, and caught the window retraction in §3. **Design every consult to be scorable** — it corrected
4 of my facts on 07-27 and the window claim tonight.

---

# 9. OPEN ITEMS

- **The bring-up (§0). Nothing else until the oracle is clean.**
- `TRACE=1` per-super-tile claimer timeline — would show what `HEAD` is doing, and the outlier reps.
- Per-shape **superM** auto-tuning (needs one bin per class; ~2× still on the table for small shapes).
- The 3 `--segk 128` shapes.
- 30-shape gate at the **auto-pool + cuts** config once the bring-up passes.
- Coordination-path instruction economy (§2.4) — completely unprofiled.
- Tasks #43, #45, #46 still `in_progress`.

# 10. TREE STATE AT LOGOFF
Modified: `occ_kernel_dsws_flow.s` (5 cuts + guard), `build_flow.sh` (FN knob, NFV gate),
`occ_dispatch.cpp` (DSWS2_FN, NFV gate, FM whitelist), `dsws_realshape_bench.py` (--fn, --allow-nonstd,
--pool-auto), `DSWS_TESTING_LOG.md` (4,500 lines), `DSWS_BRIEF_2026-07-28_AM.md` (superseded banner),
3 `.lds` sidecars. Untracked: `best15_4.*`, `poolauto.*`, 3 new `.lds`.
**NOTHING STAGED** — the tree is shared with a live weight-pager session.
Bin `58e965a46f3e162d` · `.text` 28,852 B · LDS 34,304. **Latch CLEAR. No claim held.**
