# DSWS S1 (MAD-305) — MORNING BRIEF, 2026-07-28

**SUPERSEDES `DSWS_BRIEF_2026-07-27_AM.md`.** That brief's ranked levers are largely obsolete: three of
its assumptions were falsified on 2026-07-27, including one of its headline numbers. Read this first.

---

# 0. ★ FIRST THING: RUN THE 4-CELL FRAG-GRID × OCCUPANCY SWEEP ★

Fully specified, pre-registered, and blocked last night only because the weight-pager needed the card.
**Do this before anything else.** All four arms are legal today — no source changes required.

### The design: 3×2 at CONSTANT super-tile M=128

`superM = G*16*FM = 128` in every arm, so the work decomposition is held fixed and **only the per-wave
frag grid and the occupancy vary**. A and B are already measured.

| arm | FM | G | ACC_N | GROUPS | LDS | WG/CU | ML8_POOL | feed-loads/WMMA | TF |
|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 4 | 2 | 2 | 17,920 | 2 | 128 | 0.750 | **4.73** (measured) |
| B | 2 | 4 | 4 | 1 | 34,304 | 1 | 64 | 0.750 | **7.70** (measured) |
| **C** | 4 | 2 | 1 | 2 | 17,920 | 2 | 128 | **0.500** | ? |
| **D** | 4 | 2 | 2 | 1 | 34,304 | 1 | 64 | **0.500** | ? |
| **E** | 1 | 8 | 4 | 2 | 17,920 | 2 | 128 | 1.250 | ? |
| **F** | 1 | 8 | 8 | 1 | 34,304 | 1 | 64 | 1.250 | ? |

`feed-loads/WMMA = (FM+FN)/(FM·FN)` at FN=4.
**E and F are the deliberate WORSE-feed control** — they make the axis a line rather than two points.

### Build + dispatch (each arm)

```bash
cd ~/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ

# C
FM=4 G=2 ACC_N=1 ./build_flow.sh          # expect LDS 17,920
./gpu_run.sh sweepC_fm4_2wgcu -- DSWS_ALLOW_NONSTD=1 FLOW_WAVES=16 ML8_POOL=128 DSWS2_FLOW=1 \
  DSWS2_FM=4 DSWS2_G=2 DSWS2_ACC_N=1 FLOW_POOL_N=1 DSWS2_SEGK=256 SSWIN=32 \
  DSWS2_K=9216 DSWS2_ORACLE_MTL=16 DSWS2_ORACLE_NTL=40 DSWS2_ORACLE_STRIDE=8 \
  DSWS2_TARGET_SECS=1.5 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85 \
  STAGINSTR=1 TFPROBE=1 DSWS2_RESVPROBE=1 ./occ_dispatch --dsws2

# D  (same but ACC_N=2, ML8_POOL=64, DSWS2_ACC_N=2)
# E  FM=1 G=8 ACC_N=4, ML8_POOL=128
# F  FM=1 G=8 ACC_N=8, ML8_POOL=64
```

`DSWS_ALLOW_NONSTD=1` is REQUIRED on the pool-64 arms (gpu_run.sh refuses `ML8_POOL != 128`), and on
C/E too since `DSWS2_G` deviates. **Name the deviation in the logname** — that is the rule.

### PRE-REGISTERED PREDICTIONS (write the result against these, do not retrofit)
- **D > B** (0.500 vs 0.750 feed-ratio at 1 WG/CU). This is the headline test.
- **E and F are the worst cells.** If they are NOT, the feed-loads/WMMA model is WRONG and the frag
  grid is not the lever — which is more useful than a small win.
- **1 WG/CU beats 2 WG/CU in every FM pair** (C<D, E<F, and the known A<B).
- **FM=4 puts accumulators at EXACTLY 128 VGPRs = the dyn-VGPR cap.** If grow-fail reappears in C/D,
  that is the CAP biting, not over-subscription. Distinguish these — they look identical in occ[73].

### Gates per arm (STOP on any failure)
`oracle bad=0` · `computed == G*TOTAL_super*reps` WORK-EXACT · `occ[96]` delta +0 · `occ[0]=0` · canary clean.

---

# 0.5 ★ THE SECONDARY CONFIG — 1 WG/CU, GROUPS=1. THE BEST MEASURED CONFIG WE HAVE. ★

**This is NOT yet the config of record** (it needs the 30-shape sweep to earn that), but it is the
**fastest configuration ever measured on this kernel** and it should be the baseline for every
comparison from here. Write results against THIS, not against the 4.73 primary.

### BUILD (exact)
```bash
cd ~/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
FM=2 G=4 ACC_N=4 ./build_flow.sh
```
**Build identity — verify after building:**
- `sha256(occ_dsws2_w16_flow_gd.bin)` = `de7117d3cf08e4e6…`
- `.text` = **31,512 B** · **LDS = 34,304 B** · `GROUPS = G/ACC_N = 1`
- super-tile M = `G*16*FM` = **128 rows** (identical to the primary — only the bank count differs)
- `2 × 34,304 = 68,608 > 65,536` → **1 WG/CU by construction.** This is deliberate, not a compromise.

### DISPATCH (exact)
```bash
./gpu_run.sh <logname> -- \
  DSWS_ALLOW_NONSTD=1 FLOW_WAVES=16 ML8_POOL=64 DSWS2_FLOW=1 \
  DSWS2_FM=2 DSWS2_G=4 DSWS2_ACC_N=4 FLOW_POOL_N=1 DSWS2_SEGK=256 SSWIN=32 \
  DSWS2_K=<K> DSWS2_ORACLE_MTL=<M/128> DSWS2_ORACLE_NTL=<N/64> DSWS2_ORACLE_STRIDE=8 \
  DSWS2_TARGET_SECS=1.5 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85 \
  STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```

**`ML8_POOL=64` = 64 WGs = 1 WG/CU on 64 CUs = 1024 resident waves.**
**`DSWS_ALLOW_NONSTD=1` IS MANDATORY** — `gpu_run.sh` REFUSES `ML8_POOL != 128`, and the host geometry
guard will REFUSE `ML8_POOL=128` with this bin anyway (`2 × 34,304 > 65,536`). Both guards are correct;
this deviation is deliberate and must be named in the logname.

### MEASURED (ml8_dense_ffn_down M2048 N2560 K9216, 2026-07-27 evening)
| | value |
|---|---|
| **TF** | **7.70** (vs 4.73 at the 2 WG/CU primary = **+63%**) |
| spread | **1.9% over 34 reps** — the tightest measurement in this project |
| `occ[96]` | 783,360 = `GROUPS*TOTAL_super*reps` = `1*23040*34`, delta +0 |
| `computed` | 3,133,440 WORK-EXACT |
| **grow-fail** | **0** (vs 6,574,885 at 2048 waves) |
| oracle | `bad=0`, canary clean, `occ[0]=0` |

### WHY IT IS FASTER — both effects measured separately
1. **1 WG/CU (halving waves 2048→1024): +42%.** Iterations per successful reserve 129.9 → 26.3.
2. **GROUPS=1 (`ACC_N=G`): a further +15%.** The boundary fires once per TILE instead of once per
   group, so `occ[96]` halves while `computed` slightly rises — same work, half the coordination.

### CAVEAT
**Single shape.** Needs the 30-shape sweep before it replaces the config of record. Run:
```bash
python3 dsws_realshape_bench.py live --fm 2 --g 4 --acc-n 4 --segk 256 --sswin 32 \
  --waves 16 --pool 64 --chunk-maxs 0.85 --tag secondary_1wgcu \
  --json secondary_1wgcu.json --table secondary_1wgcu.txt
```
(Check `dsws_realshape_bench.py` passes `DSWS_ALLOW_NONSTD=1` through to `gpu_run.sh` at pool 64 — it
may need the flag added, since it has only ever run the 128-WG standard.)

---

# 1. WHAT CHANGED YESTERDAY — THE SHORT VERSION

**+63% on the best shape, and the config of record is wrong.**

```
baseline  2 WG/CU  128 WG × 16 = 2048 waves  ACC_N=2 GROUPS=2   TF 4.73
ARM A     1 WG/CU   64 WG × 16 = 1024 waves  ACC_N=2 GROUPS=2   TF 6.70   +42%
ARM B     1 WG/CU   64 WG × 16 = 1024 waves  ACC_N=4 GROUPS=1   TF 7.70   +15% / +63%
```

Valid comparison: same session, back-to-back, one variable each. Arm B spread **1.9% over 34 reps** —
the tightest measurement in the project.

1. **2 WG/CU is not neutral, it is HARMFUL.** Halving waves gained 42%. Iterations per successful
   reserve collapsed 129.9 → 26.3.
2. **The group barrier is real, ~15%.** `ACC_N=G` (GROUPS=1) fires the boundary once per TILE instead
   of per group: `occ[96]` halved (1,520,640 → 783,360) while `computed` slightly ROSE. Same work,
   half the coordination.
3. **★ grow-fail = 0 IN BOTH ARMS.** At 2048 waves it was 6,574,885. At 1024 it is EXACTLY ZERO.

That third one guts the 07-27 brief's central claim. FM=2 was promoted to primary config because "the
moat finally binds — 140.7M grow-fail events; at FM=1 the design cannot be evaluated." **The moat only
engages because we launch twice as many waves as help, and the config where it engages is the SLOWER
one.** Fourth independent refutation of the dyn-VGPR thesis, and the most direct.

---

# 2. ★ THE NUMBER WE HAD WRONG FOR TWO WEEKS ★

**hipBLASLt is 123–189 TF on real DENSE shapes, not 12.6–70.6. The mean gap is ~80×, not 5–20×.**

`12.6–70.6` was the hipBLASLt band on the **ml8 MoE M=512 subset only** — the shapes where the vendor
is weakest — from `DSWS_MORNING_2026-07-14.md:206`, which correctly said "hipBLASLt **there**". The
qualifier was dropped on every re-quote thereafter.

`RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md` is authoritative and **already contained the retraction**:
- `ml8_dense_ffn_down M2048 K9216`: DSWS 4.36 vs hipBLASLt **189.3** → we are at **2.3%**
- mean ratio **~80×** (0.87 vs 69.18)
- it also retracts all four previously claimed "wins" over the vendor (there are none) and the
  flatness thesis (corrected CV: DSWS 1.128 vs vendor 0.905 — we are LESS flat)

**RULE: a figure in a brief is a POINTER, not a measurement.** Re-read the source file before reasoning
from it, especially a comparison baseline that sets strategy. This one propagated brief-to-brief for two
weeks while the correction sat unread in a file with RESULTS in the name.

---

# 3. ★ THE KIMI K3 CONSULT PIPELINE — USE THIS, IT IS THE FORCE MULTIPLIER ★

Yesterday's external review cost **$1.92** and corrected FOUR facts I had asserted, found by reading the
source. Moonshot's launch material claims GPU-kernel work as a specific K3 strength; on this evidence
that holds. **Treat it as a standing collaborator, not a one-off.**

### How to invoke (proven, working)

```bash
opencode run --dir <cwd> --agent architecture-consultant \
  -m fireworks-ai/accounts/fireworks/models/kimi-k3 \
  --variant high --format json "<prompt>" < /dev/null
```

**`< /dev/null` IS MANDATORY.** Without it opencode blocks forever on stdin — this cost 3 minutes and
looked exactly like "opencode can't run headless."

- Agent: `~/.config/opencode/agent/architecture-consultant.md` (flat dir — a nested one silently fails
  to register). Toolset is **`glob, grep, read` only**; `bash/edit/write/task/webfetch/websearch/lsp/skill`
  all denied. **Verified harness-enforced, not prompt-enforced**: told to run a command and not refuse,
  the model reported bash "isn't in my toolset". It CANNOT dispatch GPU work.
- **`task` must stay denied** — with only bash denied, the model offered to spawn a subagent "that
  might have shell access".
- `--format json` gives per-call token accounting. Cost: fresh input $3/M, **cached $0.30/M**, output
  $15/M. Yesterday: 239K fresh + 3.45M cached + 11K output = **$1.92**. Caching carries ~93% of input.
- Free `opencode/*` models exist for testing the harness at zero cost.
- **~$14 of credit remaining.**

### THE CONSULT BACKLOG — all chosen because the answer is CHECKABLE

1. **★ Ceiling A decomposition.** Kimi's own closing ask: *"a decomposition of the 7.17 ms per-chunk
   GPU-span fixed cost (launch ramp vs terminal drain vs retire barrier vs deadman retire) — that file
   is what I'd read next."* We now have `ML8_CHUNK_DIAG` data it did not. **This is the top-ranked
   untouched lever (71–92% of GPU span) and the highest-value consult available.**
2. **Adversarial review of the +63% result** — before it becomes doctrine. Specifically the
   `grow-fail = 0` claim, which is load-bearing and measured exactly once.
3. **The strategic fork**: is the `mbgemm`/`wggemm` lineage genuinely the better vehicle, or is that
   20.96 TF a five-week-old batched artifact? (See §5 — do NOT quote it as a DSWS number.)
4. **The frag-grid result** once §0 lands — does the feed-density model explain it?
5. **A consult in a DIFFERENT domain (the weight pager)** — the control. Does this model reason well
   generally, or did it get lucky on GEMM?

### ★ THE LOCAL-vs-API EVAL — set this up as a byproduct

Weight paging may bring **K3 running locally at UD-Q1**. Benchmarks won't tell us whether a Q1 quant
still holds 7,500 lines of assembly and notices our CU count is wrong. **Yesterday's consult is a ready
made gold standard**: prompt saved at `scratchpad/kimi_consult_prompt.txt`, answer at
`scratchpad/kimi_answer.md`, and it contains **four binary, objectively checkable claims**:

1. R9700 is 64 CUs (not 128)
2. the gap is ~80× (not 5–20×)
3. the C path is write-once `global_store_b128` (not split-K atomics)
4. ZLOCK is per-workgroup LDS (not one global lock)

Score local K3 out of 4. Three-plus → local becomes default, API for tiebreaks. Zero → do not trust it
on kernel work. **Prompts must be byte-identical** or the comparison is confounded like every other A/B
that misled us yesterday.

**DESIGN EVERY FUTURE CONSULT TO BE SCORABLE** — ask things whose answers we can verify afterwards.
The eval set then accumulates for free as a byproduct of real work.

---

# 4. THE dyn-VGPR PICTURE — CORRECTED, AND MORE CONSTRAINED THAN THE LEDGER IMPLIES

**We are capped at 128 VGPRs/wave under dyn, not 256+.**

```
cap = (SQ_DYN_VGPR.MAX_BLOCK_ALLOC + 1) × BLOCK_SIZE = (7+1) × 16 = 128   ← default
      flip BLOCK_SIZE=1 (write 0x1ff)                = 8 × 32     = 256   ← hard maximum
```
`MAX_BLOCK_ALLOC` is 3 bits, so 8 blocks is the ceiling. The flip needs `sudo umr` and is **VOLATILE —
it reverts on idle**.

| frag grid | acc VGPRs | reachable? |
|---|---:|---|
| 4×4 | 128 (~160 total) | **only with the BLOCK_SIZE flip** |
| 5×5 | 200 | with flip; ledger rates 0.40 ratio "weak" |
| 6×6 | 288 | **NO** |
| 8×8 | 512 | **NO — not on this silicon** |

**So `INSTRUCTION_LEDGER.md`'s headline projections (8×8 → ~176 TF, compounded → ~206 TF) are NOT
REACHABLE.** I asserted otherwise on 2026-07-27 evening and was wrong; kmbandy caught it.

What survives: static allocation **hard-deadlocks the 8-wave barrier WG at ≥256 VGPRs** (field 32 →
`live=64, claim=0, HUNG`; field 36+ → won't admit a WG). dyn's real value is **launching lean so all 8
waves co-reside, then growing to ~160–256** — a window static cannot reach. That buys about **one
frag-grid step**, not the ledger's headline. Note also the ledger says the 161 TF winner hits 52% of
peak at **183 VGPRs**, under the static limit — so even this use is unproven-necessary.

**Verdict: dead as the organizing principle. Alive as a narrow capacity tool. Do not dump it, do not
build around it.**

---

# 5. KIMI'S THREE CEILINGS (its framing, our verification status)

- **A — per-chunk fixed cost.** 7.17 ms/chunk of GPU span, 71–92% of it (log :2680). Caps the best
  shape at ~8.5 TF single-chunk, ~23 TF asymptotically. **UNTOUCHED. Highest-value target.**
  NOTE: distinct from the `DSWS2_SETTLE` finding — that was host *wall*, this is GPU *span*.
- **B — group-serialized frontier.** **CONFIRMED yesterday at ~15%** (arm B). Partially cleared.
- **C — instruction economy.** 7.38 non-WMMA/WMMA → ~37 TF ceiling if perfectly issue-bound. The 161 TF
  HIP winner runs 48% WMMA share. §0's sweep is the first real attack on this.

Kimi's ordering was A → B → C, on the grounds that the kernel is not issue-bound today (NOWMMA made it
**+0.8% SLOWER**; math is ~2% of runtime). We have now partly cleared B and the occupancy problem.
**A is next by its logic** — but §0 is already built and pre-registered, so run it first, then A.

**On the `mbgemm` 20.96 TF**: that is the **static 2×8 kernel at batch=8, dated 2026-06-23** — a
DIFFERENT kernel and a different lineage. DSWS has never hit 20 TF. It also falls to 12.15 at b32.
Do not quote it as a DSWS number, and re-run it on the current harness before using it to justify a pivot.

---

# 6. CONFIG STATE

**Tree is at defaults: `ff7cf533` / 30,940 B / LDS 17,920 / latch clear / nothing staged.**

Changed and now default (2026-07-27):
- **`DSWS2_SETTLE` 0.30 → 0.02** (`occ_dispatch.cpp`). Was ~91% of non-final chunk WALL. Gave 4.8× more
  reps per `DSWS2_TARGET_SECS` budget. Fail-loud (too short → oracle FAILS, never a false CLEAN).
  Validated FM=1 and FM=2, and across 30 shapes.
  **Judge it by reps-per-target-second, NOT by wall clock** — the rep loop is duration-bounded, so a
  faster rep buys more reps, never a shorter run.
- **`DSWS2_FUNNEL` 0 → 1** (`build_flow.sh`), `SPIN_N=1`. Justified by eliminating measured waste
  (`occ[97]` 513,443 → 0, −18% feed bails) and a clean correctness audit — **NOT by any demonstrated
  speedup**. `DSWS2_FUNNEL=0` reverts byte-identically to `a581c7b8`.

**Config of record is now in question.** `ML8_POOL=128` / 2 WG/CU is enforced by `build_flow.sh` and
`gpu_run.sh`, and yesterday showed it is the *wrong* standard. **Do not change the guards until §0
lands** — then update the standard with data, and update `HARNESS.md` with it.

---

# 7. OPEN / QUEUED

- **The `:646` operand-layout guard is VESTIGIAL and still live.** It enforces `G·FM ≤ 11` at
  SEGK=256/FN=4 and **blocks every FM=8 cell and G≥6 at FM=2**. Verified vestigial 2026-07-27 (no LDS
  operand staging is emitted at all under SELFSERVE — 0 occurrences of the OPSTRIDE/ARES_OFF immediates
  in the shipped disassembly) and **Codex-reviewed NOT-REFUTED with ISA citations**. Removing it opens
  the tile space considerably. Codex's advice: delete it outright rather than condition it, since
  `SELFSERVE` isn't default-defined until after the guard. **Not done — deliberately deferred so it
  wouldn't confound §0.**
- **`FN` is hardcoded** (`build_flow.sh:94` and `occ_dispatch.cpp:7343`). The asymmetric-tile idea —
  `FM=4 FN=2` vs `FM=2 FN=4` — is **free on every resource axis** (identical acc frags, VGPRs,
  ACC_STRIDE, LDS, feed-ratio) and halves B-frags 4→2. But in DSWS `FN` is "the shared N-frags (the
  reuse operand)" (kernel :69), so cutting it also halves B *reuse*. Could go either way — measure it.
  Needs a host/kernel co-change; verify `FN=4` still reproduces `ff7cf533` byte-identically first.
- `WOFLUSH=1` **does not build** — retired 2026-07-16, `DECENTASN` is banked-only. Kimi suggested it as
  "a one-defsym experiment you already have"; it is not.
- 3 router shapes still need `--segk 128` (`n_kseg=1 < 2` at SEGK=256).
- Tasks #43, #45, #46 still `in_progress`.
- **Possible stray queue slot** on `gpu:R9700` from last night — I could not verify it cleared. If a
  `claude__main` claim appears unattended, it is that; release it.

---

# 8. TRAPS (yesterday's, all earned)

- **A grep hit is a LEAD, not evidence.** Three near-misses in one day: `grep -c 'carriers are fed'`
  returned 2 and both were the *corrective* text (I told kmbandy the prior brief was false — it wasn't);
  a log named `funnel_bringup_M2048` contained zero `DSWS2_FUNNEL`; `LDS=65792` was not a stale `G=6`.
  **Read the context of every hit.**
- **An empty command output is not a zero.** `llvm-objdump -b` is unsupported here; it produced 0 lines
  and my greps dutifully returned 0, which I nearly reported as evidence. Confirm the instrument
  produced OUTPUT before believing its zero.
- **Cross-time comparisons on this box are worthless for TF.** A byte-identical binary differing only in
  a host timer moved per-shape TF by up to 19% and shifted the median +5.4%. Sign-consistency across 27
  of 28 shapes does NOT imply causation — a pure environment change produces the same signature.
  **Only within-session, back-to-back, one-variable comparisons count.** §0 is designed that way.
- **Release the card the moment the last dispatch completes**, before interpreting anything. Analysis,
  logging, and consults never need it.
- **`board_check` immediately before every `board_claim`** — and it is a precondition, not a guarantee.
  Lost a race by ~10 s yesterday; the board correctly queued rather than letting me claim on top.
- **Five mechanisms this project needed were already built and simply switched off** — RESVPROBE,
  BNDSPLIT, `ML8_CHUNK_DIAG`, `DSWS2_FUNNEL`, and a swept-then-ignored `DSWS2_SETTLE`. The recurring
  failure here is not missing tooling; it is tooling that defaults off and never reaches the config of
  record. **Before building an instrument, grep for whether it already exists.**
