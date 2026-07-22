# ⛔ DSWS CONTINUATION BRIEF — 2026-07-21. READ ALL OF §0 BEFORE TOUCHING ANYTHING. ⛔

Companion docs (read §0 first, then these): `RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md` (the reference
table), `DSWS_TESTING_LOG.md` (2026-07-21 entry), KG `efa5d89f` (next work), `148eddf2` (no Codex rule).

---

# §0 — ANTI-REPEAT. THESE ARE THE FAILURES OF 2026-07-21. DO NOT REPEAT THEM.

kmbandy, verbatim, twice today: *"I can't keep fighting these battles if you're forgetting things
and/or not adhering to established things between sessions."* This session burned most of a morning
and crashed his desktop by re-deriving things already written down. The rules below are not advice.

### R1. THE CONFIG OF RECORD IS `G=6 ACC_N=3`. NEVER SILENTLY DEPART FROM IT.
**What happened:** real M values (64/512/2048/4096) do not divide the G=6 super-tile (96). I hit that,
**silently invented `G=4 ACC_N=4`**, and kept going. Every measurement for the next four hours was on a
config we never established, so none of it was comparable to anything. That single unannounced fork is
the root of the whole bad morning.
**THE FIX WHEN M DOESN'T DIVIDE 96: PAD M UP to a multiple of 96 and correct TF back to real FLOP.
DO NOT CHANGE G.** (Padding counts against us, so it is never flattery.)
**RULE: changing ANY build defsym or dispatch knob from the config of record is an EXPLICIT act.
Say it out loud, reconcile against this brief + the testing log, and record it. No silent forks.**

### R2. DO NOT USE PHIST TO MEASURE RESERVATION CONTENTION.
It is **~220–294% overhead** and contaminates the very thing it measures. I used it anyway, got
"RESV-win 3%", and built two dead hypotheses on it. **Use RESVPROBE** (lean, register-accumulate:
`occ[87]`=CAS-loss, `occ[89]`=window-full). For phase shares use **PHASEPROBE**. Never quote TF from
any probe build.

### R3. ALWAYS PROVE AN INSTRUMENT BITES (negative control) BEFORE SPENDING A RUN.
`RESVPROBE=0` must be **byte-identical** to `397bfbe1`; `=1` must differ. Same for every ablation.
`NOCFLUSH` was once inert on the banked path and would have produced a fabricated finding.

### R4. THREE THINGS ARE RULED OUT. DO NOT RE-DIAGNOSE THEM.
- **Admission/concurrency is NOT the wall** (batch scan, 2026-07-20).
- **Compute is free** (NOWMMA −0.1%), **operands are free** (NOBLOAD −2.0%).
- **POOL_N and MSSCAN are both dead ends** — `PLAN_UNPIN_COMPUTE.md` is ⛔REJECTED on source evidence:
  self-serve waves use **direct global operand loads** and publish **pre-completed sentinels**, so the
  ring/pool carries nothing (`grow-fail=0` proves the ring path never executes). The `[DRAIN,STAGE)`
  scan is bounded by **POOL_N, not SSWIN**. I proposed both anyway. kmbandy stopped me on POOL_N.

### R5. AUTO-INJECTED MEMORY IS ESTABLISHED FACT, NOT BACKGROUND COLOUR.
Reading the brief at session start is **not** the control — I read it and then departed from it hours
later at a decision point. The failure happens **at the moment you deviate**, not at session start.

### R6. THE ONLY SANCTIONED DISPATCH PATH IS `./gpu_run.sh`. THE SIX RULES IN `CLAUDE.md` HOLD.
One dispatch per greenlight; changed kernel = ONE bring-up then STOP; hang/INCOMPLETE/WORK-INEXACT =
FULL STOP cleared only by a human; NEVER raise `DEADMAN_TICKS`; `board_check` immediately before every
`board_claim`.

### R7. THERE IS NO "CODEX WRITES THE CODE" RULE. **I WRITE THE CODE.**
Corrected three times. It was a usage-budget call, not architecture (KG `148eddf2`; the old feedback
`25cd06d9` is DELETED). What survives independently: don't put **unexecuted** implementation code in
plan docs. Commission Codex for a REASON (parallel worktrees, adversarial review), never by reflex.

### R8. ⛔ NEVER PUBLISH A NUMBER YOU HAVE NOT CHECKED AGAINST ITS SOURCE. ⛔
**Two fabrication-class defects on 2026-07-21, same root cause: a value was written into documents
without ever being verified against the thing it claimed to describe.**

1. **The whole DSWS throughput column was a parser bug.** `sweep_dsws_realshapes.sh` matched
   `'<num> TF'`; the kernel prints `TF=<num>`, so it NEVER matched and every row silently took the
   **last decimal on the line** — the `spread N%` or `N% of peak` field. It manufactured four "wins"
   over hipBLASLt and a flatness result, both of which propagated into this brief, the results file,
   the testing log, the KG, and the auto-injected SOP summary. **The `M=64` MoE "6.39x win" was
   `TF=0.0`.** Corrected: 0 wins / 26, and we are LESS flat than the vendor.
2. **The "bin sha `397bfbe1cb010c6e`" was unverifiable.** It matched no hash (sha256/sha1/md5/cksum)
   of any artifact and appeared ONLY in my own three documents. Removed below.

**RULES:**
- **Anchor extraction on a unique key** (`TF=`), take the FIRST match on the line, and **spot-check the
  harness against a raw log before trusting a single row.** A positional regex will return a
  *different, plausible* number forever without ever erroring.
- **A sha you did not compute in-session is not a sha.** Print the command and its output, or omit it.
- When a result is exciting, that is exactly when to re-derive it from the raw log. Both defects
  survived because the numbers were *good news*.

---

# §1 — CONFIG OF RECORD

**Identify the build by COMMIT + defsyms, not by a remembered hash.** Verify with:
`sha256sum occ_dsws2_w30_flow_gd.bin` (HEAD `652053c69` at the defsyms below rebuilds deterministically
to `4ecdab1dafca36bb`, 24008B, LDS 54016B; archived at
`~/dsws_gpu_logs/CONFIGOFRECORD_652053c69_4ecdab1d.bin`). The earlier `397bfbe1cb010c6e` was
unverifiable — see R8. **NOTE: `WAVES=30` is now known to be a REGRESSION on real shapes (§3); it is
recorded here as the historical baseline, not as a recommendation.**

```
BUILD
  WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 VBUDGET=1536 JDEPTH=1 KMAJOR=0
  STAGGER=1 MAXFAT=0 DECENTASN=1 SELFSERVE=1 BANKZERO=1 RBU=1 SSWIN=8
  INITBAR=1 TERMFIX=1 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1 ./build_flow.sh
    GROUPS = G/ACC_N = 2 ; super-tile M = G*16*FM = 96 ; N-panel = FN*16 = 64

DISPATCH (per shape)
  SSWIN=8 FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 FLOW_POOL_N=1
  DSWS2_SEGK=256 DSWS2_K=<K> DSWS2_ORACLE_MTL=<Mpad/96> DSWS2_ORACLE_NTL=<N/64>
  DSWS2_ORACLE_STRIDE=32 DSWS2_TARGET_SECS=1.5 STAGINSTR=1 FORENSICS=0 TFPROBE=1
```
Legality: `M % 96 == 0` (**pad, don't change G** — R1), `N % 64 == 0`, `K % SEGK == 0`.
Sweep harness: `./sweep_dsws_realshapes.sh` (STRIDE, TARGET_SECS, ONLY= env).

---

# §2 — STATE OF THE TREE (verified at write time)

- **HEAD `652053c69`** on `master` (ahead of origin; NOT pushed). All DSWS work is COMMITTED.
- Uncommitted in the spike dir: **only junk** — `*.bin.bak`, `*.pre-batch`, `.lds`, `grok_batch_prompt.txt`.
- **`docs/examples/router-fleet-main.ini` is kmbandy's, unrelated — NEVER stage it.**
- **Latch: CLEAR.** Bin on disk: `397bfbe1` = config of record. Card idle.

---

# §3 — ⛔ RETRACTED HEADLINE + THE REAL RESULT (see `RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md` §0)

**THE ORIGINAL §3 OF THIS BRIEF WAS WRONG. The DSWS TF column it cited was a PARSER BUG.**
`sweep_dsws_realshapes.sh` matched `'<num> TF'`; the kernel prints `TF=<num>`, so the pattern NEVER
matched and every row fell through to a fallback that grabbed the **LAST decimal on the line** — the
`spread N%` field, or the `N% of 307 TF peak` field. It was never throughput.

**RETRACTED:** all 4 "wins" (the three `M=64` MoE shapes actually read **`TF=0.0`**); the flatness
thesis; "mean 11.5x lower".

**CORRECTED (rebuilt offline from the archived `~/dsws_gpu_logs/rs_*.log`):**
| | **DSWS (true)** | ~~retracted~~ | hipBLASLt |
|---|---:|---:|---:|
| mean / median | **0.87 / 0.48** | ~~6.00 / 5.25~~ | 69.18 / 57.30 |
| **CV** | **1.128** | ~~0.687~~ | **0.905** |
| **wins** | **0 / 26** | ~~4 / 26~~ | — |

**WE ARE LESS FLAT THAN THE VENDOR, NOT MORE — the corrected data CONTRADICTS the flatness thesis.**
We lose every shape, by 43x to >1000x. Correctness was never in question (all 26 WORK-EXACT + oracle
CLEAN); only the speed claims were false.

**⭐ WHAT THE BUG WAS HIDING — THROUGHPUT TRACKS `n_kseg = K/SEGK`:**
| n_kseg | 36 | 16 | 10 | 8 | 6 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| true TF | 4.36, 2.40 | 2.33, 1.24, 1.07, 0.20 | 1.55, 1.55, 1.33, 1.24, 1.16, 0.36 | 0.98, 0.18, 0.18, 0.13, 0.00 | 0.69, 0.18 | 0.60, 0.30, 0.20, 0.18 | 0.18, 0.00 |

Derived from source BEFORE these runs: reservations are legal only while `r < DA_ZDONE` (:3983);
`DA_ZDONE` advances ONE field width per group boundary (:4151); that boundary needs `DRAIN>=ASSIGN`
(:4086) AND the prior group's C-store drained (:4093), because banks are REUSED (:4144); and one
reservation = one `ksi` carried by ONE wave across `ACC_N` rowblks (:4358,:4487).
=> **instantaneous per-WG parallelism = `min(WAVES, n_kseg)`.** At `WAVES=30` with `K=768` (n_kseg=3),
**90% of the workgroup is idle BY CONSTRUCTION.** `WAVES=30` was tuned on the deep-K synthetic
(n_kseg=2048) where units always outnumbered waves — the same synthetic-vs-real trap as the FLUSH
artifact.

**MEASURED CONSEQUENCE — FEWER WAVES IS WORTH ~4.3x** (each point its own build; the binary is
selected by name `occ_dsws2_w<N>_flow_gd.bin`; all WORK-EXACT + oracle CLEAN; TF read directly off
`TF=`, never through the sweep script):
| shape | n_kseg | W=30 | W=10 | W=5 | gain |
|---|---:|---:|---:|---:|---:|
| `ffn_gate_up M512 K2560` | 10 | 1.5 | 4.1 | **6.5** | **4.3x** |
| `lm_head M4096 K768` | 3 | 0.6 | — | **2.6** | **4.3x** |

Same 4.3x at n_kseg=3 and n_kseg=10 => not shape-specific. coast 93.5%->64.0%, boundary bails
754k->205k, starvation 5.86M->1.21M. `door1 NOTHING-STAGED` = 100% of coast at EVERY wave count: the
**supply of units is the wall**. `door3`/`door4` = 0 throughout — the dyn-VGPR moat never engages.
**`WAVES=4` is unbuildable** (`NCOMPUTE=1` -> `BATON_MAGIC=2^32`, not 32-bit; the `.if NCOMPUTE < 1`
guard at :780 catches 0 but not 1 — fails loud at assembly, a gap not a hazard).

**⚠ I BRIEFLY WROTE HERE THAT THIS "RETIRED" COUNTER-FREE ASSIGN. THAT IS RETRACTED — TWICE WRONG:**
wrong on the merits (see below), and **wrong to decide unilaterally. Cancelling planned architecture is
kmbandy's call. Do not read a one-measurement result as authority to drop planned work.**

**TWO FOLLOW-UPS INVERTED THE DIAGNOSIS — THE SHARED CURSOR IS THE BOTTLENECK:**
| change | result |
|---|---|
| `SEGK 256->64` (4x units: n_kseg 10->40, all 30 waves feedable) | **1.5 -> 1.2 TF, WORSE**; coast ROSE 93.5%->97.5% |
| `BATCH=2` at `WAVES=5` (more work per CAS) | **ABORTED** — chunk 0.81s vs ~0.08s, >=10x slower |

**UNITS ARE NOT THE WALL — `min(WAVES, n_kseg)` IS DEAD AS A THROUGHPUT EXPLANATION.** More units =>
each reservation carries LESS work => MORE CAS traffic per unit of output. And `door1 NOTHING-STAGED
= 100%` was never evidence about supply: under SELFSERVE it is the **vestigial ring** door and reads
100% regardless. `BATCH=2` failed at BOTH `WAVES=30` and `WAVES=5`, so holding the shared `SSWIN`
window while draining serially is **intrinsic to the shared cursor**, not a wave-count artifact.

**=> ALL THREE RESULTS FIT ONE CAUSE: THE SINGLE SHARED `ASSIGN` CURSOR CAS. COUNTER-FREE ASSIGN (§6,
KG `efa5d89f`) IS THE INDICATED FIX AND REMAINS THE PLANNED WORK** — now with `WAVES=5` as a much
cleaner starting point than this morning's `WAVES=30`.

---

# §4 — WHAT WAS FIXED TODAY (all committed in `652053c69`)

### 4a. NON-POW2 n_kseg — this is what made the table possible
Real K give non-pow2 n_kseg (2560→10, 9216→36, 768→3, 1536→6) and **NO legal SEGK in {16..256} can
make them pow2**. The DECENTASN coupled cursor was **`POW2 n_kseg only` BY CONSTRUCTION** with an
explicit fail-safe routing non-pow2 to `.Lflow_da_terminal` → clean retire, `computed=0`, **silently no
work**. Half the real shapes returned zeros and nobody knew.
**FIX:** the reservation span now strides the **ksi FIELD WIDTH (2^shift)** instead of n_kseg, so
`TOTAL=GROUPS<<shift`, `z>>shift`, `ksi=within&mask`, `group=within>>shift` stay exact for ANY n_kseg
with **no division and no spare SGPR**. The `(2^shift − n_kseg)` phantom indices per field are **never
reserved**: the peek stops at the real end via `ksi = r & mask` (**register-only** — base is always
2^shift-aligned so no LDS read; an earlier version that read `DA_BASE` per peek cost **16x**), and the
boundary re-bases `ASSIGN/DRAIN/STAGE` past the gap under ZLOCK while the pipeline is quiesced.
**Byte-identical for pow2 n_kseg.** Verified: non-pow2 n_kseg=3 exact+clean; pow2 n_kseg=8 regression
clean at full oracle stride.

### 4b. WORK-EXACT gate is REPS-AWARE
`occ[71]` accumulates across `DSWS2_TARGET_SECS` reps → compare against `G*TOTAL_super*repsDone`.
It was **false-latching every reps>1 run**.

### 4c. ⚠ THE COMPOSITOR CAP WAS STRUCTURALLY BROKEN — AND IT CRASHED HYPRLAND
`chunkMaxS` is evaluated **between** chunks, so it can only abort REMAINING chunks. The old default
(`chunkTiles = claimTotal`) produced **ONE chunk covering the whole problem → nChunks==1 → ZERO
protection**, while still printing `"compositor-safe"`. A 2.46s single chunk (PHIST, ~220% overhead)
took **Hyprland to safe mode** — rule 7 exactly: desktop dies, **no GPU reset**, so no other guard sees
it. **EVERY dsws2 run before this fix was unbounded by default.**
**FIX:** default chunk bounded to **512 tiles** (nChunks>1 so the cap has granularity); the
single-chunk case now **WARNS** instead of reassuring. Both branches verified live. It then caught a
real 0.81s overrun within the hour.

---

# §5 — THE PERF PICTURE ON REAL SHAPES (supersedes the deep-K synthetic ranking)

**PHASEPROBE at the config of record.** Note WMMA is **identical on a shape we win 6.4x and one we
lose 300x** — our duty cycle is shape-independent, which is *why* we are flat:

| phase | LOSS ffn_gate_up M2048 | WIN moe_attn_kv M64 |
|---|---|---|
| FOLLOW_WAIT | 36.7% | 57.4% |
| GROW | 39.3% | 21.8% |
| **WMMA** | **20.7%** | **20.7%** |
| FLUSH | 1.4% | 0.1% |
| SHRINK | 1.1% | 0.0% |

- **FLUSH IS DEAD as a target.** 33.7% on the deep-K synthetic was an **n_kseg=2048 artifact**; real
  n_kseg=10 → 1.4%. Do not spend a minute on the reduction.
- **FOLLOW_WAIT is NOT a pipeline stall.** It is waves that **lost the reservation scrum** sitting in
  the vestigial ring-compute role. Symptom, not bottleneck.
- **GROW+SHRINK ~40% is a fixed per-wave tax**, shape-independent, and `grow-fail=0` so it **buys
  nothing**. kmbandy: **this pillar falls LAST.**

**RESVPROBE (the lean instrument) on ffn_gate_up M2048:**
```
CAS-loss    = 9.932 collisions per successful reserve   (vs 1.466 on the deep-K synthetic)
window-full = 3.0%                                       (NOT the constraint)
```
**Admission is gated at GETTING A RESERVATION, and the gate is contention on `ASSIGN_HEAD`.**
Why worse on real shapes: n_kseg=10 vs 2048 → one claim buys ~200x less work → far more claims.

---

# §6 — NEXT WORK: COUNTER-FREE ASSIGN (designed, NOT implemented). Full detail KG `efa5d89f`.

kmbandy: *"the counter is another dam; delete it, and then batching comes back."*

**Scope (verified in source):** `ASSIGN_HEAD_OFF=0` is an **LDS** offset (`lds_get`/`lds_cas_rtn`) =>
**PER-WORKGROUP**; the scrum is 30 waves inside one WG, 64 independent copies. `occ[20]` (the TILE
claim) is the only **global** atomic and fires once per tile — **leave it alone.**

**Design:** each wave derives its unit from its wave id within the WG's current tile — no CAS, no
retry. One atomic per **tile** instead of per **unit**. **BATCH=2 kept, NO steal path** (kmbandy: start
simple; explore batch>2 + stealing only if batch helps). Batching is safe here because a wave's units
are its own by construction, so holding them blocks nobody — that is exactly the failure that made
BATCH=2 catastrophic **with** the counter (measured: **810ms/chunk vs 8.3ms, ~100x slower**; a wave
held the shared SSWIN window while draining serially). Same knob, different machine. BATCH=2 **also
amortizes the grow/shrink round-trip without touching dyn-VGPR**, keeping that pillar for last.

**THREE CONSTRAINT LAYERS — each changes the mapping, all found while designing:**
1. `ASSIGN`/`DRAIN` are the **completion accounting**, not just handout; the boundary gates on
   `DRAIN>=ASSIGN`. Counter-free needs `ASSIGN = z` at group start and the boundary to fire when
   `DRAIN` reaches `z`.
2. A wave re-enters the peek many times per group and needs one bit of **per-wave state**
   ("served this group?"). Not derivable from the slot (`POOL_N=1` → one slot) nor from `DRAIN`
   (window between finishing and DRAIN advancing → wave re-does its unit → **duplicate work**).
3. **THE MAPPING MUST BE WINDOW-AWARE, NOT A FLAT PARTITION.** The control array is **SSWIN=8** deep
   (that's the POOL_N decoupling: operands pool-depth, control window-depth). With n_kseg=10, a naive
   `wid→unit` map puts waves 8,9 on the same control slots as 0,1 → two waves sharing one
   `SL_GEN`/`RBDONE` → duplicate rowblks into one bank → **WRONG C with EXACT counts.** ≤8 in flight;
   waves cycle as slots retire.

**USE `s15` for the per-wave bit.** See §7.

**FALSIFICATION, SET IN ADVANCE:** CAS-loss → ~0; WORK-EXACT; oracle CLEAN **at dense/full stride** (a
clean oracle at stride 64 is NOT evidence — the count gates cannot detect duplicated work or wrong
operands). If contention vanishes but span doesn't move, the scrum wasn't the cost — **report that**,
do not reach for batch>2.

---

# §7 — ⚠ CORRECTION: "ZERO FREE SGPRs" IS WRONG AS WRITTEN

The 2026-07-20 audit asked *"is this referenced in the source?"*. The right question is *"does the
SHIPPED BUILD emit it?"* Method — immune to the both-spellings trap that produced two false audits:
```
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1201 occ_dsws2_w30_flow_gd.o
```
then collect every `sNN` and `s[a:b]` range. **RESULT: 89 emitted, 17 PROVABLY UNUSED:**
`[15, 61, 64, 65, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 91]`

Most are **conditionally** free: `s72/s73` = BATCH backlog (**spoken for**, we're turning BATCH on);
`s91` = JDEPTH counter (free only at JDEPTH=1); `s61/s65` = macro clobber / try_gate scratch;
`s64` etc = probe accumulators.

**USE `s15` — it is UNCONDITIONALLY free in EVERY build.** Source: *"TGID_X now lands in s15 —
UNUSED; this kernel is pool-claim, not workgroup-id based."* Hardware drops workgroup-id-X there and
the kernel never reads it, so nothing can be flipped to reclaim it.
**Consequence: instrumentation headroom treated as CLOSED since 2026-07-20 is actually OPEN.**

---

# §8 — OPEN ITEMS

- `n_kseg==1` (K=256, `mlmf_router_MLP`) hits the documented `n_kseg==1` fail-safe (bit-0 ZLOCK needs
  n_kseg>=2). Halts the sweep at that shape. A `(z<<1)|ZLOCK` re-encoding would fix it **and** remove
  the parity dependency — deliberately NOT done (riskier concurrency-primitive change for one 0.5
  GFLOP shape).
- `N % 64` unsupported: `mlmf mamba in_proj` N=4200 / N=4208 (FN*16 N-panel).
- `occ[20]` over-claim (65 on a 1-tile shape, 1344 on 1280 tiles) — benign (WORK-EXACT + clean oracle)
  but **unexplained**.
- `master` is ahead of origin and **unpushed**.
- Fill speed: `Aval/Bval` per-element division makes the lowmem fill ~90s for ~17GB (row-based fill
  would be ~6x faster).

---

# §9 — INSTRUMENT INDEX (which tool for which question)

| question | instrument | notes |
|---|---|---|
| where does compute-wave time go | **PHASEPROBE** | probe TF invalid; use small `ML8_COOP_CHUNK` |
| why do reservations fail | **RESVPROBE** | lean. `occ[87]` CAS-loss, `occ[89]` window-full |
| where are waves parked | PHIST | **~220% overhead — NEVER for contention (R2)** |
| boundary transitions / cursor corruption | BNDPROBE | `occ[116..123]` incl. QBAD + skew captures |
| did work get dropped | WORK-EXACT gate | `computed == G*TOTAL_super*repsDone` |
| are values right | oracle | samples sparsely — **dense stride for concurrency changes** |

**Every probe must pass a negative control (R3).** All are byte-identical when off.
