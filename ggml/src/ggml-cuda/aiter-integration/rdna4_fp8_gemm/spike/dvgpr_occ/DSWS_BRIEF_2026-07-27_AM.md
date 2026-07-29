> # ⛔ SUPERSEDED 2026-07-27 EVENING — READ `DSWS_BRIEF_2026-07-28_AM.md` INSTEAD
> Three of this brief's load-bearing claims were falsified the same day:
> 1. **"FM=2 G=4 ACC_N=2 is primary because the dyn-VGPR moat finally binds."** grow-fail is EXACTLY 0
>    at 1024 waves and 6.5M at 2048 — the moat engages only under over-subscription, and that config
>    is the SLOWER one. Measured +63% by halving waves and setting GROUPS=1.
> 2. **"hipBLASLt floor is 12.6-70.6 TF."** WRONG — that is the MoE M=512 subset. Real dense shapes are
>    123-189 TF and the mean gap is ~80x. See the CORRECTION box below.
> 3. **"2 WG/CU (ML8_POOL=128) is the standard."** 1 WG/CU measured 42% FASTER.
> The lever ranking here (L0-L7) is obsolete. Kept for provenance only.

# DSWS S1 (MAD-305) — MORNING BRIEF, 2026-07-27

**READ THIS FIRST, THEN `DSWS_TESTING_LOG.md` SECTIONS 6–8, AND RECONCILE THE CONFIG KNOBS BEFORE
BUILDING OR RUNNING ANYTHING.** (2026-07-17: an entire day was run on `DECENTASN=0` because a brief's
literal command said so, when `DECENTASN=1` was the live mode. Do not execute a command from a brief
without reconciling it against the testing log.)

---

# 1. ★★★ THE NEW PRIMARY CONFIG — FM=2 G=4 ACC_N=2 ★★★

**THIS REPLACES `FM=1 G=6 ACC_N=3` AS THE CONFIG OF RECORD.** Everything from here forward is measured
at FM=2 unless explicitly stated. If you find yourself building `FM=1 G=6`, you are on the OLD config —
stop and re-read this section.

### Why it is now primary (not a preference — a mechanism change)

`s_alloc_vgpr` grow-fail is the **only** admission throttle under `SELFSERVE=1 BATONGATE=1`. The kernel
asserts exactly this at `occ_kernel_dsws_flow.s:1338`:

```
.if SELFSERVE && !BATONGATE
  .error "SELFSERVE requires BATONGATE=1: physical s_alloc_vgpr grow-fail is the only admission throttle."
```

The FATTOK/MAXFAT software token layer is compiled to no-ops. **grow-fail was EXACTLY 0 on every run in
this project's history** — the dyn-VGPR moat, the entire competitive claim of this design, had never once
engaged. FM=2 doubles `ACC_STRIDE` (`FM*FN*1024` = 8192 vs 4096); the VGPR budget finally binds;
**140,708,123 grow-fail events across the 30-shape sweep.** At FM=1 the moat is inert and the whole
dyn-VGPR thesis is untested. **You cannot evaluate this design at FM=1.**

### BUILD (exact)

```bash
cd ~/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
FM=2 G=4 ACC_N=2 ./build_flow.sh
```

Everything else comes from `build_flow.sh` defaults, which ARE the config of record:

| knob | value | | knob | value |
|---|---|---|---|---|
| `WAVES` | 16 | | `SELFSERVE` | 1 |
| `SEGK` | 256 | | `DECENTASN` | 1 |
| `G` | **4** | | `BANKZERO` | 1 |
| `FM` | **2** | | `BATONGATE` | 1 |
| `FN` | 4 | | `STAGGER` | 1 |
| `ACC_N` | **2** | | `DSWS2_OVERLAP` | 1 |
| `POOL_N` | 1 | | `DSWS2_PREFETCH` | 1 |
| `SSWIN` | 32 | | `DEADMAN` | 1 (TICKS 0.5s) |
| `JDEPTH` | 1 | | `STAGINSTR` / `TFPROBE` | 1 / 1 |
| `BATCH` | 1 | | `KMAJOR` / `CFASSIGN` | 0 / 0 |

**Build identity (verify after building):**
- `sha256(occ_dsws2_w16_flow_gd.bin)` = `a581c7b8b8825392…`
- `.text` = **30,812 B**   ·   **LDS = 17,920 B**   ·   `GROUPS = G/ACC_N = 2`
- super-tile M = `G*16*FM` = **128 rows** (was 96 at FM=1 G=6)
- `2 × 17,920 = 35,840 < 65,536` → **2 WG/CU still fits**; 16 waves × 2 WG/CU = 32 slots = exactly the
  per-CU limit.

### DISPATCH (single shape)

```bash
./gpu_run.sh <logname> -- \
  FLOW_WAVES=16 ML8_POOL=128 DSWS2_FLOW=1 \
  DSWS2_FM=2 DSWS2_G=4 DSWS2_ACC_N=2 FLOW_POOL_N=1 DSWS2_SEGK=256 SSWIN=32 \
  DSWS2_K=<K> DSWS2_ORACLE_MTL=<M/128> DSWS2_ORACLE_NTL=<N/64> DSWS2_ORACLE_STRIDE=1 \
  DSWS2_TARGET_SECS=1.5 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85 \
  STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```

`ML8_POOL=128` = **128 WGs = 2 WG/CU**. This is non-negotiable (kmbandy, standing since the successful
128-WG run). `gpu_run.sh` REFUSES `FLOW_WAVES≠16` or `ML8_POOL≠128`.

**`ML8_COOP_CHUNK_MAXS=0.85` IS REQUIRED AT FM=2.** FM=2 measures **0.81 s/chunk** against the 0.75 s
default compositor cap; without the raise every dispatch aborts. This is a *designed* knob
(`occ_dispatch.cpp:1599` names raising it as the remedy) and the check is **reactive** — measured after
the chunk completes, so an over-cap chunk has already run. **It is NOT `DEADMAN_TICKS`**, which is an
anti-brick floor and must never move.

### FULL REAL-SHAPE SWEEP (the canonical gate)

```bash
python3 dsws_realshape_bench.py live \
  --fm 2 --g 4 --acc-n 2 --segk 256 --sswin 32 --waves 16 --pool 128 \
  --chunk-maxs 0.85 --tag <tag> \
  --json <out>.json --table <out>.txt
```

Wraps `gpu_run.sh`, one dispatch per shape, fail-closed on `oracle bad=0` + WORK-EXACT.
**30/33 shapes legal.** The 3 UNSUPPORTED (`mlmf_router_MLP`, `mlmf_router_out`,
`mlmf_routerout_ML8PAD`) are `n_kseg=1 < 2` at SEGK=256 — identical at FM=1, **not** an FM=2 limitation,
and they become legal at `--segk 128`.

### THE ONE DISPATCH PATH

`gpu_run.sh` is the ONLY dispatch funnel. `dsws_realshape_bench.py live` wraps it. `build_flow.sh` builds
the kernel, `build.sh` builds the host. **Nothing else dispatches.** Do not create ad-hoc run scripts —
four were deleted on 2026-07-23 for exactly this reason.

---

# 2. STATUS — WHAT IS SOLID

- **30/30 real ml8/mlambaformer shapes: oracle `bad=0`, WORK-EXACT, grow-fail firing on all 30.**
  Results: `dsws_fm2_growfix_sweep_v2.json` / `.txt`.
- TF range **0.02 – 4.20**, best `ml8_dense_ffn_down M2048 K9216` @ **4.20**.
- Zero GPU resets during the sweep. Latch clear. Board released.
- The grow-fail work-loss defect is **fixed and validated where it fires** (before: 40% of work lost,
  oracle bad=10,064/20,480).

## WHAT IS *NOT* ESTABLISHED — DO NOT REPEAT THESE CLAIMS

- **WE DO NOT KNOW WHAT THE WALL IS.** "ASSIGN-bound" was RETRACTED (§8 of the testing log).
  `occ[86]` merges empty-frontier + window-full + boundary bails; it cannot identify a bottleneck.
  Track record: *"ASSIGN-starved 76%" became 1.8% purely by feeding the kernel.*
- **`occ[88]=0` does NOT mean "carriers are fed"** — `.Lflow_jwait` does not exist at JDEPTH=1
  (kernel `:2974`). Structural zero, not a measurement.
- **Throughput did not move.** The fix was correctness. hipBLASLt fp8 floor on these same shapes is
  **1.6–189.3 TF** (see the CORRECTION below); we are at 0.02–4.20. The gap is the whole project.

> ### ⚠ CORRECTION 2026-07-27 EVENING — THE "12.6–70.6" FIGURE WAS WRONG, AND IT IS QUOTED ALL OVER THIS TREE
> Caught by an external review (Kimi K3) that read the source instead of trusting the brief.
> **12.6–70.6 is the hipBLASLt band on the ml8 MoE M=512 shapes ONLY** — the small shapes where the
> vendor is at its weakest — traced to `DSWS_MORNING_2026-07-14.md:206`. It was then re-quoted as if
> it were the whole range, including here, in `DSWS_TESTING_LOG.md` and in the 07-27 pre-registrations.
> **THE AUTHORITATIVE HEAD-TO-HEAD IS `RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md`**, which already
> contained an explicit retraction nobody propagated:
>   - hipBLASLt spans **1.6 → 189.3 TF** across these shapes. On the big DENSE shapes it is **123–189 TF**.
>   - `ml8_dense_ffn_down M2048 K9216`: DSWS **4.36** vs hipBLASLt **189.3** → we are at **2.3%** of it.
>   - **Mean ratio is ~80x, not 11.5x and not 5–20x** (0.87 vs 69.18).
>   - That same file also retracts all four previously claimed "wins" over the vendor, and the
>     flatness thesis (corrected CV: DSWS 1.128 vs vendor 0.905 — we are LESS flat, not more).
> **We are ~1–2% of the vendor on real dense shapes, not 5–20%.** Any reasoning that used the old
> band understated the gap by roughly 4x. Do not re-quote 12.6–70.6.

---

# 3. THE LEVERS — RANKED

### L0. MEASURE THE WALL FIRST (`RESVPROBE`) — blocks everything else
We are about to spend GPU time on levers without knowing what limits us. `RESVPROBE` already exists and
splits the feed-path bails **by cause**: `occ[87]` CAS-loss (cursor contention) vs `occ[89]` window-full
(consumers behind) vs boundary remainder, against `occ[96]` wins. Its verdict logic is trustworthy
*because it is computed from the split, not the merged total*.
Needs a bin built `RESVPROBE=1`, run with `DSWS2_RESVPROBE=1`.
Its own verdict strings tell you where to go next:
- `WINDOW FULL -> STAGE-BOUND` → ASSIGN is ahead; the wall is staging/drain. **Do not shard the cursor.**
- `CURSOR-CONTENDED` (>1 CAS collision per reserve) → shard `ASSIGN_HEAD`.
- Neither → the empties are ZLOCK boundary bails (serialization).

### L1. ★ THE OPERAND LAYOUT IS THE BINDING CAP ON TILE SIZE — AND SEGK SHRINKS IT ★
**The highest-leverage untested idea.** Every larger tile (`FM=2 G=6`, `FM=4 G=4`, `FM=2 G=8`,
`FM=4 G=6`) is refused by the kernel's own guard: *"DSWS2 single-slot operand layout exceeds the 65536B
WGP limit."* So the cap is **not** the register file (RGA: peak live VGPR 48/256 at FM=1, 82/256 at FM=2 —
**32% used, 0 spills**) and **not** LDS (17,920 of 65,536 = 27%). It is the operand layout:

```
opstride = FN*16*SEGK + G*16*FM*SEGK        (40,960 B at G=6 FM=1 SEGK=256)
```

**`opstride` is LINEAR IN SEGK.** `SEGK=128` halves it; `SEGK=64` quarters it. SEGK is already a
sanctioned free knob `{64,128,256}` (kmbandy). **So SEGK=128 may unlock `FM=4` or `G=8` — tile sizes
currently refused.** That is the direct attack on the 7.38 non-WMMA-per-WMMA instruction overhead (§L4).
Caveat: SEGK=256 was measured optimal for throughput on the OLD config (128 = +21.2% slower, 64 = +59.6%),
but that was measured at FM=1 where the tile could not grow. **The trade is now different and must be
re-measured, not assumed.**

### L2. ★ THE dyn-VGPR MOAT IS LIVE FOR THE FIRST TIME — RE-OPEN WHAT WAS KILLED ★
The stagger / traveling-peak was killed **twice** on the reasoning "grow-fail=0 so the budget never
binds." **That reasoning is now void.** From the 07-13 lesson: *"I built a regime where collisions were
impossible and called collisions non-existent."* We are finally in the regime where they are possible.
- `MAXFAT` / FATTOK software token layer is compiled to no-ops under `BATONGATE=1`. With the budget now
  binding, is admission control worth re-enabling? (`MAXFAT=0` today means "cap = ACC_N".)
- Task **#43** (traveling-peak baton = pure notification) is `in_progress` and was designed for exactly
  this regime.
- A failed grow does `WaitIdleExceptStoreCnt()` — **a full pipeline drain, 140.7M times**. Historically
  eliminating all of them was worth only 0.3%, but that was measured when grow-fail was ~162k, not 140M.
  **Re-measure; do not carry the old verdict forward.**

### L3. G IS THE CONCURRENCY LEVER — COAST IS GEOMETRY, NOT MEMORY
`G == ACC_N ==` the number of waves that can compute **concurrently** (`SL_RBNEXT` hands out rowblks
`0..ACC_N-1`). Low G *geometrically forbids* most of the fleet from ever issuing a WMMA — **that is what
"coast" is.** We run `G=4 ACC_N=2` → only 2 concurrent computing waves per group. Coast is ~97%.
Raising G trades against the operand layout (L1) and against 2 WG/CU:
`G=8 ACC_N=4 FM=2` → LDS `512 + 4*8192 + 1024 = 34,304`; `2 × 34,304 = 68,608 > 65,536` → **1 WG/CU only.**
So there is a real G-vs-occupancy frontier to map. **L1 (shrink opstride via SEGK) is what buys headroom
to raise G without losing 2 WG/CU.**

### L4. INSTRUCTION OVERHEAD — 7.6× WORSE THAN THE REFERENCE
Non-WMMA instructions per WMMA (static count, RGA):
```
FM=1 G=6 ACC_N=3   13.15
FM=2 G=4 ACC_N=2    7.38     (1.78x better — this is why FM=2 matters beyond the moat)
the 165.7 TF ref    0.97     <- 7.6x below us
```
Bigger tiles amortise the non-WMMA work. This is the *same* lever as L1 and the clearest quantitative
target we have. **CAVEAT: static instruction count ≠ throughput** (ports co-issue), and the
"WMMA-bearing region" measured was first-WMMA-to-last-WMMA, which was never proven to be only the k-loop.

### L5. THE 0.81 s PER-CHUNK FIXED COST — UNEXPLAINED
Per-chunk wall was **0.81 s at BOTH `ML8_COOP_CHUNK=512` and `=256`.** A knob sweep returning a constant
is the fingerprint of a fixed cost elsewhere (07-13 lesson — we ignored the first flat sweep and paid for
it). Corollary, inverted from intuition: **fewer, bigger chunks are strictly cheaper.** Worth one
experiment: does chunk=640 (all tiles, 1 chunk) also cost 0.81 s? If yes the whole per-dispatch cost is
fixed and chunking is pure overhead.

### L6. FEED IT / CLOCK VALIDITY — CHECK BEFORE QUOTING ANY TF
Standing rule: no throughput verdict from under ~1 s of steady state. The card is `perf_level=manual`,
idles ~1147 MHz, boosts ~2350 MHz, threshold **~0.5 s**; runs below it read ~5.3 TF where boosted runs
read 8.8 TF — **same kernel, same shape**.
Measured on last night's sweep: mean per-rep span **0.688 s**, and **22/30 runs ≥ 0.5 s**.
**So 8 of 30 shapes may be clock-suppressed.** Identify them and either deepen K (`DSWS2_K`, C unchanged)
or raise `DSWS2_TARGET_SECS` before treating their TF as real. Note the dispatch is **fully drained
between reps** (fence signalled, waves exited), so clocks can sag between reps.

### L7. SEGK=128 ALSO UNLOCKS THE 3 MISSING SHAPES
`mlmf_router_MLP`, `mlmf_router_out`, `mlmf_routerout_ML8PAD` need `n_kseg ≥ 2`. `--segk 128` makes
33/33 legal. Cheap completeness win, and it composes with L1.

---

# 4. MORNING STEPS — IN ORDER

**0. Session-start ritual (mandatory).** Read this brief → `DSWS_TESTING_LOG.md` §6–8 → reconcile the
   config knobs against the log. Confirm the tree still has the grow-fail fix
   (`grep -c Lflow_da_gf_stage_walk occ_kernel_dsws_flow.s` → 2).

**1. `board_check` → `board_claim`.** Check IMMEDIATELY before claiming, every time. Size `ttl_hours` to
   the whole planned campaign (a 3 h TTL expiring mid-campaign caused a real collision on 07-26).

**2. L0 — MEASURE THE WALL.** Build `RESVPROBE=1 FM=2 G=4 ACC_N=2 ./build_flow.sh`; run ONE fed shape
   (`ml8_dense_ffn_down M2048 K9216`, the best-fed shape we have) with `DSWS2_RESVPROBE=1`.
   **Pre-register the expected split before looking.** This is a changed bin → rule 2 → one dispatch,
   then stop and read. **Do not start any lever before this returns.**

**3. Branch on the RESVPROBE verdict:**
   - *window-full / STAGE-BOUND* → the wall is staging/drain → go to **L3 (raise G)** and **L2 (stagger)**.
   - *cursor-contended* → shard `ASSIGN_HEAD`.
   - *boundary bails* → attack ZLOCK serialization.

**4. L1 in parallel, OFFLINE (no GPU needed).** Compute the operand-layout algebra for
   `SEGK ∈ {64,128,256} × FM ∈ {2,4} × G ∈ {4,6,8}` and find which combinations clear the 65,536 B
   single-slot guard AND keep `2 × LDS ≤ 65,536` for 2 WG/CU. **Assemble each candidate** (the kernel
   `.error`s at build time, so this costs nothing but CPU) and produce the legal-tile frontier table
   BEFORE asking for silicon. This is rule 6 (max work offline first) and it is the highest-value
   offline work available.

**5. Then, and only then, run the winning tile** as a bring-up (one dispatch), then the full 30-shape
   sweep as the gate.

---

# 5. TRAPS, RULES, AND THINGS THAT WILL BITE

### The pattern that cost most of 2026-07-26: **A TOOL ASSERTING MORE THAN IT MEASURES**
Ten separate instances in one day. Before believing any diagnostic:
1. **A writer exists** (`grep` the call site, not the definition).
2. **The writer is compiled in** at this config.
3. **The site is REACHED** at this config. ← *new rule; `occ[88]` was wired but unreachable at JDEPTH=1.*
An unwired counter and a wired-but-unreachable counter produce the **identical** symptom: a confident zero.

### Derive expectations from the host's formula, never from memory
On 07-26 I pre-registered `computed == 92,160` from a remembered formula; the truth was **184,320**
(`G*TOTAL_super*reps`, I dropped `reps=2`). 92,160 is *exactly half* — a 50% work loss would have read as
SUCCESS. Use the host's own printed expectation.

### Never cache or hand-copy a `.bin`
`build_flow.sh` emits `.bin` AND `.lds` **together**. Copying the bin without its sidecar caused a
**MODE1 GPU reset** on 07-26 (host allocated 13,824 B for a kernel needing 17,920 B) which killed
`llama-server`, **Hyprland**, and `hyprlock` — locking kmbandy out of his desktop. `gpu_run.sh` now
fail-closed refuses a missing/stale sidecar. **Build in place; never copy.**
Desktop recovery: `hyprctl --instance 0 eval 'hl.clear_crashed_lockscreen()'`.

### A MODE1 reset kills EVERY GPU client, including the compositor
This is *not* CLAUDE.md rule 7 (HBM starvation). A reset destroys the GL context and Hyprland aborts in
`CHyprOpenGLImpl::begin` on its next frame. After any reset, enumerate casualties via
`systemd-coredump` journal entries — **and compare each survivor's START TIME to the incident** (a live
pid is not evidence it never died).

### GPU forensics have a ~38-minute half-life on this box
A `razeraccessory` error loop floods the kernel journal; the 17:08 reset lines were gone by 19:00.
**"0 resets" may mean "the journal no longer retains it."** Worth fixing independently.

### Other standing rules
- **DEADMAN is armed and does NOT cover LDS misallocation.** It was on during the reset and did not
  prevent it. Never raise `DEADMAN_TICKS`.
- **NEVER STAGE ANYTHING** — the git tree is SHARED with a live weight-pager session. No `git add`.
- **Codex/Grok subagents: implementation and source analysis ONLY. Never GPU, never inference.**
  Use `mcp__mad-lab-memory__codex_handoff` (kmbandy, 07-26 — "significantly more reliable" than the
  codex-rescue subagent).
- **Identify a Codex rollout BY CONTENT, never by recency** — this box runs Codex from multiple sessions;
  I pulled another session's task and nearly reported it as my own result.
- My **static review of the DSWS claim/feed/frontier handoff has lost 4/4 times.** An independent
  adversarial review is mandatory before trusting any change to it.

---

# 6. WHAT HAPPENED 2026-07-26 (brief — detail is in the testing log §6–8)

1. **A GPU reset (17:08), my fault** — `.bin` copied without its `.lds` sidecar. Killed llama-server,
   Hyprland and hyprlock; kmbandy locked out of his desktop. Root-caused, fixed, fail-closed guard added.
2. **FM=2 exposed a latent work-loss defect** in the grow-fail path (40% of work lost, oracle bad=10,064).
   Codex diagnosed it: the fallback published a claimable slot but never advanced `STAGE_HEAD`.
   Arithmetic closed to the digit (73,474 lost).
3. **Fixed** (`.Lflow_da_gf_stage_walk`), reviewed, and **validated 30/30 on real shapes.**
4. **Ten host/harness diagnostic defects fixed** — `occ[96]` missing GROUPS (manufactured a false
   "over-emission" that cost an evening), `occ[97]` mislabelled, `occ[98]` unwired, coast-invariant
   unattributed, `ldsBytesRaw` over-count (the always-on warning that caused the reset), a latent
   `SSWIN*32` **under**-allocation (the dangerous direction), `gate_lds.sh` ignoring its own env,
   `LIVE_FM` hardcoded, no `--chunk-maxs`, and the `AMBIGUOUS_REAL_N` attribution halt.
5. **Two claims RETRACTED**: "ASSIGN-bound" and "carriers are fed."

**Net: the kernel is correct at a config where the dyn-VGPR moat finally engages. Throughput is
unchanged and the wall is unmeasured. That is exactly where the work starts tomorrow.**
