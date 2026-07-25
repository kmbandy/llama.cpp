# Codex (gpt-5.6-sol) — per-wave / per-dispatch fixed cost analysis, 2026-07-22

## PROVENANCE AND CONTAMINATION WARNING — READ FIRST

Produced by Codex `--model gpt-5.6-sol`, commissioned by claude__main. Codex's own sandbox was
**read-only**, so it could not write this file; the text below was relayed verbatim in its reply and
written to disk by claude__main. No content was authored or edited by claude__main — only this
header and the reconciliation section at the end are mine.

**Codex did NOT dispatch to the GPU.** Assembly and disassembly were done through pipes only.

**INDEPENDENCE IS PARTIAL. Do not read this as clean corroboration of everything.**
Codex was deliberately denied the prior conclusion docs (`DSWS_BRIEF_*`, `RESULTS_DSWS_*`, etc.) and
was given only measured numbers. However, it read `occ_kernel_dsws_flow.s` **after** claude__main had
added the `SPANFLIP` block that morning, and it **cites `occ_kernel_dsws_flow.s:2212` — which is
claude__main's own comment stating the launch/retire-ramp hypothesis in so many words.** Therefore:

- **INDEPENDENT** — the instruction-count finding that the coast loop cannot account for the span
  (derived from actual disassembly, a different route from claude__main's counter arithmetic).
- **INDEPENDENT** — the idle-workgroup, duplicate `s_alloc_vgpr`, uncounted exit-barrier, and
  wrong-deadman-comment findings. These were not in claude__main's analysis at all.
- **CONTAMINATED** — Codex's endorsement of the ramp hypothesis and its recommendation to run
  `SPANFLIP` first. It read the hypothesis in the source comment. This is NOT a second opinion on
  that point and must not be cited as one.

---

## 1. One coasting wave per iteration

The exact gfx1201 disassembly of the normal door-1, already-served `CFASSIGN` path contains **89
instructions per non-watchdog iteration**. Every 64th iteration executes eight additional watchdog
instructions, giving **89.125 instructions/iteration** on average.

| Disassembly block | Instructions | Source |
|---|---:|---|
| `occ_kernel+0x3480`, deadman fast path | 3 | `occ_kernel_dsws_flow.s:1437`, called at `:3083` |
| DECENTASN jump to body | 2, or 3 for wid0 | `occ_kernel_dsws_flow.s:3086` |
| `occ_kernel+0x3ec0`, unchanged role | 10 | `occ_kernel_dsws_flow.s:3257` |
| Role dispatch | 2 | `occ_kernel_dsws_flow.s:3276` |
| `occ_kernel+0x3f04`, door-1 test | 12 | `occ_kernel_dsws_flow.s:3282` |
| `occ_kernel+0x7464`, `.Lflow_coast` | 11 | `occ_kernel_dsws_flow.s:4811` |
| `occ_kernel+0x5398`, feed-empty checks | 8 | `occ_kernel_dsws_flow.s:4002` |
| `occ_kernel+0x53c0`, CFASSIGN already-served path | 30 | `occ_kernel_dsws_flow.s:4039`, `:4069` |
| `occ_kernel+0x73fc`, feed-empty yield | 11 | `occ_kernel_dsws_flow.s:4783` |

Per normal iteration:

- Ten `ds_load_b32` LDS reads, each immediately followed by `s_wait_dscnt 0` and
  `v_readfirstlane_b32`; this expansion comes from `lds_get` / `lds_get_r` at
  `occ_kernel_dsws_flow.s:1057`.
- **Zero global-memory accesses.**
- **Zero LDS atomics** on the steady already-served path.
- **Zero global atomics.**
- One `s_sleep 2` at `occ_kernel_dsws_flow.s:4803`.
- One `s_sendmsg_rtn GET_REALTIME` plus `s_wait_kmcnt` **every 64 iterations** at
  `occ_kernel_dsws_flow.s:1439`.
- No barrier and no cache-scope qualifier. LDS instructions have no `SCOPE_*`; every global atomic
  elsewhere in this kernel explicitly uses `scope:SCOPE_DEV`, e.g. the counter emitter at
  `occ_kernel_dsws_flow.s:2486`.

When `DA_ZDONE` is locked, the assignment portion exits after seven rather than thirty instructions,
producing a 66.125-instruction loop. Boundary-owner iterations are longer and include an LDS CAS at
`occ_kernel_dsws_flow.s:4184`.

## 2. Quantification

The source equates `2^32` shader cycles with approximately 1.8 seconds, implying about **2.39 GHz**
shader clocks, not the 100 MHz realtime-counter frequency used by the measured span
(`occ_kernel_dsws_flow.s:1340`).

For 825,068 measured coasts:

- Short ZLOCK-bail path: **54.56 million** wave instructions.
- Normal already-served path: **73.53 million** wave instructions.
- With 128 SIMDs (stated by the host diagnostics), ideal instruction-issue time is only
  **0.179 to 0.241 ms** (`occ_dispatch.cpp:2451`).
- To fill the entire 5.816 ms at 128 SIMDs, each coast would need approximately **2,153
  issue-equivalent shader cycles**, versus 66 to 89 decoded instructions.

**Therefore the coast instruction stream does not close the 5.816 ms.** Immediate LDS waits,
`s_sleep`, low effective SIMD utilization, shared-LDS pressure, and message-bus stalls enlarge it,
but they would need a roughly **24x to 33x** amplification at full-SIMD utilization. The source
itself warns that adding one hot-path LDS read caused a 16x regression, so such amplification is
possible but cannot be asserted from instruction count alone (`occ_kernel_dsws_flow.s:4099`).

The two supplied ablations leave:

```
5.81616 - 1.93872 - 0.92468 = 2.95276 ms
```

unattributed. That is about **7.05 million shader clocks**.

The normal timer measures the first wave entering through the last wave retiring. The source already
provides `SPANFLIP` specifically to separate that launch/retire ramp from the all-resident execution
window (`occ_kernel_dsws_flow.s:2212`). Without a `SPANFLIP` measurement or a retire-barrier
iteration counter, the remaining 2.953 ms cannot be uniquely reconciled.

## 3. Contention analysis

The normal loop's ten LDS reads are:

- **Per-wave addresses:**
  - `ROLE[wid]` at `occ_kernel_dsws_flow.s:3258`.
  - `GROWPERMIT[wid]` at `occ_kernel_dsws_flow.s:4793`.
- **Common per-workgroup addresses:**
  - `DRAIN_HEAD`: `:3283` and `:4047`.
  - `STAGE_HEAD`: `:3284` and `:4814`.
  - `ASSIGN_HEAD`: `:4815` and `:4046`.
  - `FLOWTERM`: `:4033`.
  - `DA_ZDONE`: `:4042`.

The regular loop only reads these words, but thirty waves repeatedly issue uniform LDS reads and
immediate waits. At a group/tile boundary all eligible waves contend on the same `DA_ZDONE` LDS CAS;
losers return to the empty loop (`occ_kernel_dsws_flow.s:4184`).

Other shared poll sites:

- **Startup:** all 29 non-coordinator waves poll the common `RINGINIT_OFF` word at `:3067`.
- **Terminal drain owner:** polls `DRAIN_HEAD`, `STAGE_HEAD`, slot generation and `RBDONE` through
  `drain_advance` (`:1213`), then `ASSIGN_HEAD`/`DRAIN_HEAD` at `:4769`.
- **Exit:** every early retiree polls the same `QUIESCE_CNT_OFF` at `:4919`.

Shared device-scope global addresses:

- `occ[20]` tile claim, with atomic return, at `:4277`.
- `occ[0]` live count at entry and exit: `:2953`, `:4899`.
- TF min/max addresses at `:2170`.
- Retire counter addresses at `:2486`.

There is no global flag poll in the steady coast path and no explicit cache invalidation instruction.
Global accesses use `scope:SCOPE_DEV`; LDS accesses have no cache-scope qualifier.

## 4. Other per-wave, non-work cost

The host defaults to 64 workgroups and dispatches `pool * WAVES_LAUNCH * 32` threads
(`occ_dispatch.cpp:1979`, `:2062`). At 30 waves/WG this is **1,920 launched waves for eight tiles.
At least 56 workgroups cannot receive a useful tile** but still pay the complete prologue, LDS
initialization participation, terminal and retire paths.

Per launched wave:

- STAGINSTR zeros nineteen private scalar registers before work (`:2455`).
- TFPROBE performs an RTC message and device-scope atomic at both ends (`:2170`).
- The wave executes `s_alloc_vgpr 32` at entry (`:2946`).
- **`cur_role` is initialized to `-1`, forcing every wave through a second same-size
  `s_alloc_vgpr 32` on first role adoption** (`:2962`, `:3263`). The source says `s_alloc_vgpr`
  drains the wave pipeline before reallocating (`:734`).
- Every wave atomically increments and decrements `occ[0]` at device scope (`:2953`, `:4899`).
- Every wave sends an initial deadman RTC message (`:1405`).

Per-workgroup initialization is also large. With `SELFSERVE=1`, `SLOT_N=SSWIN=32` (`:555`). Wid0
performs five base control stores, thirty role stores, `32 x 8 = 256` slot-control stores, four DA
stores and the final RINGINIT store — **296 lane-zero LDS stores per WG, each expanded into nine
instructions with an immediate `s_wait_dscnt`** (`:1069`, `:2975`, `:2994`, `:3008`). The other 29
waves spin on RINGINIT during those stores (`:3067`), then all execute the one-time split barrier
(`:3063`).

`zero_banks` is **not** per-wave fixed cost. At this geometry each call executes 24 `ds_store_b128`
iterations plus two TILEDONE stores, and it runs at group/tile boundaries (`:1185`, `:4254`). It
scales with useful groups, not launched waves.

At exit:

- One wave broadcasts thirty `ROLE_RETIRE` LDS stores per WG (`:4777`).
- Each wave performs the counter flush, live decrement and TF exit stamp before entering the exit
  barrier (`:4897`).
- Each wave atomically increments `QUIESCE_CNT`, then early retirees repeatedly read that common
  word and sleep (`:4919`).
- **Despite the nearby comment saying "NO RTC/message bus," the compiled `DEADMAN=1` loop actually
  executes `GET_REALTIME` every 64 exit-poll iterations** (`:4928`).

**These exit iterations are not represented in `coast` or `feedMT`, because `cnt_flush` happens
before the exit loop.** They can still delay the final productive wave through LDS, scalar-front-end
and message-bus interference.

## 5. Ranked minimal changes

1. **Cap launched WGs to actual tiles.** For this shape test `ML8_POOL=8`. Permanently, change the
   host pool calculation to `min(poolD, 64, TOTAL)` at `occ_dispatch.cpp:1979`. This removes 56
   guaranteed no-work WGs, or **1,680 of 1,920 waves**.
2. **Disable TFPROBE outside measurement runs.** `TFPROBE=0` removes two per-wave RTC messages and
   two device-scope atomics (`:2170`). The supplied ablation already establishes ~1.94 ms recovery.
3. **Use `CNTLEAN=1`.** Retains `coast` and `computed` for WORK-EXACT while removing the other
   fifteen retire emit sites (`:2205`, `:2499`). Supplied recovery: 0.925 ms.
4. **Remove the watchdog RTC bus from the hot loop without disabling the watchdog.** The measured
   coasts imply roughly 11,000 to 13,000 `GET_REALTIME` calls from coast iterations alone, because
   the cadence is 64 (`:1323`). A minimal A/B is `DEADMAN_EVERY=512` or `1024`; the stronger change
   is to use the existing one-instruction `HW_REG_SHADER_CYCLES_LO` pattern and recalibrate the
   threshold (`:1334`, `:1377`). **Do not use `DEADMAN=0`.**
5. **Skip the redundant first role-adoption shrink.** After the entry `s_alloc_vgpr 32` succeeds,
   `cur_role=-1` forces another `s_alloc_vgpr 32` before any work (`:2948`, `:3263`). Add a
   first-adoption guard that records `s35` without executing the defensive shrink. This removes
   exactly one full-drain realloc path per launched wave.
6. **Reduce repeated CFASSIGN polling for an already-served cohort.** The `s15 == cohort_end`
   rejection occurs only after three LDS reads and cohort division arithmetic (`:4042`, `:4069`). An
   early valid-`s15`/DRAIN check can avoid approximately 23 instructions and two LDS reads on the
   common path.
7. **Give the retire barrier a separate, larger backoff.** Replace only the `s_sleep SLEEPN` at
   `:4943` with a `RETSLEEPN` knob. This reduces uncounted hammering of `QUIESCE_CNT` without
   slowing useful coast/feed responsiveness.

Before performance changes, run the existing `SPANFLIP=1` diagnostic. A small or negative
all-resident window would directly prove that the 5.816 ms number is predominantly launch/retire ramp
rather than coast execution (`:2212`).

## Confidence and what I could not verify

High confidence in the exact instruction counts, active compile-time paths, memory-operation
inventory, launch cardinality and source mechanisms. **The coast loop is not demonstrated to
dominate; its decoded instruction cost does not close the span.**

Could NOT verify without GPU measurement: exact LDS latency, `s_sleep` residency effects,
`s_alloc_vgpr` latency/retry count, exit-barrier iterations, or the all-resident window. The existing
data cannot uniquely divide the remaining 2.953 ms among deadman message traffic, duplicate
`s_alloc_vgpr`, live-counter atomics, serialized LDS initialization and exit-barrier interference.

Direct answers:

- **(a) Did the coast loop dominate the span? — NO.** 66–89 instructions/iteration x 825,068
  iterations ~= 54.6–73.5M wave instructions => only ~0.18–0.24 ms of ideal issue time across 128
  SIMDs, versus the measured 5.816 ms span — a 24x–33x gap.
- **(b) Not verifiable from source alone:** LDS latency, `s_sleep` residency, `s_alloc_vgpr`
  latency/retry, exit-barrier iteration count, all-resident window.

---

# RECONCILIATION (claude__main, 2026-07-22)

## Where the two independent analyses AGREE

Both concluded **the coast loop is not the wall**, by different routes:

| | route | result |
|---|---|---|
| claude__main | counter arithmetic: 873 instrumented spin iters/wave x 1920 waves | all counted spin <= 6–15% of span even at a generous 1000 cyc/iter |
| Codex | disassembly instruction count: 66–89 instr/iter, 128 SIMDs | ~0.18–0.24 ms of 5.816 ms; a 24x–33x gap |

This is genuine corroboration — the routes share no assumptions, and Codex was not told the
hypothesis. **The leading suspect for the fixed cost is dead.**

## What Codex found that claude__main MISSED

1. **56 of 64 workgroups cannot receive a tile.** `pool` is hard-defaulted to 64
   (`occ_dispatch.cpp:1979-1980`) irrespective of `TOTAL`. On the 8-tile bring-up shape that is
   **1,680 of 1,920 waves launched with no work available to them.** claude__main computed the 1920
   figure and never asked whether the work could reach them.
2. **A redundant second `s_alloc_vgpr 32` per launched wave**, because `cur_role` starts at `-1`
   (`:2962`, `:3263`). `s_alloc_vgpr` drains the wave pipeline (`:734`) — a per-wave, not per-work,
   full drain.
3. **The exit barrier's spin is UNCOUNTED.** `cnt_flush` runs *before* the `QUIESCE_CNT` poll loop,
   so those iterations appear in neither `coast` nor `feedMT`. This is a concrete candidate for time
   that is invisible to every existing counter — and it is exactly the blind spot the counter
   arithmetic could not see.
4. **A comment in the source is wrong.** The retire loop is documented "NO RTC/message bus"
   (`:4888`), but the compiled `DEADMAN=1` path issues `GET_REALTIME` every 64 exit-poll iterations
   (`:4928`).

## What must NOT be cited as corroboration

Codex's endorsement of the launch/retire-ramp hypothesis, and its "run `SPANFLIP` first"
recommendation. It read that hypothesis in claude__main's own comment at `:2212`. On that specific
question it is an echo, not a witness.

## Status of the arithmetic

Codex's shader-clock figure (~2.39 GHz, derived from `:1340`) is firmer than claude__main's assumed
1.8–2.4 GHz band, and lands inside it. Its residual of **2.953 ms unattributed** uses the
yesterday-measured 1.939 ms for TFPROBE, which was taken on a *coarse 1 ms-resolution host wall* and
a *different host binary*; it should be re-measured on the tick span before that residual is trusted
to three decimal places.
