# Weight pager / DS4 expert prefetch — handoff

**Written** 2026-08-06, end of day. **Author** Claude (outgoing). **For** whoever picks this up next.

Read the "Traps" section before you run anything. Most of today's wasted time is in it.

---

## 1. What this is

DS4-Flash (284B total / 13B active, MXFP4 QAT natively — **quantization is off the table**) with
~150 GB of routed experts that do not fit in VRAM. Experts are served over the network by
`llama-wp-expert-worker` processes, dispatched from a spine.

Layout (from the harness header, and it is accurate):

| machine | GPU | role | experts |
|---|---|---|---|
| mad-lab-main | RX 6900 XT (ROCm1) | **spine**, `ds4-dense.gguf` fully resident | — |
| mad-lab-main | R9700 (ROCm0) | worker | 85..255 (~98 GB) |
| mad-lab-main | CPU | worker, DSpark blk.43-45 | 0..84 |
| mad-lab-2026 | GTX 1070 (CUDA0) | worker | 0..84 (~49 GB) |
| mad-lab-2026 | RX 480 (Vulkan0) | worker | 0..84, load-shared with the 1070 |

**STANDING NO — do not propose re-sharding the expert split.** 2026 holds 0–84, main holds
85–255. Settled by kmbandy.

The idea under test: the spine knows hash-layer (0..2) expert ids from the token id alone
(`ggml_get_rows(ffn_gate_tid2eid, inp_tokens)` — pure lookup, zero prediction error), so it can
hint them to the workers ahead of the dispatch, and the workers can read those pages during
their idle window instead of on the critical path.

---

## 2. Repo state

- Branch `main`, **24 commits local and UNPUSHED**: `a07f7e310..HEAD` (`6e51845b6`).
- **`a07f7e310` ALONE DOES NOT RUN.** It aborts on
  `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)` at `src/llama-context.cpp:2527`.
  The baseline exists only as that commit **plus ten uncommitted WIP files**. A checkout of the
  commit is not a checkout of the baseline.
- `~/GitHub/llama.cpp-base` exists **on both machines** as `a07f7e310` + those ten patches, built
  and working. Keep it; it is the reference baseline.
- **Eleven modified files in the tree carry unrelated WIP that is not yours** —
  `common/speculative.cpp`, `src/llama-context.cpp`, `src/llama-kv-cache*.{h,cpp}`,
  `src/models/deepseek4.cpp`, `tools/server/server-context.cpp`,
  `ggml/src/ggml-cuda/fattn-common.cuh`. A bare `git add` on any of them sweeps someone else's
  work into your commit. **Stage hunks and verify.**
- **`docs/dev/harness-2026-08-05-ds4_full.sh` is modified and UNCOMMITTED** — see §4.1. It is a
  one-line fix plus a comment; syntax-checked (`bash -n`), copied to 2026, never used in a
  completed run. Commit it or revert it, but decide deliberately.

---

## 3. How to run anything

```bash
# FROM mad-lab-2026. NOT from main.
ssh mad-lab-2026
cd ~/GitHub/llama.cpp
HOSTVICTIM_2026=0 HOSTVICTIM_MAIN=0 bash docs/dev/harness-2026-08-05-ds4_full.sh
```

**Run it from mad-lab-2026.** The harness launches the 2026 workers locally and reaches
*everything* on main via `ssh mad-lab-main`. Main cannot ssh to itself (its own key is not in its
own `authorized_keys`; `ssh mad-lab-main` from main resolves to `127.0.1.1` and is refused). Run
it from main and the R9700 and DSpark workers silently never start — you get
`pipe: connect to ...:8801 failed` and it reads like a worker crash. I lost a run to this.

### Config of record

**~22–23 t/s prefill, ~3.5 t/s decode, acceptance ~0.84.** `CTX=8192`, `NPRED=256`, `SPEC=1`
(DSpark draft on), `UBATCH` unset.

**The config of record does NOT include the host victim tier.** You must pass
`HOSTVICTIM_2026=0 HOSTVICTIM_MAIN=0`. The harness has this comment —

```
# HOST VICTIM TIER (RAM L2 between VRAM and NVMe). NOT part of the config of record.
HOSTVICTIM_2026=${HOSTVICTIM_2026:-1073741824}   # 1 GiB per 2026 worker
HOSTVICTIM_MAIN=${HOSTVICTIM_MAIN:-6442450944}   # 6 GiB, R9700
```

— sitting directly above two non-zero defaults that turn it on. **A bare run is therefore NOT the
config of record.** This invalidated my final A/B of the day; every "control" I ran had a RAM tier
in it. Fixing those defaults to `0` is the obvious cleanup and I did not do it.

The harness's own rule, in capitals in the file, is right and worth obeying: *ALWAYS RUN THE
CONFIG OF RECORD, CHANGING ONLY THE ONE ELEMENT UNDER TEST.* I quoted it and then violated it
three times in one afternoon by stacking `PREFETCH_HINT` + `SPEC_PAGEIN` + a zeroed 2026 tier into
both arms of an A/B.

---

## 4. Bugs found today

### 4.1 `VKSPLIT` made `VKFIX` unreachable — FIXED, UNCOMMITTED

```bash
VKSPLIT=${VKSPLIT:-1048576}     # was: ":-" substitutes when unset OR EMPTY
VKSPLIT=${VKSPLIT-1048576}      # now: "-" substitutes only when UNSET
```

Downstream:

```bash
if   [ -n "${VKSPLIT:-}" ]; then VKENV="$VKENV GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES=$VKSPLIT"
elif [ "${VKFIX:-1}" = "1" ]; then VKENV="$VKENV GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1"; fi
```

With `:-`, an explicit `VKSPLIT=` got the default handed straight back, so the `elif` could never
be reached from the environment. **`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` had never been set in any
run, ever** — `VKFIX` has been dead code since `VKSPLIT` acquired a default. The same file uses the
correct idiom two hundred lines earlier (`SPEC=${SPEC-1}`) for exactly this reason.

Verify the flag actually arrives before trusting any VKFIX arm:

```bash
p=$(pgrep -f "device Vulkan0"); tr '\0' '\n' < /proc/$p/environ | grep GGML_VK
```

### 4.2 `HOSTVICTIM` defaults contradict their own comment — NOT FIXED

See §3. Recommend setting both defaults to `0`.

---

## 5. Findings — measured

Everything here was measured today and I believe it. Numbers taken with the tier **off** unless
stated.

### 5.1 The RX 480 migrates its entire slot pool out of VRAM

Sampling `/sys/class/drm/card1/device/mem_info_{vram,gtt}_used` every 2 s through a run:

```
time       vram480    gtt480  vram1070 GPUActive     Shmem     avail
17:56:44         5      1469        17      1469       537     10276   <- idle
17:57:09         5      7532      7873      7544      1766      2523   <- pool ENTIRELY in system RAM
   ... 20 consecutive samples, 26 s ...
18:00:10      7129       468      7919       468      1768      9294   <- back in VRAM
```

The RX 480 dumps all **7.5 GB** of its expert slot pool into GTT (system RAM) and back.
**~17–21% of post-load samples**, consistently, across four separate runs. The GTX 1070 on the
same box, same shard, same `--slots 550`, **never does it once** after loading — clean control.

Minimum `avail` during those windows, with the tier **off**: **2205 MB**.

### 5.2 That is the OOM mechanism

At 17:35 with `HOSTVICTIM_2026=3221225472` (3 GiB × 2 workers) the box OOM-killed seven processes
— both workers plus `atlassian-mcp`, `llama-router`, `mneme.service`, `gitnexus-serve`,
`mad-lab-music-mcp`. All auto-restarted; nothing was lost. Kernel state at the kill:

```
Node 0 Normal   free:78596kB   min:78836kB       <- free BELOW the min watermark
                active_file:120kB inactive_file:1192kB   <- page cache reclaimed to 1.3 MB
                all_unreclaimable? yes
Free swap = 0kB
                shmem:7066852kB
                gpu_active:5858780kB  gpu_reclaim:0kB
```

`MemAvailable` said 10 GB. The kernel had **78 MB** and had already reclaimed the page cache to
1.3 MB. `MemAvailable` is a forecast that assumes page cache is reclaimable and anon can swap;
**mad-lab-2026 has ZERO swap**, so no anon or shmem page can ever be given back.

**`Node 0 GPUActive` is exactly the AMD GTT figure** — verified, they track each other sample for
sample (468 and 468). So the 5.86 GB of `gpu_active` at the OOM was §5.1 landing on top of a 6 GiB
tier. **The OOM was not a tier sizing error.**

### 5.3 The tier's host cost is exactly its budget, and disjoint from GTT

One worker at a time, no spine, no inference, deltas against a verified floor
(`Shmem 537 MB, GPUActive 1467 MB`):

| arm | Shmem | GPUActive |
|---|---|---|
| baseline | 537 | 1467 |
| CUDA, tier off | 749 | 1467 |
| CUDA, tier 3 GiB | 3821 | 1287 |
| Vulkan, tier off | 537 | 1469 |
| Vulkan, tier 3 GiB | 3617 | 1469 |

The tier costs +3072 MB (CUDA) / +3080 MB (Vulkan), entirely in `Shmem`, and moves `GPUActive` not
at all. CUDA's staging is shmem too (+212 MB, `WP_STAGING_PINNED=1`); Vulkan's is anon
(`posix_memalign`, `PIN=0`). Reproduce with
`/tmp/.../scratchpad/isolate-hostmem.sh` (also at `/var/tmp/isolate-hostmem.sh` on 2026).

### 5.4 `HOSTVICTIM_MAIN=6 GiB` drives main into swap

Main during a run: **131 MB free, 5.4 GB swapped, `shared` 7768 MB**, R9700 worker RSS 6.19 GB.
The tier is shmem and main *has* swap, so part of the tier lives on disk — a tier whose entire
purpose is avoiding an NVMe read was itself being read from swap. Board RAM alerts fired
repeatedly. Not dangerous (10.5 GB swap free) but it is a real effect and it is the harness
default.

### 5.5 Use-count (LFU) eviction — the one confirmed win

From the controlled regression A/B earlier today (interleaved, 2 reps, all valid, acceptance
0.84286 throughout):

| arm | prefill | decode | decode page-ins | dispatch wait |
|---|---|---|---|---|
| old (`a07f7e310`+WIP) | 22.83 | 3.25 | 12980 | 56.0 s |
| new (HEAD, LFU on) | 21.98 | 3.38 | **12310** | **52.7 s** |
| nlru (HEAD, `LFU=0`) | 23.54 | 3.15 | 12982 | 55.6 s |

**HEAD with every new feature disabled is indistinguishable from the config of record** — 12982 vs
12980 page-ins, wait within 0.5%. So the new code costs nothing dormant. With use-count eviction
on it is **−5.2% decode page-ins, −5.8% dispatch wait**. That is the single confirmed win.

---

## 6. Findings — VOID. Do not build on these.

- **Every speculation number from 2026-08-06.** The tier arms ran with the RX 480 silently not
  participating (harness `VKENV` clobber, fixed in `2fcfd8982`, never re-measured); the rest were
  compared across uncontrolled baselines.
- **The `VKFIX=1` A/B.** Ran it, got `vkfix prefill=18.31 decode=1.80` vs
  `ctl prefill=11.25 decode=2.05`. Both arms had the RAM tier on (§3), so neither is the config of
  record, and the control at 11.25 prefill is half what the same build did at 16:31 the same day.
  The RX 480's own timers were identical between arms (`ns_prep_set` 2.65–2.72 s vs 2.69–2.76 s,
  `ns_wait` 16.9–17.7 s), so there was no vkfix effect visible. **VKFIX is untested.** It is still
  worth testing — it has genuinely never been on.
- **The host victim tier's value, on either machine.** Prior finding (−6% page-ins for −21%
  decode, 19 demotes per hit) is void: measured with one of three workers absent and n=1 tok/s.
- **Host landing** (`WP_EXPERT_SPEC_HOST`) has **never executed** — `host_landed=0` in every
  attempt, `host_skip[vram]=211` of 219, because predicted pages were already resident. Either
  accept it is vacuous by construction (predictions come from the previous block, whose experts
  are still in VRAM) or find the case where it pays.
- **The previous-block predictor** (`WP_SPEC_PREDICT_PREV`): flat-to-negative in every arm.
  `WP_SPEC_PREDICT_N` is a **no-op at this block size** — draft blocks are ~1.8 tokens
  (59 accepted / 70 generated), so `min(N, block)` is the whole block for any `N>=1`. Do not sweep
  `N` again without changing block size.

---

## 7. Open questions

1. **Why does the RX 480 evict its whole pool to GTT, and what triggers it?** Not investigated —
   found it at 18:00 and ran out of runway. It is 7.5 GB of transient system-RAM pressure and
   ~20% of the run. This is the biggest unexplained thing on the board. `GGML_VK_ALLOC_LOG=1`
   (harness knob `ALLOCLOG=1`) may help.
2. **Does it cost throughput, or only memory?** While the pool is in GTT every matmul against it
   streams over PCIe. The harness documents that cost as 1413 µs vs 193 µs for the BAR case. I did
   not establish whether the migration windows line up with slow tokens — the per-worker
   `ns_prep_set`/`ns_compute` were not decisively different. **Correlating the GTT trace against
   per-request timestamps (`REQLOG=1`) is the single highest-value next measurement.**
3. **Why does the RX 480 report `host_skip[bad/pin/vram/tier]=0/0/0/0` while the 1070 saw 219?**
   Same flags, same hint volume. The `VKENV` clobber explains zero *tier* activity but not zero
   entries reaching the host-landing filter. Third time that card has silently not engaged a
   feature.
4. **Is the speculative lease's win real?** Measured `used/hinted` 1.7% → 11.1% with LATE −61%
   *and* fewer speculative reads. But measured on a fleet where the RX 480's tier was disabled and
   against uncontrolled baselines. **Highest-value thing to re-confirm** — if it survives a clean
   re-take it is the mechanism that makes prefetch work at all.
5. **Does prefetch have any path to paying off?** Hash layers are 3 of 43 and ~12.5% of page-ins;
   the honest decode ceiling was always ~8%. Best measured `used/hinted` ~12%. **The oracle is
   EXACT** (mispredict 6 of 4326), so the failure is residency, not prediction. Worth deciding
   deliberately whether this is worth more runway.

---

## 8. Traps

Ordered by how much time they cost.

1. **A health check that CONNECTS kills the worker.** The expert protocol opens with the *server*
   sending a 366-byte HELLO; a probe that connects and hangs up makes that send fail and the
   worker exits (`pipe send failed (bytes_sent=0, size_to_send=366)`). Gate on the worker's own
   `expert worker listening` log line. The harness already has a correct `ss`-based gate at ~line
   768 — read that far before building another.
2. **The shell's working directory persists between tool calls.** After `cd`-ing into the base
   worktree, every subsequent `git`/`grep`/`scp` ran there. I briefly concluded a subagent had
   reverted an API when I was reading `a07f7e310`, and I scp'd base-worktree files into 2026's HEAD
   checkout, breaking that build.
3. **`HOSTVICTIM_*` is PER WORKER.** 2026 runs two, so 3 GiB each = 6.4 GiB unswappable. See §5.2.
   Given §5.1, the practical ceiling on 2026 is ~2 GiB/worker and even that is not proven.
4. **`pkill -f <pattern>` matches your own ssh session's command line** and kills it — you get
   exit 255 and think the host died. Use explicit PIDs.
5. **Killing a wrapper script does not kill the harness it spawned**, and the harness relaunches
   workers. Always verify with `ps -eo pid,args | grep -E "llama-wp-expert|ds4_full"` afterwards,
   on **both** machines.
6. **`requests=43` in the `expert dispatch worker` line is ONE decode token** (43 = `n_layers`),
   not the run. `n_tokens=1` confirms it. I compared one arm's stall token (23.9 s, vs an 819
   ms/token run mean) against another arm's fast token (229 ms vs 650 ms mean) and reported a
   "151× improvement" that was pure sampling artifact. **Per-worker page-ins and the cumulative
   `wp expert worker stats` counters are the reliable metrics. n=1 tok/s is not** — tok/s spanned
   15.70–25.65 prefill and 1.22–3.40 decode across one day.
7. **A `str.replace` that matches nothing is a silent no-op.** `set_speculative_tier(true)` never
   landed because of one wrong indent, shipping a disarmed feature. Same day: a perl mutation
   corrupted a file, the build failed, and the **stale test binary reported "all tests passed"**.
   Check the build RC and the binary mtime before trusting any test result.
8. **Tooling that reads a previous run's files and reports them as this run's** bit three times
   (req logs, main's `$OUT`, 2026's worker logs feeding a readiness gate). Delete the arm's
   artifacts BEFORE the arm runs, on BOTH machines. `WP_HINT_LOG` is safe (`fopen(p,"w")`).
9. **The worker rejects `WP_EXPERT_HOST_VICTIM_BYTES=0`** ("requires a positive integer"). Tier-off
   means the variable **absent**, not zero. The harness gets this right; ad-hoc scripts will not.
10. **An unexplained defect in the measured system blocks the conclusion.** I found the RX 480 not
    participating, wrote it up as a footnote, and published the verdict anyway. Honest labelling is
    not a substitute for not publishing the number.

---

## 9. Suggested order of work

1. **Fix the two lying defaults** (§4.1 commit, §4.2 set to 0). No GPU. Stops this recurring.
2. **Re-establish the config of record**, 2 reps, `HOSTVICTIM_*=0`. Confirm ~22–23 / ~3.5. Nothing
   below is interpretable until this reproduces.
3. **Chase §5.1** — the RX 480 GTT migration, with `REQLOG=1` so the windows can be correlated
   against slow requests. This is the largest unexplained effect and it is on the critical path for
   both memory and throughput.
4. **Then** re-take the lease (§7.4) and the speculation matrix against that clean baseline.
5. Further eviction policies can be swept **offline** from the reference stream (`REFLOG=1`) at
   zero GPU cost — the simulator predicted the measured page-in delta to 0.11%. Belady says 36% of
   LRU's page-ins are recoverable; use-count captures ~10% of that; ARC/LRU-2/2Q all captured <5%.

---

## 10. Pointers

- `docs/dev/2026-08-06-all-runs.txt` — every run of the day with per-worker page-ins (untracked).
- `docs/dev/analyze-hint-log.py` — resolves the `WP_HINT_LOG` event stream into used/late/
  mispredict. Denominator is **hinted ids**, not `spec_pi` (`spec_pi` moves with eviction policy).
- `docs/dev/harness-2026-08-05-ds4_full.sh` — `MAIN_REPO` is overridable so an A/B can vary
  binaries without varying the harness.
- On mad-lab-2026: `/var/tmp/vram-trace-A-vksplit.txt` and `/var/tmp/vram-trace.txt` (the GTT
  traces, §5.1); `/var/tmp/run-vkfix.out`, `/var/tmp/run-ctl.out` (the void A/B, §6);
  `/var/tmp/isolate-hostmem.sh` (§5.3).
- Board announcement `063e4ba1` — the 14:56 OOM and its cause.
- Mneme brief `6173a202-cfa1-4a79-a115-9c736d917fb8` — the state as of ~16:30 today, before the
  GTT finding.

---

## 11. Standing constraints

- **Never run LLM inference or touch a GPU without asking kmbandy first and waiting for a reply** —
  *except* that holding a valid board claim IS the authorization to run. Claim via the board tools;
  you do not need a separate per-run greenlight.
- **Never restart a live service without confirming**; never `systemctl --user daemon-reload`
  without confirming.
- **DO NOT TOUCH on mad-lab-2026:** the nemotron embedder (`:8082`) and `llama-router` (`:8093`)
  are LIVE FLEET SERVICES. Also `mneme.service` and the dashboard. Kill only PIDs you started.
- **Non-Claude subagents (Codex/Grok/etc.) must never run GPU work or inference.** Implementation
  and build work only.
- **Quantization is out of the question.** DS4-Flash is MXFP4 QAT natively.
- **Do not re-shard the expert split.**
