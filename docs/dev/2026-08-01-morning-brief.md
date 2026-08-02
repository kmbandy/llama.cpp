# Morning brief — 2026-08-01

Session of 2026-07-31. Read §1 first; it is the only thing blocked on kmbandy.

---

## 1. THE ONE OPEN DECISION

**Dense tensor type for a whole-model DeepSeek-V4-Flash conversion: BF16 or Q8_0?**

The plan was "convert the original ourselves and preserve FP8, ~167 GB, no pointless widening".
That plan is dead as written:

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:2583
GGML_ASSERT(src0->type != GGML_TYPE_F8_E4M3 &&
    "GGML_TYPE_F8_E4M3 is not a real MUL_MAT weight type on the HIP backend ...");
```

`GGML_TYPE_F8_E4M3` exists but is a **sidecar type for our own ml8 work (MAD-223)** — centroids,
not runnable weights. A GGUF with FP8 dense converts fine and then aborts on the first matmul.

Consequence: **Unsloth's BF16 widening is forced, not careless.** I called it stupid; it is the
price of ggml having no FP8 matmul path. That correction is owed.

Options:

```
BF16 dense   14.54 GB   lossless from FP8, 2x the source   -> ~162 GB total (= Unsloth)
Q8_0 dense   ~7.7 GB    real matmul type, SMALLER than FP8 -> ~155 GB total
```

Q8_0 is attractive (smaller than DeepSeek's own release) but its fidelity against FP8 e4m3 is
**unmeasured** — e4m3 gets range from an exponent, Q8_0 is linear within a 32-block with an f16
scale, so they fail differently on outliers. Measurable, not guessable.

Everything else about the conversion is settled and specced. This choice gates the dispatch.

---

## 2. WHAT THE DAY ACTUALLY PRODUCED

### 2.1 The real finding: NVMe duty cycle, not bandwidth, not compute

Measured with an external `/proc/diskstats` sampler (`qd_sample.py`, kept in the job tmp dir and
at `/tmp/qd_sample.py` on main — four lines of arithmetic, no iostat needed, works on both boxes):

```
box            drive     QD      util%   MB/s avg   MB/s while busy   capability
mad-lab-2026   SN750     11.5     40      1230       ~3.1 GB/s        2.86-2.91 @ QD16
mad-lab-main   SN850X    14.0     30      1980       ~6.7 GB/s        2.95 @ QD16
```

The queue is 11-14 deep, NOT 1 — that retires the old "pager effectively runs at QD1" claim. The
drives run at rating whenever busy and are **idle 60-70% of the time**. Bandwidth is lost to
GAPS: layer N blocks on all its experts, then layer N+1 routes and issues, 75 times per token.

### 2.2 Expert deferral — built, measured, kept, DEFAULT OFF

Commits `01ccba04e` (implementation), `afc16c83d` (the reorder that made it work), plus counter
fixes. Sorts each layer's experts by router weight, computes the top K immediately, folds the
rest into the next layer's output, **issuing the deferred reads before the layer returns**.

Throughput is real (256-token runs, arms interleaved ctl/K/ctl):

```
arm    K    tok/s   2026 util%   ns_gap   n_deferred   n_deferred_late
ctl   off   0.901   33.0         126      0            0
6      6    1.020   35.9           4      37,740       0
4      4    1.180   42.4           9      75,480       0
2      2    1.333   44.6           5      106,560      0
ctl   off   0.924   33.8         145      0            0
```

Control mean 0.913 -> 1.333 at K=2 = **+46%**. Utilization 33.0 -> 44.6 monotone, controls
agreeing to 0.8pp. `ns_gap` collapsed ~30x.

**But it FAILS the quality gate at every K tested.** Measured by teacher-forced DECODE NLL
(llama-perplexity CANNOT measure this — deferral is gated to `n_tokens == 1` and perplexity runs
512-token chunks, so it would have measured an inactive feature and read as "no quality cost"):

```
K=4  +1.90% PPL    K=6  +4.21%    K=2  +6.44%     gate was <=1%
```

### 2.3 The PPL instrument cannot resolve a 1% gate — and this is the more useful finding

```
control-vs-control drift   0.94% (session A), 1.44% (session B)
cross-session control mean 1.50% apart
```

K=6 (defers 2) scoring worse than K=4 (defers 4) is **noise**, not signal. Do not set a gate
tighter than the instrument.

**The metric that works is `argmax_agree`**, which I had added only as a secondary check:

```
ctl vs ctl  94.3%   <- IDENTICAL across two independent sessions
K=6         91.7%   excess disagreement  2.6pp
K=4         90.6%                        3.7pp
K=2         83.9%                       10.4pp
```

Monotone, control value reproduced exactly. Use it, not PPL, until the noise floor is fixed.

### 2.4 THE DISPATCH PATH IS NON-DETERMINISTIC — worth fixing on its own merits

Two identical control arms, `temperature 0`, `top_k 1`, teacher-forced identical context,
**disagree on 5.7% of tokens**. That should be zero. Almost certainly float summation order
varying with which worker serves which expert: `choose_worker()` keys off residency, residency
follows cache state, cache state varies run to run.

This is the precondition for gating anything tightly, and non-determinism in a serving path is a
defect independent of deferral.

---

## 3. RETRACTIONS — three, all mine, all from the same error

Each was a number I DERIVED or a counter whose span I ASSUMED, never checked at the source.

1. **"H2D runs at 1.29 GB/s, 60% of the per-request budget."** Derived from a subtraction. First
   direct `ns_h2d` says **~3.0 GB/s**, ~85% of the 1070's gen3 x4 ceiling — and it does that
   PAGEABLE. Page-locked staging is INERT (pinned 2.971 vs pageable 3.006 GB/s on the 1070;
   14.99 vs 16.10 on the R9700) and it BREAKS VULKAN (host-visible device memory is 4096-aligned
   but not O_DIRECT-readable; `read()` returns -1 and prefill dies at layer 3). Default flipped
   to off. Keep only its instrumentation — that is what caught all of this.

2. **"GPU compute is the largest single cost."** `ns_compute` is a SUPERSET:
   `compute_started` is stamped at `wp-expert-worker.cpp:1816` BEFORE `prepare_io`, and closed
   after `batch.complete()` — which waits on every NVMe miss and does every H2D upload. Proof
   from the same run: `ns_read 10.82 + ns_h2d 7.45 = 18.27` vs `ns_compute 18.12`. Actual GPU
   work is `ns_submit` = **0.946 s of a 36 s run, 2.6%**. The GPUs were never the bottleneck.

3. **Every 32-token baseline.** 0.865-0.941 tok/s are prefill-inflated ~11%. Steady state over
   256 tokens is **~0.99 tok/s**. Quote that.

Also retracted: the "expert compute is ~80x above roofline" premise in the batching design
(same `ns_compute` misreading), and my claim that Unsloth's BF16 widening was pointless (§1).

**The pattern: a derived number is a hypothesis; only an instrumented one is evidence. Read what
a counter brackets in the source before building a budget out of it.**

---

## 4. DEEPSEEK-V4-FLASH-0731 — the pivot target

Released 2026-07-31. 304B, MIT, `deepseek4` arch, 256 experts / **6 used per token**, 43 layers,
1M context via yarn, sparse-attention indexer, `hc_mult=4` hyper-connections, `hc_sinkhorn_iters=20`.

**Beats GLM-5.2 on every published benchmark** (AutomationBench 25.1 vs 12.9, Toolathlon 70.3 vs
59.9, DeepSWE 54.4 vs 46.2, DSBench-Hard 59.6 vs 54.5). Against Opus-4.8 it is near parity on
Agents' Last Exam (25.2 vs 25.7) and Terminal Bench (82.7 vs 85.0), with real gaps on NL2Repo
(-15.5) and DSBench-Hard (-12.1) — both whole-repo, long-horizon tasks.

**Sizes, measured, all reconciled:**

```
DeepSeek original   166.89 GB = 156.03 main (45 shards) + 10.86 MTP (shards 46/47/48)
Unsloth Q8_K_XL     161.86 GB = 147.17 MXFP4 experts + 14.54 BF16 dense + 0.15 F32/I32
                                (measured by tensor-type histogram; matches the 162 GB file)
```

Unsloth = original main model with FP8 dense widened to BF16 (+6 GB) and MTP dropped (-11 GB).
Experts are **untouched native MXFP4** in both.

Residency: 162 GB against 64 GB of fleet VRAM = **~40%**, vs GLM-5.2's ~26%. Miss rate dominates
our whole budget, so that is a large move in the right term.

**Downloaded:** `/home/kmbandy/models/DeepSeek-V4-Flash-0731-Q8/` on mad-lab-main, 151 GB.

**Caveat for later:** DeepSeek recommend budgeting **384K output tokens** for high/max reasoning
effort. At ~1 tok/s that is not reachable; low effort may be the only viable mode and its quality
cost is unpublished.

---

## 5. DSPARK — the state, and what I got wrong

DSpark is the lever that would create the prefetch HORIZON. Our 2026-07-10 prefetch attempt
regressed DS4 1.629 -> 1.420 tok/s (-12.8%) purely for lack of lead time: predicting within the
current token gives 1-2 layers, under 10 ms, which cannot hide a ~5 ms cold read.

**Runtime EXISTS** — the upstream sync brought `84075273c` (#25173). **Conversion DOES NOT:**
`conversion/deepseek.py` counts and discards every `mtp.*` tensor
("Skipping %d DeepSeek-V4 MTP tensor(s) for conversion v0"). So the loader waits for tensors the
converter throws away, and **every DS4-Flash GGUF in circulation has zero MTP tensors**.

`dflash.cpp:68` reads `markov_w1.weight` behind `if (markov_meta)` — a missing head **fails
SILENTLY into plain DFlash**. No error, only an absence.

### 5.1 The pattern was already in the file and I missed it

```python
class DeepseekV2Model:   skip_mtp = True
class DeepseekV32Model:  skip_mtp = False    # <- V3.2 ALREADY ships MTP inside the main GGUF
                         block_count = num_hidden_layers + num_nextn_predict_layers
                         add_nextn_predict_layers(...)
class DeepseekV4Model:   separate class, _skipped_mtp_tensors counter instead
```

kmbandy asked for one whole-model conversion with DSpark included, like the original. I read
"whole" as "generic", anchored on a closed PR's separate-draft architecture, wrote
"do not re-encode the main model, use Unsloth's target unchanged" INTO the spec — the exact thing
he had already rejected — and had it built. **The V3.2 precedent was 400 lines above the skip I
kept quoting.**

Artifact from that mistake: `/home/kmbandy/models/DeepSeek-V4-Flash-DSpark-src/out/dspark-draft-f16.gguf`,
10.9 GB, correct but the wrong shape. **Throwaway.**

### 5.2 What IS salvageable and is uncommitted on mad-lab-main

```
M common/arg.cpp            --spec-draft-conf-min, env LLAMA_ARG_SPEC_DRAFT_CONF_MIN
M common/common.h           conf_min = 0.9f      <- KEEP. The reference PR had 0.0 = gate OFF
M common/speculative.cpp    uses conf_min (HEAD wrongly gated on p_min — a real upstream bug),
                            logs actual runtime value at WARN
M conversion/dspark.py      DSparkDraftMixin  (generic; the NAME MAPPING transfers unchanged)
M conversion/deepseek.py    thin subclass
M gguf-py/gguf/tensor_mapping.py   real mtp.2.* prefixes (the old model.markov_head.* never fired)
M gguf-py/gguf/constants.py        DFLASH tensor list expanded
```

The tensor mapping is correct regardless of which file the tensors land in:

```
mtp.2.markov_head.markov_w1.weight -> markov_w1   {256, 129280} = {dspark_markov_rank, n_vocab}
mtp.2.markov_head.markov_w2.weight -> markov_w2
mtp.2.confidence_head.proj.weight  -> conf_proj   {4352} = n_embd + rank
```

### 5.3 Reference: PR ggml-org/llama.cpp#25683

Fetched as remote `yaniss`, branch `dspark-dsv4`, head `c2e51866`, base `c71854292` (an ancestor
of our HEAD — rebases cleanly). **Closed for PROCESS reasons**: a bot flagged "PR template not
respected", "3 open PRs from a new contributor", "large PR needs prior discussion"; the author
closed it six hours later. **No human reviewed it.** It builds a SEPARATE draft model — we are
not doing that — but two of its insights are hard-won and worth keeping: the
`llama_memory_seq_rm` stale-K/V fix (non-causal decoder attending to the previous block's noise
tokens) and reading shards from disk rather than trusting `index.json`.

It also carries a ~630-line `dflash.cpp` path putting the graph on `llm_graph_context_dsv4_mla`.
**Open question:** whether that is needed at all when the MTP blocks live INSIDE the main model
rather than as a standalone draft. Determine before porting.

---

## 6. NEXT, IN ORDER

1. **Answer §1** (BF16 vs Q8_0), then dispatch the whole-model conversion. Spec is written at
   `docs/dev/2026-07-31-design-dspark-conversion.md` and is current except for the FP8
   requirement, which §1 supersedes.
2. **Fix dispatch non-determinism** (§2.4) — precondition for any tight quality gate.
3. **Measure DSpark's actual horizon** once conversion lands. Only then is prefetch worth building.
4. **Prefetch policy.** MoE-Infinity's EAMC (L×E activation matrix per request, cosine-matched
   against ~120 historical ones) is the best-evidenced candidate — but it predicts only
   layer L+1, which is our failed configuration. It fills a horizon; it cannot create one.
5. **kmbandy's transition-map idea** (see §7) — the one candidate that creates horizon without a
   draft model.

**Assessed and CLOSED this session** (read primaries; do not re-litigate):
- **FlashMoE** (arXiv 2506.04667) — NO. "All experts reside in GPU memory." Inverse of our
  problem. Also NVLink 300 GB/s + NVSHMEM + CUDA 12.8, and it fixes comms-at-68%-of-runtime while
  our dispatch issue cost is 0.059 ms/layer.
- **arXiv 2605.11537** (predictive prefetch + expert replication) — NO. One-BATCH lookahead
  (= our dead configuration at batch size 1), CPU-RAM-only tier with 256 GB against our 15, single
  A100, SwitchTransformer-scale models, and it reports "90-95% of baseline performance" — a
  5-10% quality loss, worse than the deferral we just shelved.
- **MoE-Infinity** — policy only, after DSpark. Also: degrades past 2^15 tokens as KV eats the
  expert cache (relevant to DS4's 384K reasoning traces), needs 5-50 requests to re-adapt after a
  task shift, and does not support multi-node.

**The 2026-07-30 catalogue that adopted these three was written from a second-hand survey and got
two of the three materially wrong. Do not plan against it.**

---

## 7. kmbandy's idea, parked but promising

A **persistent expert-transition map**: not a frequency hot-set but a TRANSITION model — observe
that expert 392 -> 642 -> 252 recurs, match where you are in a known trajectory, then run ahead
of your current position. That is how it buys horizon without a draft model, and it is the only
candidate on the table that does.

Relevant prior art: **Markov / correlation prefetchers** from CPU cache literature — same
structure (observe sequences, build transition tables, predict N ahead), decades of work on the
confidence/depth tradeoff and table sizing. Likely more applicable than the MoE papers.

Also: **DS4 ships `dspark_markov_rank: 256`** — `markov_w1` is `{256, n_vocab}` read by
`GET_ROWS`, i.e. a low-rank learned transition table over TOKENS. Structurally the same idea one
domain over, shipped inside the model.

Two cautions from our own data: hot sets are strongly domain-specific (Rome's top-N covers only
12-19% of CODE routing, Jaccard 15%), so the map must be KEYED, not averaged. And an adaptive map
is hard to A/B because every run mutates it — **build a freeze flag before the first line of
policy**.

---

## 8. OPERATIONAL

**Fleet:** clean. Both boxes at `ae1330a87`, ports free, all four board claims released. 2026's
tree is clean; mad-lab-main carries the §5.2 uncommitted work plus another session's DSWS spike
under `aiter-integration/` — **do not touch that**.

**Disk (main):** 216 GB free. A 156 GB source download plus a ~162 GB output does NOT fit
alongside the 151 GB DS4 GGUF and 149 GB GLM shard. **Plan the space before downloading.**

**GLM-5.2** stays on NVMe for now. Parking it on `/mnt/hdd` (231 GB free) shelves it — that
spinner is ~150 MB/s against the SN750's 2.1 GB/s, so it is storage, not a runnable location. A
separate session is migrating CachyOS to the SATA SSD to free the NVMe.

**Two incidents, both mine:**
- A rejected `hf download` **survived its own rejection** because I launched it with
  `setsid nohup ... &` over ssh — the ssh returned instantly and the process outlived the tool
  call. It ran 45 minutes, pulled 71 GB of a 1.52 TB repo, and OOM-killed kmbandy's browser.
  **Never background a remote job that way.** Foreground with a timeout, or `run_in_background`
  so the harness holds a killable handle.
- Health-checking the fleet services by **hardcoded PID** (855466 / 3025042) reported a false
  outage after a legitimate systemd restart at 16:19/16:20. **Check the unit and the port, not
  the PID.** Also seen: the embedder was OOM-killed at 14:28 and auto-recovered in 5 s — memory
  pressure on the 15 GB box is real when builds and services share it.

**Harnesses** (`/home/kmbandy/.claude/jobs/87d16c2e/tmp/`, permission rules in
`~/.claude/settings.json`): `stage7.sh` (4-GPU, `DEFER_K` on the SPINE — deferral lives in the
dispatcher inside libllama, not the workers), `defer_sweep.sh` (K sweep, interleaved controls),
`nll_sweep.sh` + `nll_probe.py` (teacher-forced decode NLL; workers start ONCE and only the spine
restarts per arm, so slot pools stay warm and cache state is not a confound), `qd_sample.py`.

---

## 9. THINGS THAT COST TIME TODAY, SO THEY DO NOT AGAIN

- **Commit before dispatching a writer.** `find -newermt` showed nothing touched, so I committed —
  grok began writing in the 2-minute gap and `675d45806` captured its headers without the .cpp
  definitions. An mtime check is evidence about the past, not a lock on the future.
- **Validate an API with one cheap request before committing a sweep to it.** The first NLL sweep
  failed on all four arms because I guessed the server's JSON shape. The contract is in
  `tools/server/server-task.cpp:292-330` — the key is `top_logprobs`, not `probs`, and `logprob`
  is ALREADY a log. Cost four spine loads.
- **fish is the shell on mad-lab-main.** No heredocs, `$last_pid` not `$!`, and `ssh` lands in
  `$HOME` — use `git -C` or an absolute path. I lost a whole sampler's data writing `echo $!` on a
  fish host, and forgot the `cd` four separate times.
- **`grep -c` returns exit 1 on zero**, which breaks `&&` chains.
- **A substring match on `00048` also matches `of-00048`** in every shard filename.
- **Read the primary source.** Every question that stalled today — the survey's mischaracterised
  papers, the server JSON contract, the `skip_mtp` precedent, the F8_E4M3 assert — was settled by
  one fetch or one grep, after I had already spent messages reasoning in circles.
