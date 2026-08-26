# DS4 10 t/s decode / 80 t/s prefill — knobs and remaining levers

Written 2026-08-20 after the env-knob lever list was exhausted on the
width-sliced 1408:320:320 rig. This is the continuation map, not a session
diary.

Targets: **10–15 t/s decode**, **80 t/s prefill**.
Quality gate: teacher-forced NLL only, baseline **2.177888**. Never text/PPL
as a gate; inspect generated text enough to reject degenerate “fast-wrong”.

Two different kinds of work live in this file:

1. **Sweeps** — one-variable A/B of knobs that already exist. Router cycle vs
   worker cycle is load-bearing (s0 live slots are 3450; a bare harness
   restart silently drops them to 3400).
2. **Build levers** — code that does not exist yet, or exists and is off
   because it hung. These are not knobs.

The old multiplicative map (3.3 × read-hide × turbo4 ≈ 10) is dead: turbo4 is
ON and did not 2× this protocol; harvest / async-issue / concurrent-dispatch
are closed.

---

## Standing snapshot (2026-08-20, both boxes HEAD `2b2aafd9a`)

| | this protocol | record / bank |
|---|---|---|
| decode 512 (essay, temp 0, ignore_eos) | **2.765 t/s**, draft 238/339, mean len **1.88**, acc/pos (0.735, 0.140, 0, …) | 3.147 this protocol; **~7.1 t/s** on 08-03 index-sliced + mean len **4.94** |
| prefill cache-bust | **29.66 t/s** / 1663 tok | 25–30 band; 27.06 / 2424 and 27.91 / 16316 with p1+p2 |
| NLL | turbo4 2.137 PASS vs 2.178 | 2.177888 f16 baseline |

Decode wall: 43 sequential layer waits at `max(s0,s1,s2)`. Width 1408:320:320
makes **s0 the long pole** (~9–10 ms/layer: ~7.8 ms compute + ~3.8 ms read).
Perfect prefetch only hides the read → **~3 t/s compute ceiling**. Dispatch
protocol cannot move that (harvest relocated blocked→before_await; wall did
not move).

Prefill wall: expert union per 2048-ubatch is ~98.6% deterministic. Bandwidth
ceiling ~170 t/s, so 80 is under the drive **if** L+1 is resident when L+1
demand arrives. p3 is that path; it hung (see §2.1).

### Where knobs live

| surface | what it controls | cycle |
|---|---|---|
| `~/.config/llama-router/router-fleet-main.ini` `[ds4-flash]` | ubatch, batch, KV type, `--spec-draft-*`, expert-dispatch endpoints | unload ds4-flash / router restart |
| `~/.config/systemd/user/llama-router.service` + `.d/*.conf` | spine env (`WP_HINT_*`, `WP_SPLIT_FRAME`, `WP_DISPATCH_DEDUP_*`, `WP_HIP_GRAPHS`) | `systemctl --user daemon-reload && restart llama-router` after unload |
| `~/ds4-runs/stackd-worker.env` | worker env (`OFFSET_SORT`, `LAYER_AHEAD`, `SPEC_*`, slots, HIP graphs) | **worker** recycle via stackd; keep `SLOTS_S0=3450` |
| `docs/dev/harness-2026-08-05-ds4_full.sh` | maps short names → `WP_*`; defaults that are **not** the live router INI | workers-only relaunch |

Live serving is the **router INI + drop-ins + stackd-worker.env**, not a bare
harness run. The harness `SPEC_CONF=0` default is **not** what `[ds4-flash]`
is passing (`--spec-draft-conf-min 0.4`). Do not mix them.

### Measurement protocol (do not invent a new one)

- One variable per arm.
- Decode: cycle as required, warmup 32, then ~512-token `/v1/completions`
  temp 0 `ignore_eos` B-tree essay. Report `timings.predicted_per_second`,
  draft accepted/generated, acc/pos, unique-word check.
- Prefill: cache-bust (`cache_prompt=false` / unique prompt). Report prompt
  eval t/s and `prompt_n`. Split PREFILL/DECODE; refuse a response with no
  timings object.
- NLL gate (`nll_gate.py`, 40 tok × 3 passages) when numerics can move.
- Confirm live `/proc/<pid>/environ` and cmdline against the file. stackd
  `set -a` never unsets; empty `PARTIAL_DTYPE=` / `DSPARK_THREADS=` must stay.
- Rebuild **both** `build-hip` (main) and `build-army-cachy` (2026) after
  worker-code changes. 1070/RX480 do not run `build-hip`.

---

## 1. Sweep variables

Two different `conf` knobs and two different `k` knobs. Mixing them is how
prior sessions compared two variables against a one-variable baseline.

| spoken name | actual knob | what it is |
|---|---|---|
| **router2 conf** | `WP_HINT_ROUTER2_CONF` | softmax floor on **prefetch predictor** hints |
| **draft conf** | `--spec-draft-conf-min` | DSpark’s **trained confidence gate** on draft length |
| **router2 K** | `WP_HINT_ROUTER2_K` | layers of lookahead the predictor scores |
| **router2 M** | `WP_HINT_ROUTER2` | top-M experts per predicted layer |
| **draft n-max** | `--spec-draft-n-max` | max draft tokens per verify |

### 1.1 Open — sweep these (they can still move 10 / 80)

Priority is “does this attack the actual wall”, not “is it easy”.

#### A. DSpark draft gates (decode; couples to §2.2)

Live INI: `--spec-draft-n-max 7 --spec-draft-conf-min 0.4`. `p-min` unset
(DSpark default). Acc/pos on the 512-tok rerun: **(0.735, 0.140, 0, …)**,
mean len **1.88**. Banked 08-03 on a different topology: mean len **4.94**,
~7.1 t/s. Restoring 4.94 is ~2.6× fewer 43-layer verify passes → ~7 t/s
without faster workers.

The harness record (08-05) is `SPEC_CONF=0` = verify the full block, matching
SGLang static mode. The **router INI is at 0.4**. That discrepancy has not
been A/B’d on this width-sliced protocol. Treat it as the first decode sweep.

| knob | live | candidate ladder | cycle | notes |
|---|---|---|---|---|
| `--spec-draft-conf-min` (`SPEC_CONF`) | **0.4** (INI) | **0**, 0.2, 0.4, 0.6 | router / unload | **Do this first.** 0 is the harness record. 0.9/0.99 were uncalibrated and are not comparable. Metric: acc/pos + mean len, then t/s. |
| `--spec-draft-n-max` (`SPEC_NMAX`) | 7 | 3, 4, 7 | router / unload | Only after conf-min: n-max=7 is wasted if pos 2+ is already 0. Do not use “decode block count” as the metric (label is assigned by width — see `docs/dev/2026-08-05-measurement-discipline.md` rule 1). |
| `--spec-draft-p-min` (`SPEC_PMIN`) | unset (0) | leave unset unless conf-min=0 still cliffs | router | Generic draft-token probability gate. **Do not stack on DSpark** unless the trained conf gate is proven off. The old “p-min must never be 0” rule was for generic draft models, not DSpark. |
| `--spec-draft-n-min` | default | only if n-max sweep wants a floor | router | Low value. |

Print the spine’s `speculative.cpp` startup WARN. That line is the authority
on what actually ran.

#### B. Router2 / reuse (decode prefetch volume; K not re-swept at conf 0.75)

Live drop-in `zz-prefetch.conf`. Conf sweep this session: **0.75 won**
(2.72 t/s) vs 0.40 (2.13), 0.90 (2.23), 0.60 (2.16). K=7 won the **earlier**
sweep at conf 0.40 (3.147 t/s). K has **not** been re-swept at 0.75.
K=15/36 are untested (mailbox drops 40–55% of snapshots at those depths).

These attack **page-in wait**, not s0’s 7.8 ms compute. Ceiling if they go
perfectly: ~3 t/s. Still worth a cheap K-at-0.75 pass so we do not leave a
known hole, then stop.

| knob | live | candidate ladder | cycle | notes |
|---|---|---|---|---|
| `WP_HINT_ROUTER2_K` | 7 | 3, 7, 15 | router | One-variable on top of standing conf 0.75. |
| `WP_HINT_ROUTER2` (top-M) | 6 | 4, 6, 8 | router | 6 matches `n_expert_used`. 4 was a **quality** reject when applied to real routing (`WP_N_EXPERT_USED`); hint M=4 is a different question. |
| `WP_HINT_ROUTER2_PAGES` | 16 /decode | 8, 16, 32 | router | Cap is per-decode, not per-flush. 16+32 reuse = 48 < 64 queue. Raising past 32 collides with reuse. |
| `WP_HINT_REUSE_PAGES` | 32 | 16, 32, 48 | router | Uncapped reuse was 2.387 t/s / +162k dropped. 32 recovered to 2.619. |
| `WP_HINT_ROUTER2_CONF` | **0.75** | do not re-sweep 0.40/0.60/0.90 | router | Already priced this protocol. |
| `WP_HINT_ROUTER2_CONF_STEP` | 0.0 | leave | — | Depth-raised floor. Off on purpose with first-snapshot mailbox. |
| `WP_HINT_ROUTER2_DEPTH_DECAY` | off | leave unless K>7 | router | Halves top-M with depth. |

#### C. Prefill shape (couples to §2.1)

Live: `ubatch-size=2048`, `batch-size=2048`. Expert set is re-swept **per
ubatch**. 2048 is standing because 4096 puts the 6900XT at 96% VRAM and is
mutually exclusive with long context. Worker `WP_IO_PREALLOC_TOKENS` **must
move with** ubatch or the arm reallocates during serving.

| knob | live | candidate ladder | cycle | notes |
|---|---|---|---|---|
| `--ubatch-size` / `--batch-size` | 2048 / 2048 | **hold 2048** for p3 work; 1024 only as a control that doubles sweeps | INI | 4096 needs `-b 4096` (second variable) and kills long-ctx headroom. Gain depends on `n_prompt mod n_ubatch`. |
| `WP_IO_PREALLOC_TOKENS` | tied to ubatch | always = ubatch | workers | Not an independent lever. |
| `LAYER_AHEAD` / `WP_PREFILL_LAYER_AHEAD` | **0** | 0 until §2.1 ships; then 1 | workers | Hung the 50-tok warmup (full L+1 catalog on the dispatch thread). |
| `WP_PREFILL_LAYER_AHEAD_WIDTH` | 32 | 32, 64 | workers | n_tokens below this is treated as decode and skipped. Must stay ≥ DSpark verify width (`1+n-max`). |
| `PREFILL_GATE` | 1 | hold 1 | workers | Pauses **guess** spec during prefill. p3 bypasses it by design (certain union). Do not turn off as a substitute for p3. |
| `SPEC_CHUNK` | 8 | 4, 8, 16 **after** p3 is chunked | workers | Pages per spec submit. Decode: 8 cut vbusy 94%→83%. p3 must reuse this so L+1 is a stream, not one 256-page submit. |
| `OFFSET_SORT` | 1 | hold | workers | Bit-exact. Keep. Neutral on the 25–30 band; needed so p3’s catalog is sequential NVMe. |

#### D. Slot / spec occupancy (only if prefetch is the arm)

| knob | live | candidate ladder | cycle | notes |
|---|---|---|---|---|
| `SLOTS_S0` | **3450** | do not “try 3400” | workers | Harness default 3400. Silent drop is a second variable. |
| `SPEC_MAX_SLOTS_MAIN` / `_2026` | 891 / 980 | hold unless occupancy is the question | workers | 26% of pool. Uncapped spec churned 1.7× the pool and decode decayed 3.0→1.8. |
| `WP_EXPERT_SPEC_HOST` | 0 | hold 0 | workers | Host landing tier was dropped; guesses live in VRAM under the cap. |
| `SPEC_PAGEIN` + `PREFETCH_HINT` | both 1 | never one without the other | both | Hint-only is the “spine offered what?” control. Spec without hints is a no-op. |

### 1.2 Standing — already priced, do not retune without new evidence

Adopted, bit-exact or quality-gated. Changing one of these is a new A/B, not
background noise.

| knob | live | why it stays |
|---|---|---|
| `cache-type-k/v` | turbo4 | NLL 2.137 PASS. 08-03 1.96× was a different (index-sliced / possibly-corrupt) protocol; this protocol got 2.912 not 2×. |
| `n_expert_used` / `WP_N_EXPERT_USED` | 6 | 6→4: NLL +8.4% and prompt-echo collapse. 3.25 t/s invalid. |
| `WP_SPLIT_FRAME` + `WP_DISPATCH_DEDUP_ACTIVATIONS` | 1 / 1 | p1. Prefill 27 t/s band. Decode n_tokens≤32 skipped. |
| `WP_SLICE_SKIP_SCAN` + `WP_SLICE_ENCODE_ONCE` + `WP_ISSUE_WIDEST_FIRST` | 1 | Bit-exact, decode-neutral, prefill-encode. |
| `WP_HIP_GRAPHS` | 1 spine + workers | Spine: replay holds, 22.6% fallback. Workers: graph-update net-negative on s0 ns_submit; left on by request, not a sweep. |
| `WP_SPEC_CONST_WIDTH` | unset | Net-negative. Do not set. |
| `STRIPEPAR` / `COALESCE` / `RESIDENT_FIRST` / `DOUBLE_BUFFER` | 1 | On. Read-scheduling / overlap. Bit-exact for resident-first (assignment-order fold). |
| `WP_EXPERT_COMPUTE_CHUNKS` | 4 (binary default) | Prefill wait −5.6%. Gated on n_pagein>0. |
| `WP_EXPERT_READ_STRIPES` / `STRIPE_MAX_PAGEINS` | 4 / 4 | Decode wait −9.1%; ungated striping cost prefill +10.7%. |
| `VKSPLIT` | 1048576 | RX480 256 MB BAR. Without it ~95% of the slot pool sits in GTT. |
| `KEEPALIVE` | 100 (2026 workers 200 µs) | 100 vs 200: −15.3% dispatch wait. |
| `DSPARK_OMP` | 8 | 4/8/16/24 sweep; 8 adopted. 24 is ~2× slower (oversubscription). |
| `FILL_HOST` / `STAGING` | on in serve path | Fill-host so evict skips D2H; staging 32 in the historical launch comment. |
| `CTX` | 512000 INI / 8192 harness | Decode independent of allocated ctx (measured). |

### 1.3 Do not recycle (closed, vetoed, or known-negative)

| item | why |
|---|---|
| `WP_DISPATCH_HARVEST` / `WP_ASYNC_ISSUE` | Closed 08-18. Rearranges how the spine waits, never when the layer completes. Async: rare multi-second await tails, zero upside at decode. |
| `WP_DEFER_K` | Approximation; PPL/quality loss on another model; user veto. |
| `WP_PREDICT_AHEAD` / learned predictor / MTP-as-prefetch | User veto. MTP on paged DS4 was 0.43× with 2.7× bytes. |
| `GATHER_MIN=8` (or =2 at n=1) | Broke determinism and gave no speedup after chunking/striping. |
| Assignment bias / decode-prefer-1070 as a “free” flag | Width-sliced: every expert is 1408:320:320. You cannot drop 480 from the layer without dropping its slice. |
| 6900XT as expert worker | −26% measured (contention with the spine). |
| `PARTIAL_DTYPE=f16` | −10% decode. Wire is latency-bound at n=1, not bytes-bound. |
| `WP_EXPERT_BATCH_MOE` | Crashes the sliced rig (wrong-shaped partials). |
| Host-victim tier as a **decode** lever | Decode −6%. Main 3 GB swapped DSpark. |
| Copying 2026’s 0–84 blobs onto main | Explicitly “ask first”. Required for unsliced decode-local. |
| `WP_SPINE_PROFILE` | Splits HIP graphs to n_nodes=1. 2.041 t/s. Never leave on for a throughput run. |

---

## 2. Remaining build levers

These are the things that can still reach 10 / 80. They are **not** env
knobs. Each one names the wall it attacks, the coupling to §1, and the
falsifier.

Recommended order: **2.1 → 2.2 → 2.3**. 10 t/s on this width-sliced rig
likely needs 2.2 and 2.3 together. 80 t/s is 2.1, then 2.4 only if prompts
span multiple 2048 ubatches.

### 2.1 Prefill: chunk p3 layer-ahead (the 25 → 80 path)

**Wall.** Prefill is linear: ~33.9 ms/tok marginal, ~5 s fixed. The 2048-token
ubatch re-sweeps ~98.6% of 256 experts × 43 layers. s0’s slice is ~9 MiB, so
one sweep is tens of GB of NVMe. Compute of layer L and the sequential read
of layer L+1 can overlap; today they do not.

**What exists.** `submit_prefill_layer_ahead()` in
`tools/wp-expert-worker/wp-expert-worker.cpp`. After `ensure_batch` for L it
calls `spec_pagein_submit` on the **entire** next-layer catalog
(`layer_pages_sorted_`, already (blob, offset) sorted). Bypasses
`PREFILL_GATE` and the 64-deep hint queue. Env `LAYER_AHEAD=1`.

**Why it hung.** 50-tok warmup is n_tokens=50 > width 32, so p3 fired. One
submit of ~256 pages on the **dispatch thread** before compute. That is a
submit-shape bug, not a wrong idea.

**The change.** Do not submit the catalog in one shot. Feed it through the
existing idle-pump / `SPEC_CHUNK=8` machinery: cap in-flight, never block
demand, harvest on the pump. Keep `OFFSET_SORT=1` so the stream is sequential
NVMe. Code stays env-gated; `LAYER_AHEAD` stays 0 until a 32-tok warmup
returns in seconds, not minutes.

**Sweep after it ships.** `LAYER_AHEAD=0/1` (the A/B), then `SPEC_CHUNK`
4/8/16, then `WP_PREFILL_LAYER_AHEAD_WIDTH` only if verify batches are being
misclassified as prefill.

**Falsifier.** Cache-bust prefill at ~1.5–3k tok. Win = prompt eval t/s
clearing the 25–30 band toward 80, with NLL bit-identical (landing time
cannot change numerics). Fail = hang, demand-starved ns_read, or t/s still
in-band because compute — not I/O — is already the marginal 33.9 ms.

**If it fails because compute is the floor.** Do not retune prefetch. Go to
§2.4 (layer-major / sparse partials / worker pipeline).

### 2.2 Decode: DSpark accept-length cliff

**Wall.** Each verify pays 43 layer RTTs for `mean_len` output tokens.
Current mean len 1.88 vs banked 4.94 is ~2.6×. Acc/pos dies after position 1.
08-16 hunted this (readout convention, tap ±1, stale-KV cleanup — all
refuted) and stopped after three patches; suspected a driver-level defect.
It has **not** been re-opened on this width-sliced + turbo4 + conf-min=0.4
stack.

**What to do before writing code.** Sweep §1.1.A (`conf-min=0` vs 0.4) with
acc/pos as the primary metric. If mean len moves toward 4–5, the cliff was
the gate, not the head. If it stays (0.73, 0.14, 0, …) at conf-min=0, resume
the 08-16 instrumentation (`WP_DSPARK_DEBUG`, per-slot conf dumps) rather
than paging work.

**Falsifier.** Acc/pos histogram vs the 4.94 bank on a matched essay. t/s is
secondary: longer accepts that pull extra expert sets can lose on a paged
model (MTP taught that). Gate = mean len **and** decode t/s both up, text
still a real essay, NLL hold.

**If the head is really this weak on this topology.** Stop spending pager
time on a 3 t/s compute floor. Go to §2.3.

### 2.3 Decode: s0 service time (compute and/or slice balance)

**Wall.** s0 ~9–10 ms/layer × 43 ≈ 390 ms/tok. Breakdown (08-18 reqlog):
ns_compute 7.76 ms (matmul + Q8 dequant, **not** launch-bound — HIP graphs
made ns_submit worse), ns_read 3.79 ms (1.71 page-in/req), ns_h2d 1.5 ms.
s1 ~4–5 ms, s2 ~5–6 ms, idle waiting on s0. Hardware floor for the resident
GEMV is ~0.04 ms/layer; 7.76 ms is ~200× that (tiny batch-1 GEMVs).

Two independent attacks:

**2.3.a Skinny-M grouped GEMV on s0.** Fuse the 6 experts × 3 sisters into
one (or few) kernels at decode/verify M=1–8. `compute_batch_grouped` already
batches at graph level; the kernels are still tiny MMVQ. This is a kernel
project (ml8/DSWS is the **prefill** compute-bound kernel and the wrong
shape). Bit-exact is the gate (NLL), then s0 ns_compute.

**2.3.b Rebalance 1408:320:320.** s0 does 4.4× the width of s1/s2 despite
being the fastest card, so it arrives last every layer. Moving width onto
1070/480 makes them slower and they share one ~3 GB/s NVMe. Equilibrium is
probably modest (1.2–1.4×), not 3.6×. This is a **reshard**, not an env
knob. Do not confuse with “decode-local R9700”: width-sliced decode
**cannot** drop 2026 without dropping 320+320 of every expert.

**Falsifier (2.3.a).** s0 `WP_REQ_LOG` ns_compute on n_tokens=1 and n=2/4
verify, NLL hold, then 512-tok t/s. **Falsifier (2.3.b).** Per-worker
arrival times equalize; layer wall = the new max; t/s follows. If 2026 NVMe
becomes the new long pole, revert.

**Arithmetic honesty.** Hide s0’s 3.8 ms read perfectly → ~3 t/s. Rebalance
to ~6.5 ms tied → ~3.5–4.5 t/s. Grouped GEMV that cuts 7.8 ms → ~2 ms, plus
mean len ~5 from §2.2, is the actual 10 t/s stack. Neither half alone is
enough.

### 2.4 Prefill follow-ons (only if 2.1 leaves compute as the floor)

Not the first move. Listed so they are not rediscovered as “new knobs”.

| lever | wall | note |
|---|---|---|
| Layer-major prompt eval | llama.cpp is ubatch-major: all 43 layers per chunk, then the next ubatch re-sweeps | 43 waves instead of `n_ubatch×43`. Pays when `n_prompt > 2048`. |
| Worker request pipeline | serve loop is recv → compute → send; next demand I/O cannot start under compute/send | Complements p3 (p3 is next-layer I/O; this is next-request I/O). |
| Sparse PIPE partials | worker D2H/sends full f32 `[n_tokens,4096]` including zeros (16–32 MB at ub 1024/2048) | Decode is not bytes-bound; this is a **prefill** wire/CPU play. f16 partials already −10% at decode. |
| `ggml_get_rows_back` → O(nnz) scatter | CUDA/Vulkan scan is O(ncols×nrows_dst×nrows_grad); idx is unique | Prefill kernel, NLL-gated. |
| Shape-stable gather (bucket/pad n_sel) | prefill graphs cache only dense+coalesce | Lets worker HIP/CUDA graphs arm at prefill width. |

### 2.5 Topology (ask before touching)

Unsliced / index-sliced decode-local on the R9700 (whole experts, 2026 for
prefill only) is how the ~7 t/s bank existed. It requires copying or
re-slicing the 0–84 blobs (~49 GB on 2026 NVMe). Explicitly not to be done
without asking. Putting experts on the 6900XT is not this, and is already
closed (−26%).

---

## 3. What a continuation week looks like

1. **Ship §2.1** (chunk p3). Keep `LAYER_AHEAD=0` until warmup is sane, then
   `=1`, cache-bust prefill, NLL only if anything but landing time moved.
2. **Sweep §1.1.A** (draft `conf-min` 0 vs 0.4) on the standing decode
   protocol. Decide whether §2.2 is a gate bug or a head/driver bug.
3. **Cheap §1.1.B** (router2 K at conf 0.75). Stop after one ladder; it
   cannot reach 10.
4. **Only then §2.3** (s0 GEMV and/or rebalance), with live `WP_REQ_LOG` on
   s0 so the 08-13 “RX480 long pole” story is not mixed with the 08-18 “s0
   long pole” measurement.

Do not start a new env-knob list. The remaining work is 2.1–2.3 plus the
small sweeps that couple to them.
