# Weight pager / GLM-5.2 — session brief, 2026-07-28 evening

Successor to `2026-07-28-weight-pager-session-brief.md` (morning). That brief's §1-§10 were
measured on Laguna and DS4-Flash and are **superseded for planning purposes** by §11 (added this
evening) and by this document. Laguna is retired as a decision vehicle.

---

## 0. Short version

**GLM-5.2 runs.** 254 GB / 750B params / 57,825 expert pages, paged from NVMe on one 32 GB card,
producing correct output. PPL 2.0412 (4 chunks) and 2.7266 (2 chunks), coherent text, exit 0.
That is the first time the whole stack — arch, loader, catalog, pool, transport — has executed on
a real frontier-scale MoE.

Four agents shipped code into one tree today (Codex, Kimi K3, GLM-5.2, me). Everything compiles
and links with HIP, all five unit suites pass, **three commits are on master and everything after
that is uncommitted.**

The single biggest lever found today is **size-class slots**: the pool wastes **39%** of its VRAM
on GLM because sub-pages are strongly non-uniform. Fixing it is worth **1.63x effective pool** and
the fix is written but not yet validated on hardware.

---

## 1. What shipped

### 1.1 Committed (3 commits on master)

```
75b268294  wp: multi-device paging, resident expert blocks, and pipeline layer bands
1e02add94  docs: target-model geometry and the per-token bill (brief §11)
d0381803a  docs: 2026-07-28 weight-pager session brief   (morning, pre-existing)
```

### 1.2 Uncommitted, in the tree, built and unit-tested

- **Size-class pre-carve** (GLM-5.2, `0e1016c6`). `PageCatalog::page_size_histogram()`;
  `PoolAllocator::carve_size_classes_()` pre-carves the arena per class at init instead of
  on-demand; `alloc_slot_size_class_` never carves when `precarved_`; auto-sizer uses the
  histogram average instead of `max_page_size()`. New `tests/test-wp-size-class-slots.cpp`
  (7 tests, all pass), now registered in `tests/CMakeLists.txt` (I added that; GLM omitted it).
- **Pipeline layer bands** (Kimi K3, `cf17d650`). `--pipeline-layers FIRST-LAST`; ownership goes
  through ONE predicate consulted by `create_tensor()` so all archs are covered without per-arch
  edits; `hparams.n_layer` stays global and absolute indices are preserved; KV allocates only for
  owned layers; bands that are empty/discontinuous/role-inconsistent refuse to start. Arch support
  deliberately limited to the deepseek2-graph family (covers GLM-5.2); others throw.
  New `tools/wp-stage-split` emits a loadable per-stage GGUF reusing the loader's own ownership
  predicate, so splitter and loader cannot disagree.
- **Multi-device paging** (Codex, `76ca3c6e`). `wp::WeightPagerSet` — one complete `WeightPager`
  per GPU buffer type, routed at `find_page()`, leaving all 123 `pool_`/`transport_` call sites
  untouched. `--weight-paging-device-layers "ROCm0:0-37;ROCm1:38-74"`.
- **Resident expert blocks.** `--weight-paging-resident-experts <off|BLOCKS>`. Whole-block only;
  no auto-fill by design.
- **Backend-generic prefetch readback.** Cross-layer prefetch was HIP-gated since `3e02da766`;
  its only backend dependency was a D2H readback of two small tensors, now
  `ggml_backend_tensor_get` with the HIP/CUDA stream path as a fallback. **Dispatches on
  `src_t->buffer` at RUNTIME, not on compile-time defines** — build-army has CUDA and Vulkan
  defined simultaneously, so a `#if` branch would run `cudaMemcpy` against Vulkan tensors.
- **Server metrics** now sum across pagers instead of reporting `primary()`.

---

## 2. GLM-5.2: confirmed geometry and the first run

### 2.1 Geometry (from the GGUF, not inferred)

```
general.architecture   glm-dsa        <- matches LLM_ARCH_GLM_DSA in our tree
block_count            79             (78 layers + 1 nextn/MTP)
leading_dense_block_count 3           -> blocks 0-2 dense, 3-77 MoE = 75 MoE layers
expert_count           256   expert_used_count 8   expert_shared_count 1
embedding_length       6144  expert_feed_forward_length 2048
expert_gating_func     2 (sigmoid)  expert_weights_scale 2.5
attention.kv_lora_rank 512   head_count_kv 1        (MLA -> ~80 KB/token KV)
nextn_predict_layers   1
1809 tensors, 7 shards, all 79 blocks carry their own indexer (NOT the shared layout)
```

Attention is DeepSeek-style MLA (`attn_q_a/q_b`, `kv_a_mqa`, `k_b`, `v_b`), which is exactly what
`llama_model_deepseek2::graph` expects — the graph `glm-dsa` inherits.

**Two knowingly-taken shortcuts.** DSA sparse attention is **not implemented** — indexer tensors
load and are never read, so attention runs dense. Exactly equivalent at ctx <= 2048; divergent and
without the memory saving beyond that. And the **MTP head is loaded but unused** (`glm-dsa.cpp`:
"NextN/MTP tensors (preserved but unused)").

### 2.2 First run (established)

Config: one GPU (R9700 / ROCm0), `WP_RESIDENT_DENSE=1`, 2500 slots, ctx 2048/512, no draft model.

```
generation   16 tokens, coherent English, 0.339 tok/s
PPL          2.0412 +/- 0.11263 (4 chunks) ; 2.7266 (2 chunks)
NVMe read    1.68-1.85 GB/s sustained
pager        57,825 pages, 2500 slots x 6,684,672 B (15,937.5 MiB)
```

The completion was fluent but unrelated to the prompt. **PPL settled that as a prompting/template
issue, not a correctness one** — a broken graph or wrong weights gives NaN or unbounded values,
not four chunks clustered 2.0-2.7. Use `llama-server` + `/completion` with a template, or accept
drift on raw completions.

**Deterministic baseline for all future A/Bs** — both control arms reproduced to the digit:

```
PPL 2.7266 | page_ins 114432 | evictions 111936 | io_gb_read 465.769
```

### 2.3 Driver gotchas

- `llama-cli` in this build is **conversation-only**; `-no-cnv` is rejected and `< /dev/null`
  leaves it spinning on empty prompts, loading the model and generating nothing.
- `llama-completion` does **not** accept `--weight-paging` (flag is scoped to CLI/server/perplexity).
- Use **`llama-server` on port 8099** (8090 is the live router) or `llama-perplexity`.

---

## 3. THE headline finding: slot-stride waste

The pool uses **uniform slots sized by the largest page**. GLM's UD quant gives `ffn_down` far
more bits than gate/up, so sub-pages are strongly non-uniform. Measured across all 225 expert
tensors:

```
  size MiB    pages   % pages   payload GiB   roles
    6.375      1024      1.8%          6.4    down
    5.156       256      0.4%          1.3    down
    4.594     18688     32.0%         83.8    down, gate, up
    3.938       512      0.9%          2.0    gate, up
    3.469     37888     64.9%        128.3    gate, up
                                     ------
                            58,368 pages, 221.8 GiB payload
```

**97% of pages are just two sizes.** The 6.375 MiB stride we pay for everything covers 1.8%.

```
uniform @6.375 MiB : 363.4 GiB of slots for 221.8 GiB payload -> 39% WASTED, pool factor 0.610
3 classes (6.375 / 4.594 / 3.469, each absorbing the size below):
                     222.5 GiB -> 0.3% waste -> 1.63x effective pool, zero extra VRAM
```

Concretely: tonight's 15.9 GiB pool held only ~9.7 GiB of actual weight.

**Classes cannot be keyed on tensor role** — `ffn_down` spans three sizes and gate/up span three.
They must come from the measured per-tensor sub-page size.

### 3.1 The crash, and the fix

`WP_SIZE_CLASS_SLOTS=1` **aborted** (exit 134) on GLM:
`"no unpinned size-class slot can fit ... allocated_slots=4096, high_water=..."`.

Root cause: `alloc_slot_size_class_` carved slots on demand, first-come, bumping `high_water_`,
with no splitting and no coalescing. 65% of demand is small pages, so the arena filled with small
slots; a later large page had no free slot of adequate class AND no *used* slot with
`slot_class >= requested_class` to evict. **The allocator cannot discover the class mix at
runtime — it must be computed from metadata up front.**

GLM-5.2 built the fix (§1.2). It kept **upward fallback** (a request may take/evict a larger-class
slot when its own is exhausted) on the grounds that pre-carving makes it rare and it guarantees
"never abort while the arena has usable space." I agree with that call.

**Status: built, unit-tested, HIP-compiled, and A/B'd on hardware — STILL ABORTS.**

The pre-carve works, but the crash moved rather than disappearing:

```
ARM ctl1  exit=0   PPL 2.7266  page_ins 114432  evictions 111936  io 465.769
ARM test  exit=134 Aborted (core dumped)
ARM ctl2  exit=0   PPL 2.7266  page_ins 114432  evictions 111936  io 465.769

no unpinned size-class slot can fit 6684672 B (class=6684672,
allocated_slots=4101, budget=16711680000, high_water=16705880064, precarved=1)
```

`precarved=1` and 4101 slots (vs 2500 uniform) prove the pre-carve ran and DID enlarge the pool.
But the failure is now "no **unpinned** slot" on the LARGEST class -- 625 such warnings before the
abort. The slots exist; they are all pinned.

**THE DESIGN ERROR IS MINE, in the spec.** I specified per-class slot counts proportional to
total PAGE COUNTS. The largest class is 1.8% of pages so it got ~74 slots. But the binding
constraint is the **maximum concurrently PINNED set per class**, not the demand share. Under
batched prefill (`n_seq=4`, 2048 tokens) the union of experts touched in one layer approaches all
256, each needing its `ffn_down` page -- far more than 74 large slots -- and pinned slots cannot
be evicted.

**Correct sizing criterion: peak concurrent pins per class, floored by the per-layer expert
union** (top_k for decode; the batch-wide union for prefill, which is much larger). Proportional-
to-demand is right for CACHE value and wrong for CORRECTNESS. The floor must come first, then
proportional allocation of whatever remains.

This was invisible to unit tests because it only appears under batched prefill. Any fix needs a
test that pins a whole layer's worth of one class simultaneously.

Next step is NOT more allocator work until the floor is derived: for GLM, count the distinct
sub-page sizes a single layer can demand at once and size each class to at least that.

---

## 4. Corrections and retractions — all mine

1. **Resident-set estimate 8.1 GiB vs actual 14.6 GiB.** Caused an OOM cascade (§8.1). I wrote
   the VRAM arithmetic as the rule requires but *estimated* the resident set when the exact number
   was in a GGUF I had already opened twice. **Measure it; it takes 20 seconds.**
2. **"Two drives = ~2x" is false for single-stream decode.** With a contiguous split the stages
   run strictly sequentially — main computes while 2026 idles, then the reverse — so token time is
   the SUM, not the max, and the drives never overlap. 2026 cannot know its experts until it
   receives the activation. Real gain is **~10%, from capacity**. The KG entry `5d3e38fa` claiming
   "two independent I/O paths working simultaneously" assumed an overlap the data dependency
   forbids. The 2x needs multiple tokens in flight.
3. **MTP speculation is not GLM's top lever.** On a paged MoE, batched verification touches the
   **union** of experts across drafted positions, and that cost is structural — it does not shrink
   with acceptance. I ranked it #1 from dense-model intuition and had to withdraw it.
4. **"Multi-device adds capacity, not bandwidth" was unjustified.** The drive is at ~38% duty
   cycle and effective QD1; it is *under-requested*, not saturated. Two independent pagers may be a
   concurrency lever. Corrected in the spec, and the binding design consequence (no shared
   io_uring ring / worker pool / submission lock) was added.
5. **Uniform expert frequency does NOT kill the eviction lever.** Uniform *marginal frequency* and
   short-range *temporal locality* are independent; measured next-token retention was ~33% against
   a 3.1% chance rate. I conflated them and withdrew it.
6. **Largest-block-first ordering was justified with the wrong quantity.** The pool stride is set
   by the largest *sub-page*, not the block total. Fixed to rank on `max_page_bytes`.
7. **Trusted fetch summaries over sources, twice.** BuddyMoE's prefetch mechanism, then the
   kimi-k3-mlx model list (I missed the REAP variants entirely). Both times the actual content
   contradicted the summary.
8. **Declared pi "not running" from three negative process greps** while two agents were live —
   `pi`'s cmdline is bare `pi`, so every pattern missed. Session-file mtimes showed writes seconds
   old and I read them as ambiguous. This created a duplicate agent (§8.2).
9. **Reported "paging hard" while `llama-cli` generated nothing**, spinning in conversation mode.

---

## 5. Kimi K3 — settled tonight

**Do not pursue before GLM is finished.** Findings:

- Arch: 93 layers, **896 experts, top-16**, 2 shared, hidden 7168, moe_intermediate 3072, experts
  in a **3584 latent** (Stable LatentMoE) so 33.0M params each. Cross-check: 92 x 896 x 33.0M =
  2.72T vs stated 2.78T. **Natively multimodal** (separate vision tower). No MTP layer.
- **`llama.cpp` has no `kimi_k3` architecture** — confirmed independently. Multiple GGUFs already
  exist (unsloth, AtomicChat IQ1_S ~540 GB, GrEarl Q2_K) staged for support that has not landed.
  A PR was reportedly opened; merge status unknown.
- **UD-Q1 is ~620 GB.** ~7.5 MB/expert, **~11 GB/token**.
- **REAP variants are MLX-only and worse for us.** REAP80 is 350 GB / 179 of 896 experts but at
  **mxfp4 4.25 bpw (bit-exact, not requantized)** -> ~21.3 MB/expert -> **~31 GB/token**, roughly
  **2.8x the I/O of UD-Q1 despite the smaller file**. Top-k stays 16 either way, so per-token cost
  is expert *size*, not expert *count*. Pruning helps residency, nowhere near enough.
- **Quality concern, unresolved:** REAP80 keeps 20% of experts. Our DS4 coverage curve
  (99.93% @ 75% kept, 96.44% @ 50%, 81.41% @ 25%) suggests ~20-25% of routing decisions would land
  on a pruned expert. Near-uniform routing is the *worst* case for pruning. The REAP table
  publishes tok/s and **no quality metric at all**.
- **The artifact that would actually be good does not exist:** REAP *and* low-bit — 179 experts at
  ~1.8 bpw is ~148 GB, ~11 GB/token, ~47% residency.
- **We cannot convert either model ourselves.** `convert_hf_to_gguf.py` has no `glm_moe_dsa` and
  no `kimi_k3` entry. Good low-bit quants also need an imatrix, which needs a working runtime.
- 2026's disk caps its Kimi shard well below the residency-optimal split; the planned CachyOS
  migration (OS to the SATA SSD) frees the full 250 GB SN750 and removes that constraint — at
  which point **main** becomes the binding constraint instead.

---

## 6. Open work, ranked

1. **Size-class pin floor.** The A/B ran and still aborts (§3.1): classes were sized to demand
   share, but the constraint is peak concurrent PINS per class. Derive the floor from the
   per-layer expert union (batch-wide for prefill), allocate that first, then distribute the
   remainder proportionally. Worth 1.63x effective pool once correct. Needs a test that pins a
   whole layer's worth of one class at once -- unit tests could not see this.
2. **Raise the slot count.** Tonight ran 2500 slots (~4% residency) deliberately for safety.
   With 14.6 GiB resident + compute on a 32 GB card there is room for meaningfully more,
   especially once size classes land. Re-derive from the measured resident set, not an estimate.
3. **Measure GLM's expert concentration** via `WP_CAPTURE_ROUTING`. Currently *unmeasured* for
   every model we care about; it sets the value of every caching and pruning decision, and would
   confirm or refute the near-uniform assumption everything now rests on.
4. **`ensure_batch_calls: 0` in the first run.** The demand-batching path worth +24% never fired.
   Find out why — eligibility is derived as `3*|active|` and GLM is top-8, so it should qualify.
5. **Pipeline parallelism Phase 2** (the protocol). Phase 1 + stage splitter are built and
   unit-tested; nothing has been load-tested. Test as a **localhost loopback on main first**.
6. **Correct the resident/island split** using the real 14.6 GiB. The 6900 XT cannot hold it
   (16 GB) alongside KV and compute — that is what OOM'd the box. Either move part to the R9700 or
   accept single-device.
7. **Per-device `--weight-paging-slots`.** The explicit override applies the same count to every
   pager; only the auto path budgets per device.
8. **`GpuTransport::init` 4-vs-3 signature** still breaks CPU-only builds. One line.

---

## 7. Method notes worth keeping

- **`page_ins` at temperature 0 is a precise instrument.** Both controls reproduced to the digit.
  Same tokens, same routing, so any delta is real. Prefer it to tok/s, which is noisy on this box.
- **Validate paging by PERPLEXITY, never by token equality.** Greedy output legitimately diverges
  from a reference within ~20 tokens because accumulation order differs.
- **A running PPL over few chunks is not an absolute score.** It is valid as a correctness signal
  and for arm-vs-arm only.
- **`cmd &` and `cmd | head && next` both take exit status from the wrong thing.** Both produced a
  false success tonight — one hid a failed build, one ran a stale binary and printed a passing test.
- **Never judge an agent by process greps.** `pi` runs as bare `pi`. Check the session file mtime
  under `~/.pi/agent/sessions/<cwd>/`.
- **On a mixed-backend box, never pick a memory path with `#if`.** build-army has CUDA and Vulkan
  defined together. Ask the tensor (`src_t->buffer`).
- **Agents editing a shared dirty tree can silently revert a decision** and nothing flags it
  (§8.3). Commit at decision boundaries.

---

## 8. Incidents

### 8.1 OOM cascade (21:14)

First GLM run placed the resident set on the 6900 XT based on my 8.1 GiB estimate; the actual set
is 14.6 GiB against a 16 GB card, so the allocator spilled to host and the OOM killer took
**llama-cli, Hyprland, Discord, llama-router, and the systemd user session.** Hyprland survived
as a second instance; the router stayed down until explicitly restarted at ~22:40 (start only,
no `daemon-reload`). Box healthy. Later runs carry a **RAM watchdog** that kills the job if
MemAvailable drops below 3 GB.

### 8.2 Duplicate agents on one tree (20:07-20:11)

Believing the first pi handoff had not started, I launched a second. Both ran on the same tree for
~4 minutes. Killed the newer one; verified no duplicated declarations and no conflict markers.
No damage found.

### 8.3 A decision was silently reverted

The `auto`/`SIZE` removal from `--weight-paging-resident-experts` (your call) was reverted by an
agent that wrote back a stale copy of `wp-router.cpp` — parser, header, model branch, CLI help
**and the tests**, as a set. The build passed and the tests passed *because they had been reverted
alongside the code*. Caught only by reading the built binary's `--help`. Restored.

---

## 9. Exact state at session end

- **Tree**: 3 commits on master (`75b268294` tip), everything since is **uncommitted** —
  size-class fix, the CMake test registration, and the brief.
- **Build**: `build-hip` green, `EXIT=0`, 0 errors, carrying all of today's work.
- **Tests passing**: `test-wp-resident-experts`, `test-wp-multidevice-partition`,
  `test-pipeline-band`, `test-wp-stage-split`, `test-wp-size-class-slots`.
- **Model**: `~/models/GLM-5.2/` — 7 shards, 237 GiB, `+C` directory, **0 compressed extents**,
  each sha256-verified at copy. `/home` is `compress=zstd:1` so any future model needs the same
  treatment or O_DIRECT falls back to read-and-decompress.
- **Services**: `llama-router.service` **active**, 10 models unloaded, 0 VRAM. Now running
  tonight's binary.
- **Board**: claim held on `mad-lab-main` — **release it in the morning** if not continuing.
- **Disk**: ~320 GB free on main.

### The A/B result (completed)

Ran to completion: **abort, exit 134**, controls clean and identical. Full analysis in §3.1.
Logs: `/tmp/sizeclass_ab2.log`, per-arm `/tmp/sc_{ctl1,test,ctl2}.log`.

`WP_SIZE_CLASS_SLOTS` stays **off by default**, which is safe — the uniform path is untouched and
both controls prove it. The 1.63x remains available but needs the pin-floor fix first.
