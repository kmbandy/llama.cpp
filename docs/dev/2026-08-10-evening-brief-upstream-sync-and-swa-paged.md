# Morning brief — upstream sync landed, and SWA models never get paged KV

**Written** 2026-08-10 evening, by claude__main, for the next session.
**Read order:** this file → the "Open work" section → `2026-08-09-predictor-handoff.md`
(still current for the §4 predictor track, untouched by tonight).

---

## 1. State

Both machines are on branch **`sync/upstream-2026-08-10` @ `cdc486d29`**, identical.

| | mad-lab-main | mad-lab-2026 |
|---|---|---|
| branch | `sync/upstream-2026-08-10` @ `cdc486d29` | same |
| working tree | 5 untracked `segk_*.dis` (deliberate) | clean |
| build dir rebuilt | `build-hip` (HIP, gfx1201+gfx1030) | `build-army-cachy` (CUDA sm_61 + Vulkan) |
| binary version | `11456 (3637b3f37)` | `11458 (cdc486d29)` |
| `test-wp-expert-worker` | all tests passed | all tests passed |
| live services | router :8090 untouched | embedder :8082 + router :8093 untouched |

`master` is still at **`51c70e079`** on main (the pre-merge tip) and **is pushed to origin**.
The sync branch is **NOT pushed anywhere** and master has NOT been moved onto it. That is a
deliberate hold — see "Open work".

**main does not need a rebuild** despite being 2 commits ahead of its binaries: the delta
`3637b3f37..cdc486d29` is two docs plus `ggml-vulkan.cpp`, and `build-hip` has
`GGML_VULKAN:BOOL=OFF` and no `libggml-vulkan.so`. It never compiles that file.

### Recovery paths (nothing was deleted)
- `backup/pre-upstream-sync-2026-08-10` — main's pre-merge tip `51c70e079`
- `~/llama-cpp-2026-dirty-backup-2026-08-10.tar.gz` — all 39 dirty entries from 2026
- `~/llama-2026-strays-2026-08-10/` — 2026's untracked files, **moved not deleted**
- `pre-sync-2026-08-10/` inside each box's `build*/bin/` — the previous shared libraries

---

## 2. What landed: 123 upstream commits

Brings in **Muse Glimmer** (#26841 — 50-layer ViT vision encoder, mtmd, and a
`MuseGlimmerAssistant` DFlash drafter), Granite-Switch, Qwen3-TTS, MTP for
Nemotron / Qwen3-Next / GLM-4.7-Flash, the MiniMax-M3 memory rework, DSv4 fixes.
We now support every upstream architecture.

51 files collided, 12 conflicted, 24 hunks. No `git checkout --theirs`. Four resolutions were
judgment calls — all documented in the merge commit `3637b3f37`, read it before touching any of:

- **`n_outputs_max`** — upstream's new `common_speculative_get_output_limits()` returns exactly
  the `n_parallel*(1+n_max)` budget we measured as insufficient on 08-03, and its new *per-seq*
  cap has its own abort that one DFlash block draft exceeds. Both held at `n_batch`.
- **`k_idx`** — upstream deleted it; nothing in the fork used it; removal adopted, turbo tensors kept.
- **`ggml-backend`** — our coalesced xdev copy path now uses upstream's growable inputs array.
- **router vs upstream's new `server_lru_sched`** — it is wired into 11 call sites in
  cleanly-merged code, so it could not be declined. Eviction now goes through it, and our
  `placement.pinned` + `stopping_models` guards were reinstated **inside** its `pick_victim()`
  (search `FORK GUARDS`). `inflight` and upstream's `req_count` were the same counter and are
  unified on `req_count`.

`cdc486d29` fixes a real merge fallout: upstream changed `VK_CHECK` from `(err,msg)` to
`(err,msg,dev)` and our fork-local `ggml_backend_vk_wp_event_wait()` still passed two. It only
surfaced on 2026 because main builds Vulkan OFF and cannot even syntax-check that file.

---

## 3. THE FINDING — SWA models never enter the paged KV path

**Measured, not inferred.** `MAD_PAGEDATTN_PROBE=1` over a 3389-token Muse prefill produced
**zero** probe lines, and only two plain `llama_kv_cache` instances were created — no
`llama_kv_cache_paged`.

Cause is structural, `src/llama-model.cpp` ~3158:

```cpp
if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
    res = new llama_memory_hybrid_iswa(...);   // ← Muse, gpt-oss
} else {
    ... llama_kv_cache_paged ...                // ← Qwen, Ornith, MusaCoder, LFM2.5
}
```

with the in-tree comment *"SWA hybrids are still TBD (would need a paged-iswa variant)"*.
Muse sets `swa_type = STANDARD` (`src/models/muse-glimmer.cpp:12`; 39/52 sliding layers,
window 2048). So it is excluded **by construction** from the WMMA paged prefill tile kernel
that carries Qwen3.6-27B to ~1015 t/s PP.

### Blast radius across the fleet

| model | arch | paged/tiered KV |
|---|---|---|
| Muse-Glimmer-30B | `muse-glimmer` | **inert** (measured) |
| gpt-oss-20b | `gpt-oss` | **inert** (`src/models/openai-moe.cpp:8` sets `swa_type = STANDARD`) |
| Qwen3.6-27B | `qwen35` | works |
| MusaCoder-27B | `qwen35` | works |
| Ornith-1.0-35B | `qwen35moe` | works |
| LFM2.5-8B | `lfm2moe` | works |

Both affected presets set `kv-tier-paged-blocks = true` plus a `kv-tiered` split; gpt-oss also
asks for 262144 ctx with `kv-tiered = 100,0,0`. **All of it is doing nothing**, and the server
still prints a reassuring, false line:

```
tiered KV (paged): total ctx=131072 … paged cache handles tier movement
```

That log line is the part that costs days. It is the same failure family as the
`mamba_three_phase` episode: a run that succeeds, produces plausible numbers, and silently
describes a configuration nobody asked for.

### Measured on the stock iSWA + FA path (spec OFF, 3389-token prompt)
```
prefill  460.5 t/s   (incremental chunks 556–564 t/s)
decode    19.75 t/s
```
vs kmbandy's spec-ON observation of 160–300 t/s prefill. **That delta is unexplained and is the
cheapest open experiment** — see Open work #4.

---

## 4. Traps found tonight (all cost real time; do not re-pay)

1. **`--no-mmap` is mandatory for Muse.** Launching without it hangs the loader indefinitely —
   100% *user* CPU inside the `use_mmap` branch of `load_all_data`
   (`src/llama-model-loader.cpp:1635`), stuck on `output.weight`, `read_bytes` flat, 10 min+ no
   progress. Diagnosed by measuring progress (read_bytes delta + log-line delta + utime/stime),
   not by picking a timeout.

2. **The router's `[*global*]` block cascades into every model.** Replicating only a
   `[model]` section reproduces the wrong config. Global sets:
   `jinja`, `no-mmap`, `no-warmup`, `flash-attn on`, `n-gpu-layers 999`.
   Missing `flash-attn on` would have invalidated the probe result even if it had loaded.

3. **ptrace is blocked (`yama=1`) and `perf` is not installed** on mad-lab-main. Stacks are not
   available; measure progress via `/proc/PID/io`, `/proc/PID/stat` utime/stime, and log deltas.

4. **Q6_K is NOT off the WMMA path** — exonerated from source. `mmq-config-rdna4.cuh` gives
   Q6_K and Q8_0 identical treatment: 12 `CASE` entries each, same nthreads/occupancy/I/J
   geometry; only the LDS unpack layout differs. `amd_wmma_available()` is true for RDNA4 with
   no per-type exclusion. Do not re-raise the quant as the prefill cause.

5. **Retracted mid-session:** the `n_head_kv` 2-vs-4 paged-tile theory. The 2026-07-11
   root-cause (fast-math UB on `-INFINITY`) superseded it and explicitly called that
   discriminator a testing artifact. I re-derived a dead hypothesis from the older note.

---

## 5. Open work, ranked

**1. Make the lie stop (small, do first).** Warn when tiered/paged KV is requested but the arch
falls back to iSWA, and suppress the "paged cache handles tier movement" line in that case.
Contained, and it is the bit that prevents the next multi-day confusion.

**2. Clean the two presets** so `muse-glimmer-30b` and `gpt-oss-20b` stop claiming capability
they do not receive.

**3. Build paged-iswa — the real feature.** Scope in §6.

**4. Drafter prefill A/B (cheapest, high information).** 460 t/s spec-OFF vs 160–300 spec-ON.
My arithmetic said the drafter should cost 10–15% (5 layers ≈ 10% of target work; tap export
~130 KB/token). That estimate looks wrong. One restart with the `spec-*` lines restored,
same prompt, settles it. **If Muse cannot have the paged path, the drafter may be its only
available prefill lever.**

**5. Decide the branch endgame.** `master` has not moved and the sync branch is unpushed.
Nothing has run a serve session or a bench on the merged tree yet. Recommend: land #4 and a
short serve soak first, then FF master and push.

---

## 6. paged-iswa — what building it actually requires

Anchors verified tonight.

**The key simplification:** `ggml/src/ggml-cuda/mt_pagedattn.cu` has **zero** window awareness
(0 hits for `n_swa` / `sliding_window`) — but it walks a **block table you hand it**. If an SWA
layer's block table contains only the in-window blocks, the existing kernel computes the right
answer *unmodified*. That moves the feature out of the RDNA4 WMMA kernel and into cache
bookkeeping, which is CPU-side and unit-testable without a GPU.

**⚠ This simplification is UNVERIFIED and is the first thing to prototype.** Specifically:
does the kernel's `context_lens` and mask math stay correct when logical block indices are
non-contiguous? If not, item 1 below grows a kernel change and the cost roughly doubles.

1. **Windowed block lifetime in `llama_kv_cache_paged`** — the real work. No SWA concept today;
   nothing recycles blocks leaving `[pos - n_swa, pos]`. (Note: every `swa` grep hit in that
   file is `swap_block` — false positives.) Per-sequence window-bounded allocation + recycling,
   and the block table exposed to the op must contain only in-window blocks. Fully CPU-testable.

2. **A paged-iswa container.** `llama_kv_cache_iswa` holds `std::unique_ptr<llama_kv_cache>`
   (hard-typed), while `llama_kv_cache_paged` derives from `llama_memory_i` — different bases,
   no drop-in swap. Either abstract the sub-cache type or add
   `llama_kv_cache_paged_iswa` holding two paged caches (base = full ctx over global layers,
   swa = window-sized over sliding layers), reusing iswa's existing filter split.
   **This container-vs-abstraction choice is a real fork in the road — brainstorm before coding.**

3. **Graph dispatch for the iSWA path.** `build_attn_inp_kv` selects paged via `dynamic_cast`
   at `src/llama-graph.cpp:3164`. `build_attn_inp_kv_iswa` has **no** paged branch (0 matches).
   Needs that branch plus per-layer selection (SWA layer → windowed sub-cache, global layer →
   full sub-cache), and `build_attn` honoring `is_paged` on the iswa input class.

4. **The creation gate** at `src/llama-model.cpp:~3158` — route `swa_type != NONE` to the new
   container when `kv_tier_paged_blocks` is set, instead of `hybrid_iswa`.

5. **Tier-system integration** — `make_tier_view()`, `state_write`/`state_read`,
   `mt_restore_tag_slot` for the new container. Easy to forget, and forgetting it silently
   reproduces exactly the bug in §3.

6. **Tests + the oracle.** `tests/test-paged-lifecycle.cpp` and `tests/test-paged-semantic.cpp`
   already exist (both touched in tonight's sync) and are the natural home for window-recycling
   tests. Gate = paged-iswa output matching stock iSWA on the same prompt.

**Shape:** multi-day. Items 1–2 are the bulk, 3–4 plumbing, 5 the one that gets dropped.
**Payoff:** Muse and gpt-oss get the WMMA paged prefill path (worth ~2× on Qwen) plus working
tiered/SSD/semantic KV.

---

## 7. NOT verified — do not treat any of this as proven

- **No serve session, no bench, nothing on silicon** for the merged tree beyond one 3389-token
  Muse prefill and the two unit-test runs.
- **Muse Glimmer's own correctness is unexercised** — one prefill + 16 tokens, output never
  inspected for coherence. No PPL, no A/B against a reference.
- **`ggml-vulkan.cpp` has never been syntax-checked on main** (no Vulkan headers). It compiled
  on 2026 after `cdc486d29`; that is the only evidence it is correct.
- **The DFlash drafter was NOT loaded** in tonight's measurement (spec off). The 460 t/s number
  is spec-OFF and is not comparable to kmbandy's spec-ON figures.
- **2026's `build-hip` was not rebuilt** — kmbandy chose `build-army-cachy`. That dir still
  holds pre-sync objects.
- **The block-table simplification in §6 is a hypothesis**, not a finding.

---

## 8. Housekeeping

- Board claim `065b55ec` (gpu:R9700) was released with results; test entry closed.
- Both probe servers stopped by PID; R9700 back to 3% VRAM. No live service was ever disturbed —
  the move-aside protocol held, helped by the soname bumping 11325 → 11456.
- Scratch logs: `muse-probe.log` (the mmap hang), `muse-probe2.log` (the good run) under this
  session's scratchpad. They are ephemeral; the numbers that matter are in §3.
