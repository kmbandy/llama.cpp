# Weight-paging decode-speed work — continuation (morning pickup)

**Branch:** `feat/wp-vnext` (edit/build-check on **mad-lab-2026** at `/home/kmbandy/GitHub/llama.cpp`).
Pushed to `origin`; **mad-lab-main** is on `feat/dsws-phaseb-conversion`, FF-merged to the same tip
`78358b158`. All GPU validation is on mad-lab-main's **R9700 (ROCm0, gfx1201, 32 GB)**; the 6900XT
is ROCm1. Remote shell is **fish** → always `ssh mad-lab-main bash -s <<'EOF'`.

## TL;DR of the session

Took **paged-resident dense-27B decode from 7.0 → 21.0 t/s = native speed**, with **PPL exactly
matching native** (5.4623). Three gated, default-off, committed+pushed commits. Then found that
the last/biggest fix (`WP_BATCH_EVAL_CB`) **faults on MoE** — that's the open item for the morning.

## UPDATE — morning session 2026-07-07 (Step 1 DONE; Step 2 scoped)

**Step 1 complete — dense win locked in and shipped default-on:**
- `2a221b9ef` **dense-only guard**: `batch_safe()` now also requires `!catalog_.has_experts()`. On MoE
  → per-op path (no batching), so no fault.
- `aa1130789` **flip `WP_BATCH_EVAL_CB` default ON** (off only when `WP_BATCH_EVAL_CB=0`). Self-gated:
  only engages under `--weight-paging` + size-class + fully-resident + dense.
- Both pushed to `origin/feat/wp-vnext`; cherry-picked + built + validated on mad-lab-main
  (`5919648e2`, `656ae5a14`).
- **Validated:** dense 27B PPL = **5.4623** with explicit `=1` AND with the flag **unset** (default takes
  effect). MoE guard proof: guarded `=1` is **byte-identical** to `=0` on gpt-oss (same NaN, same stats,
  no fault) — guard makes `=1` a no-op on MoE.

**Step 2 evidence gathered (LFM2.5-8B-A1B, a small 8B/1B-active MoE, valid finite PPL — a far better
test case than gpt-oss):**
- LFM native (no paging) PPL = **27.2266**.
- LFM paged **per-op** (BATCH=0, slots=750, fully resident) = **27.0938** — matches native within
  reduction-order noise (same 0.13-ish gap dense shows: 5.4543 vs 5.4623). **Per-op MoE paging is CORRECT.**
- `routing_ptrs_discarded_unconsumed: 6` appears on BOTH LFM (correct output) and gpt-oss (NaN) →
  **that counter is a benign accounting artifact, NOT the corruption source.** The gpt-oss NaN is
  gpt-oss/MXFP4-specific, off the critical path.
- **Batching fault is GENERAL:** temporarily removed the guard, rebuilt, ran LFM with batching →
  `Memory access fault … address 0xdc000 … Page not present` (same near-null class as gpt-oss's
  0x11f000). So the batching-on-MoE fault reproduces on a clean valid-PPL MoE. GPU recovered cleanly.
  (Experiment reverted; mad-lab-main back to shipped guarded state.)

**Net for Step 2:** the ONLY MoE bug is the batching range/pin lifetime. Per-op is a correct baseline.
We now have a **clean repro + a valid PPL target (27.09)** to validate the redesign against.
**Do the redesign on LFM2.5-8B-A1B** (fast, fully resident at slots=750, valid PPL), not gpt-oss.

## The result (dense Qwen3.6-27B-Q6_K, fully resident, R9700)

| config | decode t/s | PPL (wiki, -c512 --chunks 4) |
|---|---|---|
| native (no paging) | ~native | **5.4623** |
| paged, all flags OFF (per-op sync) | 7.0 | 5.4543 |
| paged + size-class + resident-fadvise | 9.4 | — |
| **paged + WP_BATCH_EVAL_CB=1** | **20.97** | **5.4623** ✅ matches native |

**The 5.4623 vs old 5.4543:** confirmed it's not a regression — native (no pager) is *also* 5.4623.
The old per-op-sync 5.4543 was the anomaly (serialized execution → different GPU reduction order).
Batching makes the paged path converge to native in both speed AND numerics. Determinism proven:
batch-off → 5.4543 (x2), batch-on → 5.4623 (x2), fully reversible.

## Commit stack this session (on `feat/wp-vnext`, all pushed)

- `390a7e18d` feat(wp): **size-class VRAM slots** `WP_SIZE_CLASS_SLOTS` (default off) — packs the
  whole model resident in a size-class arena (vs fixed max-size slots). The enabler: on 32 GB the
  27B (22.8 GB) can't be resident with fixed slots (each = 70 MB), but size-class packs all 863
  pages. Known risk documented: no coalescing → a large *required* page can `alloc_slot → -1`
  (never triggered on the 27B at slots=345).
- `ec48c2316` perf(wp): env-tunable `WP_PREFETCH_DEPTH` / `WP_IOURING_DEPTH` (from before this session).
- `5ee99edcd` diag(wp): **`WP_PROFILE_EVAL`** — RAII host-time profiler inside `weight_pager_eval_cb`.
- `74a5c5aa2` perf(wp): `find_page(const char*)` alloc-free overload + gate the LOW_ADDR diagnostic
  loop. (Measured ~0 effect — the string alloc was NOT the cost; kept as cheap hygiene.)
- `c1bb508e1` diag(wp): split `WP_PROFILE_EVAL` into pre-Step1 / step1-resolve / ensure buckets.
- `b7a33e849` perf(wp): **resident-aware fadvise** — `advise_layer_lookahead` skips the
  `posix_fadvise(WILLNEED)` NVMe readahead when the lookahead window is already resident
  (`page_loaded_`). Was ~35% of decode when resident (readahead warms nothing). **+34% (7.0→9.4).**
- `78358b158` perf(wp): **`WP_BATCH_EVAL_CB`** (default off) — the 3× (9.4→21). See below.

## The big one: WP_BATCH_EVAL_CB (root cause + fix)

**Root cause (verified in `ggml/src/ggml-backend.cpp:1700-1729`, confirmed by two independent
consults — codex + Fable):** when an eval callback is registered, the scheduler abandons the
whole-split async path (line 1701) and runs **per-node**, issuing a full
`ggml_backend_synchronize(split_backend)` after each node-*range* (line 1729). The range only
extends **while the callback returns `false`** (line 1716). `weight_pager_eval_cb` returned `true`
at every op → every one of ~3700 nodes/token became a singleton compute + GPU sync → decode fully
serialized (submit→sync→submit→sync). Native (no callback) submits the whole split in one async
compute. **This per-op sync was the entire ~2-3× gap** — not `ensure()`, not I/O, not the string
work (all measured near-zero when resident). The profiler saw it as ~100 µs/weight-op of "host time"
(the CPU blocking in `ggml_backend_synchronize`, off-book from the callback's own timer).

**The fix:** the return value doesn't gate patching (the scheduler calls `ask=true` per node during
range-building, so `src->data` patching still happens before compute); it *only* controls whether
the scheduler syncs after the node. The pager does nothing on the `ask=false` post-callback, so no
node needs observing. So return `false` when safe → the scheduler batches → sync per-split like
native. Implemented as `eval_cb_op_return()` at the two op-level exits, returning `false` only when:
- `WP_BATCH_EVAL_CB=1` (default 0 → returns `true` → byte-identical old behavior), AND
- `pager->batch_safe()` = `stats_.evictions == 0 && pool_.size_class_slots_enabled()` (no eviction
  ⇒ no slot reuse ⇒ unpinning the prev op's slots before the batched compute can't recycle an
  in-flight slot — MAD-231 pin-lifecycle safe), AND
- `!routing_tls_set` (set inside the `GGML_OP_MUL_MAT_ID` routing branch when it calls
  `ggml_cuda_set_routed_expert_ptrs`), AND
- `sync_fallback_count()` didn't increase this op.
The MAD-230 `ggml_cuda_discard_routed_expert_ptrs()` still runs at the top of every ask=true call.

## OPEN / NEXT MORNING — WP_BATCH_EVAL_CB faults on MoE
> **SUPERSEDED by the 2026-07-07 UPDATE above.** Step 1 (guard + default-on) is DONE and shipped.
> The MoE fault is now confirmed GENERAL (reproduces on clean LFM2.5-8B-A1B), and per-op MoE paging
> is confirmed CORRECT (valid baseline 27.09). What remains below is the still-accurate root-cause
> analysis and the redesign spec (path 2). Ignore the "decide first thing" framing — path 1 is done.

**Validated on gpt-oss-20b-MXFP4 (11.3 GB, resident, slots=1024 → 12.24 GB arena, ~15 GB total VRAM):**
- native PPL = **427** (garbage — gpt-oss is NOT a raw-text LM; wikitext PPL is meaningless for it,
  so it's a poor PPL test model. Pick a real MoE with valid base-LM PPL next.)
- batch0 (no batching): **completes**, but `routing_ptrs_discarded_unconsumed: 6` — a pre-existing
  MAD-230-class routing leak on gpt-oss *independent of batching*.
- **batch1 (batching on): HARD GPU FAULT** — `Memory access fault … addr 0x11f000 … kernel
  mul_mat_q … void const* const*` (the routed-expert pointer array), preceded by `[1]nan,[2]nan`.
  Classic MAD-230 near-null expert-pointer fault; batching triggered it. GPU recovered cleanly
  (no wedge, device enum OK, VRAM back to baseline — a contained fault).

**Why the routing guard was insufficient (root cause of the MoE fault):** the scheduler's
`while(!need)` loop takes the *first* node that returns `true` as the **last** node of the batched
range (`j1` at line 1716-1721), NOT a standalone. So a routing op returning `true` still computes
**inside** a batched range together with the preceding non-routing ops — the expert-pointer TLS /
pinned-expert-slot lifetime is NOT isolated. `routing_tls_set → return true` does not break the
range *before* the routing op.

**Two paths for the morning (decide first thing):**

1. **Lock the dense win safely (small, do first regardless):** add `!catalog_.has_experts()` to the
   batch gate (in `batch_safe()` or alongside it) so `WP_BATCH_EVAL_CB` only ever engages for
   **dense** models. Then it's safe to flip **default-on** for dense (see below) with zero MoE risk;
   MoE always keeps the working per-op path. `catalog_.has_experts()` already exists
   (`wp-page-catalog.h`). This is the minimum needed before any default-on.

2. **MoE batching redesign (the real MoE prize — its own focused effort, likely codex+Fable):** make
   a routing op **break the range *before* it** (so it never computes inside a batched range), and
   keep the active-expert slots pinned across the batched range. This is a genuine correctness
   redesign of the routing/batching interaction, not a quick patch. gpt-oss is a fragile test case
   (pre-existing leak, junk PPL) — validate the redesign on a **real MoE with valid PPL** (candidate:
   `ornith-1.0-35b-Q5_K_M.gguf` = qwen35moe, 23 GB — but that's tight on 32 GB resident: 23 GB
   arena + overhead ≈ 26-28 GB, right at the ceiling, do the VRAM math carefully; or find a smaller
   valid-PPL MoE). Confirm PPL matches native AND `routing_ptrs_discarded_unconsumed == 0` AND no
   fault.

## Default-on decision (pending step 1)

`WP_BATCH_EVAL_CB` is inherently scoped: the eval callback it controls is **only registered when
`--weight-paging` is active** (non-paged inference never hits this code), and it self-gates on
`batch_safe()` (only fires when fully resident + size-class, reverts to per-op the instant anything
evicts). After step 1 (dense-only guard), flipping the default to ON (i.e. on unless
`WP_BATCH_EVAL_CB=0`) is safe and gives the dense win with no flag. Hold the flip until the
dense-only guard is in.

## Validation recipes (copy-paste)

Common env: `LLAMA_WP_TRANSPORT=p2p LLAMA_WP_TRANSPORT_FORCE=1 WP_DENSE_PREFETCH_N=8 WP_SIZE_CLASS_SLOTS=1`
Common args: `--no-mmap --weight-paging --weight-paging-slots <N> --weight-paging-prefetch -ngl 99 --device ROCm0 -c 512`

- **Dense 27B** (`~/models/Qwen3.6-27B-Q6_K.gguf`, `wikitext-2-raw/wiki.test.raw`): slots=**345**
  (~24 GB arena). PPL gate = **5.4623** (== native) with `WP_BATCH_EVAL_CB=1`. Decode via
  llama-server `/completion` `predicted_per_second`.
- **VRAM safety:** size-class arena = `n_slots × max_page_size` allocated up front — read the pager
  init line first (`wp::WeightPager: N pages, S slots x B budget …`) on a tiny slots=64 load to get
  `max_page_size`, then size slots for arena ≥ model, total ≤ 28 GB. gpt-oss max_page = 11.95 MB /
  2760 pages; 27B max_page = 70 MB / 863 pages.
- **Tools:** llama-cli is UNUSABLE (parks interactive, ignores -no-cnv); llama-bench has no
  `--weight-paging`; use **llama-perplexity** (PPL) + **llama-server** (`/completion` decode t/s).
- **llama-server build:** must be rebuilt when `common_params` changes — a stale `libllama-server-impl.so`
  vs fresh `libllama-common` = ABI-skew segfault in the arg parser (hit + fixed this session). Always
  include `llama-server` in the build target set.

## Build commands (mad-lab-main, capped)

```
systemd-run --user --unit=wp-build -p MemoryMax=13000M -p MemoryHigh=11000M -p CPUQuota=600% \
  --working-directory=/home/kmbandy/GitHub/llama.cpp \
  bash -c "cmake --build build-hip -j2 --target llama llama-server llama-perplexity > /tmp/x.log 2>&1"
```
build-hip = multi-arch gfx1201;gfx1030, GGML_HIP_AITER=ON (Triton AOT — reuse stamps, don't rm).

## Env flags added this session (all default-off / behavior-preserving)

`WP_SIZE_CLASS_SLOTS`, `WP_PREFETCH_DEPTH`, `WP_IOURING_DEPTH`, `WP_PROFILE_EVAL` (diag),
`WP_BATCH_EVAL_CB`. Existing knobs still relevant: `WP_DENSE_PREFETCH_N`, `WP_FADVISE_LOOKAHEAD`
(now resident-aware), `LLAMA_WP_TRANSPORT=p2p`, `LLAMA_WP_TRANSPORT_FORCE`, `WP_ASYNC_ENSURE`.

## Cleanup state at logoff

No GPU processes running; R9700 VRAM at ~1.4 GB (desktop baseline). All work committed + pushed;
`feat/wp-vnext` == `origin` == mad-lab-main tip `78358b158`. Only uncommitted = pre-existing WIP
(`examples/pagedattn-*`, `src/llama-graph.cpp`, `examples/CMakeLists.txt`) — do NOT touch.
