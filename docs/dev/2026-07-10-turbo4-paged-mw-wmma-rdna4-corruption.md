# Turbo4 Paged-Attention Garbage on RDNA4 (Qwen3.6-27B / qwen35, hd256) — Findings & Fix Plan

**Date:** 2026-07-10
**Machine:** mad-lab-main, R9700 (gfx1201 / RDNA4), build `build-hip`
**Branch:** `feat/wp-dflash-ds4` @ `d71223efa` (+ uncommitted WIP + an embedder crash-fix, see §9)
**Status:** Root cause **runtime-proven**. Fix **not yet applied.** Two separate issues were worked this session — see §1.

---

## 1. Two separate issues (do not conflate)

| # | Issue | Status |
|---|---|---|
| A | `llama-server-qwen36-27b-swatm-r9700` **crashed (SIGABRT)** on first message after swapping the semantic embedder to LFM2.5-Embedding-350M | **FIXED + verified** (§9) |
| B | Dense Qwen3.6-27B (arch `qwen35`) produces **pure `////` garbage** with turbo4 paged KV | **Root-caused, not yet fixed** (this doc) |

Issue A is done and unrelated to B. **This doc is about Issue B.**

---

## 2. Symptom (Issue B)

- Model: `Qwen3.6-27B-Q8_0.gguf`, `general.architecture = qwen35` (dense hybrid: gated-delta-net recurrent + attention + **dense** FFN). head_dim (key/value_length) = **256**, n_head = 24, **n_head_kv = 4**, n_embd = 5120, 65 blocks.
- Config: the production service config — `--cache-type-k/v turbo4 --flash-attn on --kv-tiered 90,10,0 --kv-tier-paged-blocks` on `--device ROCm0`.
- Output: `reasoning_content` is a solid run of `/` characters, `content` empty, `finish_reason=length`. No crash — the server serves HTTP 200, the tokens are just garbage.
- User observation that cracked it open: the **same** garbage was seen earlier on **MusaCoder-27B** (also `qwen35` dense), while **ornith-1.0-35b** (arch `qwen35moe`) was fine. MusaCoder last worked "a couple days ago."

---

## 3. Root cause (RUNTIME-PROVEN)

**The multi-warp WMMA paged-attention _prefill_ tile kernel corrupts on RDNA4/gfx1201 for head_dim=256 + turbo4.**

- Kernel: `mt_paged_attention_tile_mw_kernel` / `launch_paged_attn_tile_mw` in `ggml/src/ggml-cuda/mt_pagedattn_tile.cu`.
- It scrambles the prompt's K/V **during prefill**; scalar decode then reads a corrupted cache and emits garbage tokens.
- Everything else on the turbo4 paged path is **correct**: the turbo4 **scatter (write)**, the **single-warp** tile kernel, the **split-K / flash decode** kernel, and the scalar prefill/decode fallbacks.

### The A/B isolation matrix (all runtime env toggles, ONE binary, no rebuilds)

Repro command base (vary only the leading env):
```
<ENV> ./build-hip/bin/llama-server \
  --model ~/models/Qwen3.6-27B-Q8_0.gguf --n-gpu-layers 999 --ctx-size 131072 \
  --parallel 1 --cache-type-k turbo4 --cache-type-v turbo4 --flash-attn on \
  --kv-tiered 90,10,0 --kv-tier-ssd-path <dir> --kv-tier-paged-blocks \
  --no-mmap --no-warmup --jinja --host 127.0.0.1 --port 8091 --device ROCm0
```
Prompt: `"What is 17 plus 25? Reply with only the number."`, `max_tokens` ~120.

| Env | Prefill path | Decode path | Result |
|---|---|---|---|
| (none / default) | multi-warp WMMA tile | split-K decode | **GARBAGE** |
| `GGML_PAGED_TILE=0 GGML_PAGED_DECODE=0` | scalar | scalar | **coherent** |
| `GGML_PAGED_DECODE=0` | multi-warp WMMA tile | scalar | **GARBAGE** |
| `GGML_PAGED_TILE=0` | scalar | split-K decode | **coherent** |
| `GGML_PAGED_TILE_MULTIWARP=0` | **single-warp** tile | split-K decode | **coherent** |
| (default) `--cache-type q8_0` instead of turbo4 | multi-warp tile | split-K decode | **coherent** |

**Reads:**
- Row 2 (all scalar coherent) ⇒ the **scatter/write is fine** (scalar attention reads the same scattered cache and is correct). Kills the "post-RoPE stride" scatter hypothesis.
- Row 3 (decode off, tile on ⇒ garbage) ⇒ the **prefill tile** is the corruptor.
- Row 4 (tile off, decode on ⇒ coherent) ⇒ the **decode kernel is fine**.
- Row 5 (single-warp tile ⇒ coherent) ⇒ **only the MULTI-WARP variant is broken**; single-warp WMMA tile is correct.
- Row 6 (q8_0 coherent) ⇒ turbo4-specific in combination with the mw tile.

### Env-var toggles (defined in `ggml/src/ggml-cuda/mt_pagedattn.cu`, ~lines 22–90)
- `GGML_PAGED_TILE` (default 1) — WMMA tile prefill vs scalar prefill.
- `GGML_PAGED_TILE_MULTIWARP` (default 1) — multi-warp cooperative tile vs single-warp tile.
- `GGML_PAGED_DECODE` (default 1) — split-K/flash decode vs scalar decode.
- `GGML_PAGED_FUSED` (default 0) — fuse scatter into the attention kernel.
- `MAD_PAGEDATTN_PROBE=verbose` — dumps dispatch decisions (`avg_q_len`, `wmma`, `total_q`, etc.). Useful for confirming which path fires.

### Dispatch call sites (tip `mt_pagedattn.cu`)
- Tile gate (`HS==128||256` & cache in {F16,TURBO4_0,TURBO3_0} & `avg_q_len>=16`): ~L1683–1697.
- multi-warp vs single-warp select (`mw_on = get_paged_tile_multiwarp_mode()`): ~L1720.
- Split-K/flash decode gate: ~L1754.
- Turbo4 scatter kernel + its contiguous input-stride assumption (`token*n_kv_heads*HEAD_SIZE + kv_head*HEAD_SIZE + d`): ~L352. (Proven **not** the bug — see Row 2 above — but note the assumption for future readers.)

---

## 4. It is a REGRESSION, from the 2026-07-06 upstream sync

Bisected on `feat/wp-dflash-ds4`, repro as §3:
- **GOOD:** `66eec7f54` (2026-07-01) — merge-base with `master`, **pre-sync**. Coherent.
- **BAD:** `676b87554` (2026-07-09). Garbage.
- The **only** commit in `66eec7f54..676b87554` that touches the paged/turbo/KV path is the **07-06 upstream sync merge `06a3da0e6`** ("Merge remote-tracking branch upstream/master into sync/upstream-2026-07-06"). Everything else in range is weight-pager / DS4 / dflash / docs — inert for single-device turbo4-paged.

What the sync changed in the GPU core (`git diff 66eec7f54..676b87554 --stat`):
```
ggml/src/ggml-cuda/mt_pagedattn.cu       | +511
ggml/src/ggml-cuda/ggml-cuda.cu          | +311
ggml/src/ggml-cuda/turbo-quant.cuh       | +142
ggml/src/ggml-cuda/cpy.cu                | +109
ggml/src/ggml-cuda/mt_pagedattn_tile.cu  | +88
ggml/src/ggml-cuda/mt_pagedattn_aiter.cu | (changed)
```
> ⚠️ The premise "nobody touched paged/turbo4" is **false** — the sync pulled substantial changes into these exact files. Easy to miss because it rode in via a merge, not a local edit.

### What has been RULED OUT as the regression cause
- **`9e8310ecb`** ("size multi-warp tile Q_TILES to device shared-mem limit", the only sync change to `mt_pagedattn_tile.cu`): reverted that single file (`git checkout 9e8310ecb^ -- mt_pagedattn_tile.cu`) on the tip, rebuilt → **still garbage.** Not it.
- **`mma.cuh`**: **not changed** by the sync (confirmed via path-limited bisect scope). So the RDNA4 WMMA C-output lane-mapping code is unchanged.
- **Scatter/write**: fine (§3 row 2).
- The `mt_pagedattn_tile.cu` file at tip == its `66eec7f54` version except `9e8310ecb` (only commit to touch it), and reverting `9e8310ecb` doesn't help.

### Therefore
The multi-warp kernel's own file is effectively unchanged, `mma.cuh` is unchanged — so the break comes from a **changed shared dependency the multi-warp path consumes**, most likely:
- **`turbo-quant.cuh` (+142)** — turbo4 dequant. (But the single-warp tile also dequants and is fine, so it must be an mw-specific dequant path/usage, or…)
- a shared device helper in **`mt_pagedattn.cu` (+511)** used only by the cooperative (multi-warp) K/V smem staging.

Per repo docs and the external (GPT-5.6) review, the **RDNA4 WMMA hd256 multi-warp path was never validated on RDNA4** — the historical oracle was GTX1070 CUDA + RX480 Vulkan. So this was a **latent, unvalidated path** that the sync's shared-dep change tipped into corruption.

### Still-open sub-question
Why is **ornith** (`qwen35moe`, hd256, turbo4, paged) **clean** while **Qwen3.6-27B** (`qwen35`, hd256, turbo4, paged) is garbage? Both hit the mw tile. They differ in `n_head_kv` (2 vs **4**), `n_head` (16 vs 24), `n_embd` (2048 vs 5120), block_count (40 vs 65). Leading theory: the cooperative tile grid/smem layout (grid.y indexes `kv_head*N_QBLOCKS + qb_idx`) only trips the bug at `n_kv_heads=4`. **Confirm in the morning** — it will point straight at the buggy indexing.

---

## 4.1 (2026-07-11 update) — full static audit: NO turbo4_0 logic line changed → regression is a second-order compile effect

Went back and diffed **every** ggml-cuda file the sync touched, against the turbo4_0 mw path specifically (the §4 "changed shared dependency" hypothesis was under-verified — it never checked `mt_pagedattn_ops.cuh`). Result: **the turbo4_0 path has no changed logic anywhere.**

| File (sync Δ) | What actually changed | turbo4_0 mw-path impact |
|---|---|---|
| `turbo-quant.cuh` (+142) | adds `TURBO_CENTROIDS_4BIT_N64` + `TURBO_MID_4BIT_N64` `__constant__` tables, `TURBO4_64_OL*_CHANNELS` `__constant__`, `__device__ int g_turbo4_64_ol_use_n64_table`, and `turbo4_64*`/`_ol*` dequant helpers. `turbo4_64_dequant_element` now uses the N64 table. | **turbo4_0 (`turbo4_0_dequant_element`, TURBO_CENTROIDS_4BIT) unchanged.** |
| `mt_pagedattn.cu` (+511) | almost entirely new `mt_scatter_kv_turbo4_64_ol{,8,12}_kernel`; new TURBO4_64_OL* case labels (HS==64 only); one idempotent `ggml_cuda_turbo4_64_ol_sync_n64_table_flag()` call at top of `ggml_cuda_op_paged_attn_mt`. Two turbo4_0 scatter hunks only add explicit `WARP_SIZE` arg to `__shfl_xor_sync`. | scatter shfl change is a **verified no-op** (offsets ≤16, `WARP_SIZE=32`; width-32 vs RDNA default width-64 give identical results since 64-lane segments align). Rest is HS==64-gated → inert for hd256. |
| `mt_pagedattn_ops.cuh` (+172) | **entirely new** `paged_cache_ops<TURBO4_64_OL{,8,12}>` specializations appended after the TURBO4_64 struct. | existing `paged_cache_ops<GGML_TYPE_TURBO4_0,…>` (the loader the hd256 mw kernel calls) **unchanged**. |
| `mt_pagedattn_tile.cu` (+88) | only `9e8310ecb` (Q_TILES smem sizing). | reverting this file → **still garbage** (§4). |
| `mma.cuh` | not in sync range. | unchanged. |
| `vendors/hip.h` (+1) | `#define cublasSgemmBatched hipblasSgemmBatched`. | irrelevant. |

**Conclusion / reframe:** there is no "broken line" to revert on the turbo4_0 path. The corruption is a **second-order compilation effect** — the sync added `__constant__` tables, a `__device__` global, and 3 new `paged_cache_ops` template instantiations into the **same translation units** as the cooperative multi-warp kernel, shifting its constant-bank / register / occupancy / smem allocation. The **multi-warp cooperative WMMA prefill kernel is latently buggy on RDNA4/gfx1201** (a pre-existing race / smem overrun / uninitialized-lane defect on a path the historical oracle — GTX1070 CUDA + RX480 Vulkan — never exercised), and the sync's added constant-memory/occupancy pressure tipped it from "accidentally-correct" into "corrupt."

**Implications for the fix:**
- Do **not** expect a clean single-commit/single-line revert to fix (B). The bug is in the mw kernel's own cooperative logic, not in the sync.
- (A) — routing RDNA4 hd256 → the (proven-correct) single-warp tile — is therefore not a workaround **around** the real bug; the mw kernel is genuinely defective and (A) is the correct durable fix until the cooperative kernel is repaired.
- (B) becomes a **kernel-correctness debugging task** (races/smem/lane-init in `mt_paged_attention_tile_mw_kernel`), not a "find the regressing dependency" task. The ornith-vs-27b `n_kv_heads=4` discriminator (§4 open sub-question) is still the best lead into the faulty cooperative index.
- Cheap corroboration for the "occupancy/constant-pressure tipped a latent bug" theory (do first, low risk): rebuild the **good** commit `66eec7f54` but locally add the sync's new `__constant__` tables + `__device__ global` to `turbo-quant.cuh` (no call sites). If that alone reproduces the garbage, the second-order-compile theory is confirmed and the mw kernel is proven latently buggy.

## 4.2 (2026-07-11) — Codex static-analysis review + corrections + FIX APPLIED

A Codex deep static-analysis pass (no builds/runs) reviewed the §4.1 hypothesis and the mw kernel. It **confirmed the core conclusion** (mw tile prefill is the corruptor; single-warp is correct) and **found no source-level defect** in the mw kernel — no incorrect index, no missing barrier, no OOB, no race. It also corrected several claims in this doc; the corrections are **verified**:

- **Separate translation units.** `ggml/src/ggml-cuda/CMakeLists.txt:105` does `file(GLOB … "*.cu")` — every `.cu` is its own TU. So `mt_pagedattn.cu`'s new `turbo4_64_ol` kernels **cannot** shift the register/occupancy of `mt_paged_attention_tile_mw_kernel` (in `mt_pagedattn_tile.cu`). The §4.1 "same-TU occupancy shift" mechanism is **wrong for the .cu additions**. The **only** surviving second-order vector is the `turbo-quant.cuh` **header** (new `__constant__` tables + a `__device__` global) being `#include`d into the tile TU, perturbing its device-symbol layout / codegen. Static analysis cannot distinguish that from a latent-UB or ROCm-codegen sensitivity.
- **Q8_0 was never in the tile gate.** `mt_pagedattn.cu:1685-1687` admits only `F16 / TURBO4_0 / TURBO3_0` at HS 128/256. So §3 Row 6 ("q8_0 → coherent") is **confounded** — q8_0 fell to a non-tile path; it does **not** prove turbo4-specificity. (The load-bearing Rows 3 & 5 still stand and still prove the mw tile prefill is the corruptor.)
- **RDNA4 is wave32 here** (`common.cuh:77`), so all wave-64 lane-math theories are moot; `mma.cuh` reduces `threadIdx.x` to a wave-local lane (`mma.cuh:202,338`).
- **`n_kv_heads==4` discriminator: REFUTED.** The cooperative address formula (`mt_pagedattn_tile.cu:248`) is identical to the proven single-warp loader (`mt_pagedattn_ops.cuh:131`); `grid.y` is `seq_idx`, not a KV head (`mt_pagedattn_tile.cu:827`). No behavior changes at 4 heads. (So the §4 "still-open sub-question" is closed — not the lead.)
- **The mw code WAS tuned on gfx1201 historically** (`mt_pagedattn_tile.cu:971`) — so "never validated on RDNA4" (§4/§4.1) is too strong; it was exercised, just without a correctness oracle for this model/cache combo.

**Codex's strongest suspect (unproven):** compiler-sensitive resource pressure in the HS=256 mw specialization — 16 Q + 16 output WMMA fragments (`:905,:919`), ping-pong K frags + 2 score accumulators (`:981,:996`), exactly 64 KiB dynamic LDS (`:92,:97`), `__launch_bounds__(192,2)` (`:802`). This explains multi-warp-HS256-turbo4 specificity **without** any incorrect source index.

### FIX APPLIED (tip `d71223efa`, `mt_pagedattn.cu` mw_on select, ~L1720)
Routed RDNA4 HS=256 mw prefill → the proven-correct single-warp WMMA tile:
```cpp
const bool mw_on = get_paged_tile_multiwarp_mode() != 0
                   && !(GGML_CUDA_CC_IS_RDNA4(cc) && HS == 256);
```
**Divergence from Codex (deliberate):** Codex scoped the gate to `CT == GGML_TYPE_TURBO4_0` (only the proven-failing combo). I gate by **HS only** (all cache types at HS=256), because the root cause is resource pressure in the HS=256 specialization *generally*, and F16/TURBO3_0 at HS=256 also hit this kernel with **no** RDNA4 correctness evidence. Asymmetric risk: the broader gate only pessimizes a near-unused path (single-warp is still WMMA; F16 KV at 1M ctx is memory-prohibitive), while the narrow gate risks silent corruption on F16/TURBO3_0. Uses `GGML_CUDA_CC_IS_RDNA4(cc)` (not `amd_wmma_available`, which also matches RDNA3). Env var cannot re-enable the broken path. **RDNA3, HS=128, and all non-RDNA4 hardware are unaffected.**

**Status:** fix applied to working tree (uncommitted). **`build-hip` rebuilt + VERIFIED on the 9700 (2026-07-11):** launched Qwen3.6-27B-Q8_0 with the exact production config (turbo4 K/V, flash-attn, `--kv-tiered 90,10,0 --kv-tier-paged-blocks`, ctx 131072) under **default env** (which previously produced `////`) → **coherent** ("17 + 25 = 42" reasoning, no slashes). Clean A/B: same binary except this one-line gate. **Pending:** `build-army` (mad-lab-2026, CUDA sm_61) rebuild for shipping consistency — no runtime effect there (gate macro is a no-op on Pascal); and a commit if desired.

## 5. Proposed long-term solution

Two layers. Do **both**: land (A) immediately for correctness, pursue (B) for performance.

### (A) Correctness now — route RDNA4 hd256 to the single-warp tile
Not a feature-kill: the single-warp tile is still WMMA-accelerated prefill and is **proven correct** here. This restores good-commit (`66eec7f54`) behavior.
- **Runtime, zero-rebuild:** set `GGML_PAGED_TILE_MULTIWARP=0` in the service unit env. Dense `qwen35` turbo4-paged then works today.
- **Durable code gate** in `mt_pagedattn.cu` at the `mw_on` selection (~L1720):
  ```cpp
  const bool is_rdna4 = /* cc corresponds to gfx1201 */;
  bool mw_on = get_paged_tile_multiwarp_mode() != 0
               && !(is_rdna4 && HS == 256);   // mw WMMA tile unvalidated/broken on RDNA4 hd256
  ```
  (Prefer keying off a proper `amd_wmma`/arch predicate already available in that scope — see `amd_wmma_available(cc)` used nearby.)

### (B) Real fix — repair the multi-warp cooperative kernel on RDNA4 hd256
1. **Pin the exact changed dependency** (mechanism): with the repro, revert-test the two sync-changed shared deps independently against the mw path:
   - restore `turbo-quant.cuh` to its `66eec7f54` version, rebuild, test;
   - if not that, bisect the `mt_pagedattn.cu` (+511) shared helpers the mw kernel calls.
2. **Confirm the ornith-vs-27b discriminator** (`n_kv_heads=4`) — instrument the mw kernel's per-(kv_head, qblock) tile indexing / smem offsets; the divergence at `n_kv_heads=4` localizes the faulty index.
3. If it turns out to be a **WMMA C-output lane-layout** issue on RDNA4 `v_wmma_f32_16x16x16_f16` (cross-reference the earlier `mma.cuh` WMMA-decode-bug investigation), write the **layout probe kernel** (identity A, `B[i][j]=i*16+j`, per-lane printf of the 8 C outputs) to derive the true `(lane,slot)→(row,col)` map before editing formulae. Do **not** guess the layout.
4. Add an **RDNA4 hd256 turbo4 paged e2e correctness gate** (the missing validation) so this can't silently regress again.

> ⚠️ **gfx1201 brick risk:** editing/running experimental WMMA kernels on this card can wedge the desktop. Use crash-survivable progress logging and keep the known-good single-warp path (A) as the default while iterating on (B).

---

## 6. Morning quick-start (fastest path to productivity)

1. Reproduce (30 s): launch the §3 base command with default env → confirm `////`. Then `GGML_PAGED_TILE_MULTIWARP=0` → confirm coherent. That is the whole bug in two runs.
2. Decide fix scope: land (A) code gate immediately (low risk), then start (B) step 1 (revert-test `turbo-quant.cuh`).
3. Answer the ornith discriminator (§4 open sub-question) — it likely hands you the exact buggy index.

---

## 7. Build / test reference

- Build (incremental): `make -C build-hip llama-server -j4` (~45 s for 1 kernel file; **several minutes** if `ggml-cuda.cu`/`mt_pagedattn.cu` changed).
- ⚠️ **Incremental builds across the 07-06 merge are unreliable** — checking out `06a3da0e6` and incremental-building produced a binary that **core-dumps on startup**. `66e` and `676` incremental builds ran fine. For clean per-commit bisect, prefer a fresh build dir.
- Both builds must stay in the set for shipping: `build-hip` (this machine) **and** `build-army` on mad-lab-2026.
- Test harness pattern used this session: `setsid nohup … &`, poll log for `listening on`, `curl /v1/chat/completions`, parse `choices[0].message.reasoning_content`. Kill servers with `fuser -k 8091/tcp` (do **not** `pkill -f <pattern>` where `<pattern>` also matches your own shell — that self-terminates the command, exit 143/144).

---

## 8. Interim workarounds (both correct, until (B) lands)
- `GGML_PAGED_TILE_MULTIWARP=0` (keeps turbo4 paged + WMMA single-warp prefill), **or**
- `--cache-type-k/v q8_0` (turbo4 only breaks on the mw paged path; q8_0 paged is clean).

MoE (`qwen35moe`, e.g. ornith) and hd128 (`llama`) models are unaffected either way.

---

## 9. Issue A (embedder crash) — FIXED, for the record (don't lose this)

Swapped the tiered-KV semantic embedder from `granite-embedding-small-english-r2` (384-dim) to **`LFM2.5-Embedding-350M-Q8_0`** (arch `lfm2` / Lfm2BidirectionalModel, 1024-dim, CLS pooling, cosine, max input 512, requires `query: ` / `document: ` prefixes).

Root cause of the crash: `mt-embed.cpp` clamped each chunk to `llama_n_ctx(ctx_)`, but the model silently inflates n_ctx to its trained max (granite→8192, LFM2.5→128000), so a chunk exceeded the 512 batch budget → `decode()` assert `n_tokens_all <= n_batch` (`llama-context.cpp:1806`; the earlier granite variant hit the encoder assert `n_ubatch >= n_tokens` at `:1462`).

Changes (in `src/memory-tier/mt-embed.{h,cpp}`, `mt-tiered.{h,cpp}`, `tools/server/server-context.cpp`), **built + verified on build-hip AND build-army**:
1. pooling `MEAN` → `CLS`.
2. `n_embd_ = llama_model_n_embd(model_)` → `llama_model_n_embd_out(model_)` (dense_2 projection → 1024; falls back to n_embd for plain encoders, so safe for all).
3. clamp each chunk to `EMBED_BATCH_TOKENS` (512 = n_batch/n_ubatch), **not** `llama_n_ctx` — this is the crash fix.
4. added `enum mt::EmbedRole{Document,Query}`, prepend `document: `/`query: ` prefixes; query site `server-context.cpp:3686` passes `EmbedRole::Query`, fingerprint sites default `Document`.

Verified: embedder ran clean on a 1542-token prompt (`tier semantic: prefill fingerprint sweep — 96 blocks`), no abort. These edits are currently **uncommitted** (part of the working-tree WIP).

---

## 10. Current repo / machine state
- Branch `feat/wp-dflash-ds4` @ `d71223efa`, **uncommitted WIP present** (pre-existing weight-pager/mt-mover-recurrent WIP + the §9 embedder fix). Not committed.
- `build-hip/bin/llama-server` rebuilt at tip, **consistent** (reproduces the garbage, no startup crash).
- All test `llama-server` processes stopped; systemd services stopped.
- Systemd unit `llama-server-qwen36-27b-swatm-r9700` currently points `--kv-tier-semantic-index` at `LFM2.5-Embedding-350M-Q8_0.gguf`.

## 11. Key references
- Corrupting kernel: `ggml/src/ggml-cuda/mt_pagedattn_tile.cu` (`mt_paged_attention_tile_mw_kernel`, `launch_paged_attn_tile_mw`).
- Dispatch + env toggles: `ggml/src/ggml-cuda/mt_pagedattn.cu` (~L22–90 toggles, ~L1680–1760 dispatch, ~L352 scatter).
- WMMA tile math: `ggml/src/ggml-cuda/mma.cuh` (RDNA4 C-output layout — unchanged by sync; cross-ref prior WMMA-decode-bug notes).
- Bisect: GOOD `66eec7f54`, BAD `676b87554`, culprit merge `06a3da0e6`; ruled-out `9e8310ecb`.
- Upstream commits in the merge worth eyeing for (B) step 1: `009d9716e`/`c2280285a` (sp2.5 turbo4_64, touched `mt_pagedattn.cu`+`turbo-quant.cuh`), `0ed235ea2` ("cudaMemcpy2DAsync fast path", `cpy.cu`), `78d2f5246` (quantized concat), `5a460dea9` (gdn copy removal — unlikely, would hit q8 too).
