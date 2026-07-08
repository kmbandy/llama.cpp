# DFlash-on-DS4-Flash Implementation Plan

> **For agentic workers:** implement task-by-task. Each task ends with an independently checkable deliverable. Steps use `- [ ]`.

**Goal:** Enable `--spec-type draft-dflash` for DeepSeek V4 Flash in our llama.cpp fork using the RedHatAI DFlash drafter, running against the NVMe-paged DS4 trunk.

**Architecture:** DFlash spec-decode already exists (PR #22105) for single-residual arches. DS4 Flash uses 4-stream hyper-connections, so the drafter's encoder input is `5 layers × hc_mult(4) × 4096 = 81920` wide. Two buckets: (A) converter tensor-mapping for the DS4 drafter, (B) runtime coupling to feed the draft DS4's pre-collapse multi-stream residual.

**Tech Stack:** C++17 llama.cpp, Python converter (`convert_hf_to_gguf.py` + `conversion/*.py`), HIP/ROCm build (`build-hip`, dual-arch gfx1201;gfx1030).

## Global Constraints

- Branch: `feat/wp-dflash-ds4` (off `feat/wp-attention-island`), on mad-lab-main `/home/kmbandy/GitHub/llama.cpp`. Verify `git branch --show-current`; if not on it, STOP and report — do NOT switch/cherry-pick to another branch.
- Spec: `docs/superpowers/specs/2026-07-08-dflash-ds4-speculative-design.md` — read it first; it has the tensor table, shapes, and existing-code anchors.
- Do NOT run LLM inference (llama-server/llama-cli/model decode). GPU validation is human-gated. You MAY run the converter (it's not inference) and build.
- Commit per task on `feat/wp-dflash-ds4`; do NOT push. Do NOT amend existing commits.
- Artifacts on disk: drafter `/home/kmbandy/models/dflash-speculator/` (config.json already flattened; original at config.json.orig), target cfg `/home/kmbandy/models/ds4flash-hf-cfg/`.
- HIP code changes must build via `build-hip` (`cmake --build build-hip --target llama-server llama-gguf -j"$(nproc)"`, no reconfigure unless new files). CPU-only pieces (converter, pure C++ metadata) don't need HIP.

---

## Task 1: Converter — map the DS4 DFlash drafter to a GGUF

**Files:**
- Modify: `conversion/qwen.py` (the `DFlashModel` class) — or add a dedicated DS4 DFlash class if cleaner.
- Reference (do NOT duplicate blindly — reuse): `conversion/llama.py` EAGLE3 handling of `d2t`/`t2d`/`.hidden_norm.weight`/`fc` (~lines 225-333).
- Reference: `src/llama-arch.cpp` — `LLM_TENSOR_FC` ("fc"), `LLM_TENSOR_D2T` ("d2t"), `enc.*`, `LLM_ARCH_DFLASH`.

**Deliverable:** `python3 convert_hf_to_gguf.py /home/kmbandy/models/dflash-speculator --target-model-dir /home/kmbandy/models/ds4flash-hf-cfg --outfile /home/kmbandy/models/dflash-speculator-DS4.gguf --outtype bf16` runs to completion and produces the GGUF.

- [ ] **Step 1: Reproduce the current failure** — run the convert command above; confirm it fails at `ValueError: Can not map tensor 'model.d2t'` (the flatten already got past hparams). This is the starting point.

- [ ] **Step 2: Map the DFlash-specific tensors.** In `DFlashModel.modify_tensors` (or a new class), handle the 7 non-standard tensors from the spec's table:
  - `d2t` → `LLM_TENSOR_D2T` and `t2d` (reuse the `conversion/llama.py` EAGLE3 logic: capture d2t's original int dtype before the parent F32-casts, write d2t as **absolute target token ids** with the range/duplicate validation against the target vocab size 129280).
  - `fc.weight` → `LLM_TENSOR_FC` (keep it [4096, 81920] — do NOT reshape).
  - `hidden_norm.weight` → the eagle3 hidden_norm slot (see `conversion/llama.py:234`).
  - `embed_tokens.weight` → `token_embd`, `lm_head.weight` → `output`, `norm.weight` → `output_norm`.
  - Standard `layers.N.*` map via the inherited Qwen3 path (q/k/v/o + q_norm/k_norm + MLP) — verify they don't error.

- [ ] **Step 3: Emit the metadata.** Write GGUF KV: `dflash.block_size` (8), target layers `[3,13,23,32,42]` (the existing `add_target_layers` path — note the converter's existing `+1` offset convention; confirm it matches what the runtime expects for deepseek4), `mask_token_id` (1), draft vocab size (32000), and a **new `dflash.hc_mult` = 4** key (add the writer helper in `gguf-py/gguf/gguf_writer.py` + the KV constant in `gguf-py/gguf/constants.py` and `src/llama-arch.cpp` `LLM_KV`). If the deepseek `hyper_connection.count` KV is a cleaner fit, reuse it instead — document the choice.

- [ ] **Step 4: Run the converter to completion.** Re-run the command from the Deliverable. Expected: a GGUF is written, no unmapped-tensor errors.

- [ ] **Step 5: Inspect the GGUF.** `./build-hip/bin/llama-gguf /home/kmbandy/models/dflash-speculator-DS4.gguf r n 2>/dev/null | grep -iE "dflash|target|hc_mult|fc|d2t|arch"` — confirm arch=dflash, fc present [4096,81920], target layers, block_size, hc_mult=4. (llama-gguf is a metadata reader, not inference — allowed.)

- [ ] **Step 6: Commit.** `git add conversion/ gguf-py/ src/llama-arch.cpp && git commit -m "feat(dflash): convert DS4 Flash DFlash drafter (d2t/t2d/fc/hidden_norm + hc_mult)"`

---

## Task 2: Runtime — size the DFlash encoder feature by hc_mult

**Files:**
- Modify: `common/speculative.cpp` (`common_speculative_impl_draft_dflash`, ~line 903+): `n_embd_enc` computation.
- Modify: `src/llama-model.cpp` / `src/llama-arch.cpp` — read the new `dflash.hc_mult` metadata into the model (accessor like `llama_model_dflash_hc_mult`).

**Deliverable:** with the converted draft loaded, the DFlash impl computes `n_embd_enc = target_layer_ids_n * hc_mult * n_embd_tgt` (= 81920) and the feature buffer / `batch_inject` sizes match `fc`'s input dim.

- [ ] **Step 1:** Add a model accessor for `hc_mult` (default 1 when absent, so Qwen/Gemma drafters are unchanged).
- [ ] **Step 2:** In the dflash impl ctor, change `n_embd_enc = target_layer_ids_n * n_embd_tgt` → `target_layer_ids_n * hc_mult * n_embd_tgt`. Size `features_buf` and the injection accordingly. Guard: `hc_mult==1` reproduces the current behavior exactly (regression-safe for the existing supported drafters).
- [ ] **Step 3:** Build: `cmake --build build-hip --target llama-server -j"$(nproc)"` — must exit 0.
- [ ] **Step 4: Commit.** `git commit -m "feat(dflash): size encoder feature by hc_mult (DS4 4-stream)"`

---

## Task 3: Runtime — expose deepseek4 pre-collapse multi-stream at tap layers

**Files:**
- Investigate + modify: `src/llama-context.cpp` (`extract_layer_inputs` ~2273, `get_layer_inp`), `src/llama-graph.cpp` (deepseek4 HC graph: `get_hca_plan`, `dsv4_set_comp_inputs`, `n_stream`), `src/llama-model.cpp` (deepseek4 build).

**Deliverable:** when a DFlash draft is attached to a deepseek4 target, `extract_layer_inputs` captures the **`n_stream`-wide (4×4096) pre-collapse** hyper-connection residual at each target layer id, concatenated to 81920 across the 5 layers — matching `fc`.

- [ ] **Step 1: Investigate (write findings into the task report).** In the deepseek4 graph build, identify the tensor that is the `n_stream`-wide residual *before* the hc_head collapse at a layer boundary. Confirm what `get_layer_inp(il)` currently returns for deepseek4 (single-stream 4096 vs multi-stream). This is the crux — report the exact tensor/op.
- [ ] **Step 2:** Make the layer-input extraction, for a deepseek4 target with an attached DFlash draft, expose the multi-stream pre-collapse state (16384-wide) at the tap layers. Keep single-stream behavior for non-HC arches (Qwen/Gemma) unchanged.
- [ ] **Step 3:** Ensure the DFlash graph build (`LLM_ARCH_DFLASH`) accepts the 81920-wide `fc` input (verify it isn't hard-coded to `target_layers * n_embd`; if it is, wire it to `n_embd_enc`).
- [ ] **Step 4:** Build `build-hip` (exit 0).
- [ ] **Step 5: Commit.** `git commit -m "feat(dflash): expose deepseek4 pre-collapse n_stream residual for DFlash tap"`

- [ ] **Step 6: Report the exact GPU-validation command** for the human (do NOT run it): the `llama-server` invocation with `--model <trunk> --model-draft /home/kmbandy/models/dflash-speculator-DS4.gguf --spec-type draft-dflash --spec-draft-n-max 4 --no-mmap --weight-paging ... --device ROCm0,ROCm1 ...`, plus what to look for (draft acceptance > 0, coherent output, decode t/s vs 1.038 baseline).

---

## Uncertainty guidance for the implementer

- Task 3 is the hard one and is investigation-first: if the pre-collapse tensor isn't cleanly extractable via the existing `get_layer_inp` path, STOP and write a detailed report of what the deepseek4 graph exposes and where — do not force a wrong-shaped hack. The human/Claude will review before proceeding.
- If Task 1's `add_target_layers` `+1` offset or the d2t remap conflicts with what deepseek4 expects, flag it rather than guessing.
- Prefer `hc_mult==1` fall-through everywhere so the existing Qwen3/Gemma DFlash support is provably unchanged.
