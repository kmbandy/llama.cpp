# DFlash Speculative Decoding for DeepSeek V4 Flash — Design

**Date:** 2026-07-08
**Status:** Approved for planning (user greenlit spec → plan → Codex handoff)
**Branch:** `feat/wp-dflash-ds4` (off `feat/wp-attention-island`)
**Related:** `docs/dev/2026-07-08-ds4flash-decode-levers.md` (MTP/DFlash is the spine lever)

## Goal

Make DeepSeek V4 Flash decode faster on mad-lab-main (single stream, `--parallel 1`) by enabling **DFlash speculative decoding** in our llama.cpp fork, driven by the ready-made RedHatAI DFlash drafter, running against our NVMe-paged DS4 Flash trunk. Target: a real decode speedup (DeepSeek reports 60–85% for the full DSpark stack; plain DFlash without the SAR refinement is a large fraction of that). The paging cost amortizes over each accepted draft block, and the drafter also gives us the signal for MTP-aware prefetch/retention later.

## Why this is reachable (and why not vLLM)

- DFlash speculative decoding is **already merged into llama.cpp** (PR #22105 + follow-ups #25110, #25246), present in our fork: `--spec-type draft-dflash`, `common_speculative_impl_draft_dflash`, and a `DFlashModel` converter. It supports Qwen3/Qwen3.5/GPT-OSS/Gemma-4 drafters today.
- The big-speedup DFlash/DSpark reference stacks are vLLM-shaped, but **vLLM cannot run DS4 Flash on this box** (151 GB model, 48 GB VRAM + 15 GB RAM, no NVMe expert paging). Our llama.cpp weight-pager is the only thing that runs the trunk here — so the DFlash drafter must run *in llama.cpp*, which #22105 makes possible.
- **The only reason it's not plug-and-play** is DS4 Flash's hyper-connections (HC / multi-hyper-connection). The generic DFlash path was built for single-residual arches (Qwen/Gemma); DS4's drafter consumes the **4-stream** pre-collapse HC residual. That coupling is the work.

## Target artifacts (already on disk)

- Drafter: `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` → `/home/kmbandy/models/dflash-speculator/` (2 B bf16, 3.6 GB, `model.safetensors`). Config was flattened from the vLLM `speculators` schema into `config.json` (original kept as `config.json.orig`).
- Target config/tokenizer for `--target-model-dir`: `/home/kmbandy/models/ds4flash-hf-cfg/` (config.json + tokenizer.json + tokenizer_config.json).
- Trunk GGUF (paged): `/home/kmbandy/Downloads/DeepSeek-V4-Flash-UD-Q8_K_XL-0000{1..5}-of-00005.gguf`.

## Drafter structure (measured from the safetensors)

62 tensors, 5 layers. Standard Qwen3-style layers (q/k/v/o with QK-norm, GQA 64:1, SwiGLU MLP) **plus** these DFlash/DS4-specific tensors:

| tensor | shape | meaning | maps to |
|---|---|---|---|
| `fc.weight` | [4096, **81920**] | encoder projection of the tapped target features → draft hidden | `LLM_TENSOR_FC` ("fc") |
| `d2t` | [32000] | draft→target token id map (reduced 32000 draft vocab) | `LLM_TENSOR_D2T` ("d2t") |
| `t2d` | [129280] | target→draft token id map | eagle3 `t2d` (see `conversion/llama.py`) |
| `embed_tokens.weight` | [129280, 4096] | draft embedding over full target vocab | `token_embd` |
| `lm_head.weight` | [32000, 4096] | draft output over reduced draft vocab | `output` |
| `hidden_norm.weight` | [4096] | norm on injected target features | eagle3 `hidden_norm` |
| `norm.weight` | [4096] | final norm | `output_norm` |

**The load-bearing number: `fc` input width = 81920 = 5 tapped layers × 16384 = 5 × (hc_mult 4 × 4096).** The generic DFlash runtime feeds `target_layers × n_embd = 5 × 4096 = 20480` — exactly `hc_mult`× too small. Matching 81920 is the crux of the runtime work.

Config facts: `aux_hidden_state_layer_ids = [3,13,23,32,42]`, `block_size = 8` (drafts ≤7 tokens), `mask_token_id = 1`, `draft_vocab_size = 32000`, `hc_mult = 4`, sliding-window 2048 on all 5 layers, `head_dim = 256`, hidden 4096, 64 heads / 1 KV head.

## What already exists in our fork (build on, don't rebuild)

- **Arch slots**: `LLM_ARCH_DFLASH`, `LLM_TENSOR_FC`, `LLM_TENSOR_D2T`, full `enc.blk.*` encoder tensors, `enc.output_norm`. No new tensor *types* required.
- **Converter precedent**: `conversion/llama.py` already maps `d2t`/`t2d`/`.hidden_norm.weight`/`fc` for EAGLE3 (captures d2t dtype pre-F32, writes d2t as absolute target ids with range/dup validation). The DS4 drafter's non-standard tensors are exactly this set.
- **Runtime**: `common_speculative_impl_draft_dflash` (block drafting, draft-side KV injection, non-causal draft, `target_layer_ids`, `dflash.block_size`), `llama_set_embeddings_layer_inp` / `extract_layer_inputs` / `get_layer_inp`, `n_embd_enc = target_layer_ids_n × n_embd_tgt`.
- **deepseek4 HC machinery**: the graph is already multi-stream — `n_stream`, `get_hca_plan`, `dsv4_set_comp_inputs(..., "hca", ...)`, HC tensors (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`), and `hyper_connection.count` metadata. The pre-collapse residual is the `n_stream`-wide state before the hc_head collapse.

## Work buckets

### Bucket A — Converter (bounded; has precedent)
A DS4-aware DFlash converter path that:
1. Reads the flattened speculators schema (already flattened on disk; the converter should ideally read `transformer_layer_config` natively so we don't depend on the manual flatten — but consuming the flattened `config.json` is acceptable for v1).
2. Maps the 7 non-standard tensors per the table above, reusing the `conversion/llama.py` EAGLE3 handling for `d2t`/`t2d`/`hidden_norm`/`fc`.
3. Emits GGUF metadata: `dflash.block_size`, target layers `[3,13,23,32,42]`, `mask_token_id`, and a **new `dflash.hc_mult` (=4)** key (and/or reuse the deepseek `hyper_connection.count`), plus draft-vocab size for the d2t remap.
4. Produces a GGUF that `llama-model` loads as arch `dflash` with `fc` sized [4096, 81920].

**Gate:** `convert_hf_to_gguf.py` runs to completion on the drafter and produces `dflash-speculator-DS4.gguf`; `llama-gguf` / a model-load dry-run shows the expected tensors + metadata (fc 81920 wide, target layers, hc_mult).

### Bucket B — Runtime HC coupling (the real work)
1. **DFlash tap sizing:** when the draft declares `hc_mult > 1` (new metadata), size `n_embd_enc = target_layer_ids_n × hc_mult × n_embd_tgt` (→ 81920) so the injected feature matches `fc`.
2. **deepseek4 pre-collapse exposure:** at each `target_layer_id`, expose the `n_stream`-wide (4×4096=16384) pre-collapse HC residual for extraction, instead of the collapsed single-stream 4096. Hook `extract_layer_inputs` / the layer-input tap so a DFlash draft attached to a deepseek4 target receives the multi-stream state; concatenate across the 5 layers → 81920.
3. Feed the 81920-wide features into the draft's `fc`; the rest of the DFlash block-draft loop is unchanged.

**Gate:** with the trunk + converted draft loaded, `--spec-type draft-dflash` reserves and runs without dimension asserts; a short generation is coherent; draft acceptance > 0.

### Bucket C — Validation (GPU, human-gated)
Run the paged DS4 Flash trunk + the DFlash draft with `--spec-type draft-dflash --spec-draft-n-max 7`, `--no-mmap`, on mad-lab-main. Measure: draft acceptance rate / avg accepted length, decode t/s vs the 1.038 t/s single-card baseline, coherence, 0 GPU faults. Log to `/home/kmbandy/wp_logs`.

## Out of scope (later)
- **DSpark SAR refinement** (the second-stage semi-autoregressive pass, issue #25096) — vanilla DFlash first; SAR is a follow-on.
- **MTP-aware prefetch/retention** (feeding the draft's expert predictions into the pager) — separate lever once DFlash decodes.
- Cross-device draft/verify split — deferred; get DFlash working single-device (all on the paging card) first.

## Risks / unknowns
- **Pre-collapse extraction point.** Which exact tensor in the deepseek4 graph is the `n_stream`-wide pre-collapse residual at a layer boundary, and whether `get_layer_inp` can be pointed at it, is the main investigation Codex must do against the live graph.
- **Drafter arch fidelity.** The drafter's own layers are Qwen3-style, but the `fc`/`hidden_norm`/injection path in `LLM_ARCH_DFLASH`'s graph must accept an 81920-wide input; verify the dflash graph build isn't hard-coded to `target_layers × n_embd`.
- **Draft vocab remap.** d2t/t2d must map correctly against the DS4 Flash 129280 vocab (the EAGLE3 validation path guards range/dups).
- **Acceptance rate.** Vanilla DFlash (no SAR) suffers suffix decay past the first ~2 positions; `--spec-draft-n-max` may need tuning (start ~3–4, not the full 7).

## Execution
Codex implements Buckets A and B (Claude reviews); CPU-buildable pieces build via the CPU test path, HIP pieces via `build-hip`. GPU validation (C) is human-gated. No inference in Codex tasks.
