# Instella-MoE-16B-A3B support — design

**Date:** 2026-07-25
**Repo:** `~/GitHub/llama.cpp` on **mad-lab-2026**, master
**Model:** `amd/Instella-MoE-16B-A3B-Think` (Base/SFT/DPO/Midtrain share the architecture)
**Deployment target:** RX 6900 XT (gfx1030, 16 GB) on **mad-lab-main**

## 1. Why

AMD's in-house MoE: 27 layers, 2048 hidden, 64 routed + 2 shared experts, top-6,
16B total / 3B active. At Q6_K it lands around 13 GB and fits the 6900 XT
resident with room for KV — a genuinely good fit for that card.

It is also a far more convenient MoE pager target than DS4-Flash: 31.7 GB of
source weights against DS4's 151 GB, with 64 experts per layer to page.

`llama.cpp` cannot convert it today. `conversion/__init__.py` dispatches on
`architectures[0]`, and this model reports `InstellaMoEForCausalLM`, which is not
registered.

## 2. What it is, precisely

`model_type` is `deepseek_v3` and the tensor inventory is DeepSeek-V3's, verified
against `model.safetensors.index.json`:

- MLA attention: `q_proj` (no q_lora — `q_lora_rank` is null),
  `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj`.
  `kv_lora_rank` 512, `qk_nope_head_dim` 96, `qk_rope_head_dim` 32,
  `qk_head_dim` 128, 16 heads, `num_key_value_heads` 16.
- MoE routing: `mlp.gate.weight` + `mlp.gate.e_score_correction_bias` — the
  DeepSeek-V3 sigmoid-with-bias router. `routed_scaling_factor` 2.5,
  `norm_topk_prob` true, `n_group` 1.
- Layer 0 is dense (`first_k_dense_replace` 1); layers 1-26 are MoE.
- YaRN rope scaling, factor 40, `original_max_position_embeddings` 4096,
  `rope_theta` 8e6, `rope_interleave` true.

**The entire delta from DeepSeek-V3 is two features.**

### 2a. Gated attention — one extra tensor, one line

`config.gated_attention` is true. Each layer carries
`self_attn.gate_proj.weight` (27 of them, the only non-DeepSeek tensor in the
model). From `modeling_instella_moe.py:125`:

```
attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
```

applied to the attention output **before** `o_proj`, with `hidden_states` being
the layernormed attention input. Trivial to express.

### 2b. farskip — a dual residual stream

`config.farskip` is true. The config class defaults (`configuration_instella_moe.py`)
are `farskip_start_idx=0`, `farskip_end_idx=1e4`, `attn_only_farskip=False`,
`mlp_only_farskip=False`, and `config.json` overrides none of them — so **farskip
is active on every layer in range**, i.e. all MoE layers.

A farskip MoE layer carries two tensors instead of one. Writing `R` for the full
residual and `A` for the shared-experts-only stream, per
`modeling_instella_moe.py:184-229`:

```
in: (R, A)
attn_out = Attn(LN(A))          <- attention reads A
R'       = R + attn_out         <- but the residual add is onto R
routed, shared = MoE(LN(R'))
out: (R' + routed,  R' + shared)
```

The point, stated in their own comment, is that `residual_no_routed` is
"combine-free and feeds the next block's attention" — the next layer's attention
never depends on the routed-expert output. It is a latency-hiding structure.

Boundary conditions that must be right:

- **Layer 0 is dense**, so its `mlp` is not a `FarSkipMoE` and it returns a
  single tensor. The dual stream begins at layer 1.
- On the **first** farskip layer the input is not yet a tuple, and all three of
  `residual`, `input_to_attn`, `input_to_mlp` are the same tensor.
- The **final** layer's output feeds `model.norm`; take the full residual `R`,
  not `A`.

Getting the two streams crossed produces a model that loads and generates
fluent-looking text while being subtly wrong, so this is the part to test hardest.

## 3. Work required

1. **Converter.** A class registered for `InstellaMoEForCausalLM`, subclassing
   the existing `DeepseekV2Model` in `conversion/deepseek.py` — that class
   already handles MLA, the sigmoid router with `e_score_correction_bias`,
   expert stacking, and `first_k_dense_replace`. Add the attention-gate tensor
   mapping and emit the two new hparams. Register in `conversion/__init__.py`
   alongside the existing `"DeepseekV3ForCausalLM": "deepseek"` entries.
2. **gguf-py.** A new architecture constant and a tensor enum for the attention
   gate, plus KVs for `gated_attention` and the farskip range, in
   `gguf-py/gguf/constants.py` and `tensor_mapping.py`.
3. **`src/llama-arch.{h,cpp}`.** Architecture name, tensor-name table, KV keys.
4. **`src/llama-hparams.h` / `llama-model.cpp`.** Load the flags and range;
   create the per-layer gate tensor.
5. **`src/models/`.** The graph. Start from the DeepSeek-V3 builder, add the
   sigmoid gate, and carry the second residual through MoE layers.

Deciding whether this is a distinct `LLM_ARCH` or a flag on the existing
DeepSeek-V3 arch is the implementer's call; state the reasoning in the commit.
A separate arch is cleaner if the graph divergence is significant, which farskip
probably makes it.

## 4. Verification

1. **Conversion completes with no unmapped tensors.** The converter raises on
   unknown tensors; that is the desired behaviour and must not be suppressed.
   Confirm all 5344 tensors are accounted for and that `self_attn.gate_proj`
   landed somewhere rather than being silently dropped.
2. **Coherent generation** at Q8_0 or bf16 before any quantization judgement.
3. **Perplexity against a reference.** There is no llama.cpp baseline for this
   model, so the reference must be the HF implementation with `trust_remote_code`
   on the same corpus and context. Note the standing fleet finding that HF and
   `llama-perplexity` differ by ~0.30 PPL on identical f16 weights from framework
   differences alone — so a small gap is expected and only a large one is
   evidence of a bug. Do not compare across frameworks without restating that.
4. **A farskip-specific check.** Equivalence on shapes that exercise the dual
   stream is not provable from generation quality alone. Compare hidden states
   layer-by-layer against the HF model for a fixed short prompt, or at minimum
   verify that deliberately swapping the two streams changes the output — a test
   that cannot fail is worthless.
5. **Expert paging**, once it runs: the pager matches experts on the tensor-name
   pattern `ffn_(up|gate|down)_exps\.` with no architecture gating, so it should
   work with no pager changes. 64 experts across 26 layers.

## 5. Practical notes

- Source weights are 31.7 GB across 6 safetensors shards; **`/mnt/storage2`**
  (1.3 TB free) is the only place on mad-lab-2026 with room. `/` has 38 GB free
  and cannot hold source plus output.
- `conversion/` output at Q8_0 is roughly 17 GB; Q6_K for the 6900 XT is roughly
  13 GB. Only the quantized file needs to reach mad-lab-main.
- The 6900 XT lives on **mad-lab-main**, whose tree has unrelated uncommitted
  work including a live DSWS spike. Do the development on mad-lab-2026 and move
  only the GGUF.
- A board claim is required before any GPU run on either machine.
