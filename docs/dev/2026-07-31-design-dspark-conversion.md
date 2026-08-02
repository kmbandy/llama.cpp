# Design: convert DeepSeek-V4-Flash WHOLE — MTP and DSpark included, FP8 preserved

Status: design, ready for implementation. SUPERSEDES the earlier version of this file, which
specified the wrong architecture.
Author: Claude (design/review). Implementation to grok.
Date: 2026-07-31

## 0. THE REQUIREMENT, AND THE MISTAKE THAT PRECEDED IT

**Produce ONE GGUF from the ORIGINAL DeepSeek-V4-Flash-0731 safetensors, containing the whole
model exactly as DeepSeek ships it: dense at FP8, routed experts at native MXFP4, AND the
`mtp.*` subtree with the DSpark heads. Target size ~167 GB. One file. No separate draft model.
No dependency on anyone else's GGUF.**

An earlier version of this document said the opposite — "do not re-encode the main model, use
Unsloth's 162 GB target unchanged, emit a separate draft GGUF". That was built (a 10.9 GB draft
at `/home/kmbandy/models/DeepSeek-V4-Flash-DSpark-src/out/dspark-draft-f16.gguf`) and it is the
wrong artifact. It is throwaway. Two things were wrong with it:

1. It splits the model in two when the original is one checkpoint.
2. It depends on Unsloth's file, which WIDENS FP8 dense to BF16 — measured 14.54 GB BF16
   against ~8.9 GB of FP8 source. That is a pointless 6 GB of lossless-but-larger re-encoding
   and it is exactly what we are converting ourselves to avoid.

## 1. THE PATTERN ALREADY EXISTS IN THIS FILE

`conversion/deepseek.py` already ships MTP inside a main-model GGUF for DeepSeek V3.2:

```python
class DeepseekV2Model(TextModel):
    skip_mtp = True
    ...
    # skip Multi-Token Prediction (MTP) layers
    if self.skip_mtp:
        block_count = self.hparams["num_hidden_layers"]
        match = re.match(r"model.layers.(\d+)", name)
        if match and int(match.group(1)) >= block_count:
            return

class DeepseekV32Model(DeepseekV2Model):
    skip_mtp = False                                                   # <- includes MTP
    self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
    if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
        self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)
```

`DeepseekV4Model` is a SEPARATE class that does not inherit that flag or that mechanism. It
drops the tensors instead:

```python
class DeepseekV4Model(TextModel):
    _skipped_mtp_tensors = 0
    ...
    if name.startswith("mtp."):
        cls._skipped_mtp_tensors += 1
    ...
    logger.info("Skipping %d DeepSeek-V4 MTP tensor(s) for conversion v0", ...)
```

**This is the same three moves V3.2 already makes, keyed on the `mtp.` prefix instead of a
layer index.** V3.2's mechanism matches `model.layers.{N >= block_count}`; DS4's MTP is
namespaced `mtp.{0,1,2}.*` with `model.layers` topping out at 42. So the index test does not
apply, but everything else transfers.

## 2. What to build

### 2.1 Include the MTP subtree in the main conversion

In `DeepseekV4Model`:

- delete the `mtp.*` skip and the `_skipped_mtp_tensors` counter
- `block_count = num_hidden_layers + (number of mtp stages present)`. Derive the stage count
  from the tensors, do not assume `num_nextn_predict_layers` (config says 1; the checkpoint
  ships `mtp.0`, `mtp.1`, `mtp.2`)
- `add_nextn_predict_layers(...)` with the real stage count
- map `mtp.{stage}.{rest}` onto the block namespace so the MTP stages become blocks
  `43, 44, 45` following the main stack's `0..42`
- map the three DSpark heads to the existing tensor enums:
  `mtp.2.markov_head.markov_w1.weight` -> `DSPARK_MARKOV_W1`
  `mtp.2.markov_head.markov_w2.weight` -> `DSPARK_MARKOV_W2`
  `mtp.2.confidence_head.proj.weight`  -> `DSPARK_CONF_PROJ`
- emit `dspark_block_size` (5), `dspark_target_layer_ids` (config `[40,41,42]`, +1 offset),
  `dspark_noise_token_id` (128799) as mask token, `dspark_markov_rank` (256)

`gguf-py/gguf/tensor_mapping.py` currently expects `model.markov_head.markov_w1`, which the
release does not ship. Fix the prefixes to the real `mtp.2.*` names. That mapping has never
fired.

### 2.2 PRESERVE FP8 — do not widen to BF16

`GGML_TYPE_F8_E4M3` exists in `ggml/include/ggml.h`. The source dense weights are FP8 e4m3.
Emit them as F8_E4M3, not BF16. Experts stay native MXFP4 (the class already sets
`_is_mxfp4 = True` / `MOSTLY_MXFP4_MOE` — keep that).

Expected result: ~167 GB, matching DeepSeek's own total, NOT the 172 GB you get by widening.

**If ggml has the F8_E4M3 type but no compute kernels for it on the tensor roles involved,
STOP AND REPORT.** A model that loads and then aborts at the first matmul is worse than one
that never converted. Check before converting 167 GB. This is a design question and it is mine
to answer.

### 2.3 Runtime — mostly already done

The upstream sync brought DSpark's decode path (84075273c / #25173). Already correct on HEAD
after tonight's work:

- `common/common.h` — `conf_min = 0.9f` (KEEP THIS; DeepSeek's guidance and antirez's DwarfStar
  both use 0.9; the reference PR had 0.0 which disables the gate)
- `common/arg.cpp` — `--spec-draft-conf-min`, env `LLAMA_ARG_SPEC_DRAFT_CONF_MIN`
- `common/speculative.cpp` — uses `conf_min` (HEAD wrongly gated on `p_min`) and logs the
  actual runtime value at WARN

Leave those alone; they are right.

**What may still be missing:** the reference PR carries a ~630-line `dflash.cpp` path putting
the DFlash graph on `llm_graph_context_dsv4_mla` for a DeepSeek-V4 backbone. Whether that is
needed when the MTP blocks live INSIDE the main model — rather than as a standalone draft — is
an open question. Determine it and report; do not port 630 lines speculatively.

## 3. Reference, not authority

PR ggml-org/llama.cpp#25683, fetched as remote `yaniss`, branch `dspark-dsv4`, head
`c2e51866`. Read it: `git diff c71854292..yaniss/dspark-dsv4`.

**It was closed for PROCESS reasons** — a bot flagged "PR template not respected", "3 open PRs
from a new contributor", "large PR needs prior discussion" — and the author closed it six hours
later. No human reviewed it.

It builds a SEPARATE draft model. **We are not doing that.** Use it for the tensor-name mapping
and for the two hard-won runtime insights (the `llama_memory_seq_rm` stale-K/V fix and reading
shards from disk rather than trusting `index.json`), and ignore its architecture.

## 4. Source material

The full original: `deepseek-ai/DeepSeek-V4-Flash-0731`, 48 shards, 166.89 GB, 72,317 tensors.
`model-00046/47/48` (10.86 GB) are 100% `mtp.*`; the other 45 are the main stack. All of it is
needed for a whole-model conversion.

`/home/kmbandy/models/DeepSeek-V4-Flash-DSpark-src/` already holds the three MTP shards from
tonight — reuse them, download only the remaining 45.

Disk on mad-lab-main: ~222 GB free before the download. 156 GB of source plus a 167 GB output
does NOT fit alongside everything else. **Check free space and report the plan before
downloading.** Do not start a 156 GB download that fills the disk.

## 5. Correctness gate

```
python3 gguf-py/gguf/scripts/gguf_dump.py <model.gguf> | grep -E "markov_w1|markov_w2|conf_proj"
```

Three tensors, `markov_w1` shaped `{256, 129280}` = `{dspark_markov_rank, n_vocab}`. Plus:

- `block_count` = 46 (43 main + 3 MTP), not 43
- tensor-type histogram shows **F8_E4M3 dense, not BF16**, and MXFP4 experts
- total size ~167 GB, not ~172 GB

`dflash.cpp` reads `markov_w1.weight` behind an `if (markov_meta)` guard, so a missing head
FAILS SILENTLY into plain DFlash — no error, only an absence. That is the failure mode this
gate exists to catch.

## 6. Constraints

- **NO INFERENCE, NO GPU.** Conversion and build only. All GPU work is run by the interactive
  Claude session, which holds the board claims.
- mad-lab-main only. Do not ssh to mad-lab-2026.
- **No git operations.** The tree carries another session's DSWS spike under
  `ggml/src/ggml-cuda/aiter-integration/` — do not touch it.
- **No detached background jobs.** No `setsid nohup ... &`. A rejected command must actually
  stop; earlier today a backgrounded download survived its own rejection, ran 45 minutes,
  wrote 71 GB and OOM-killed the user's browser.
- Build all four targets: `llama llama-server llama-wp-expert-worker test-wp-expert-worker`.
- STOP AND REPORT on any design question — F8_E4M3 kernel support, the dflash.cpp backbone
  question, disk space. Those are mine to answer.
