# Spec: put the MTP/NextN layer on the HEAD stage, not the tail

Repo: `/home/kmbandy/GitHub/llama.cpp` on **mad-lab-main**, `master` at
`9475687be`, plus your uncommitted multi-shard `wp-stage-split` work from the
previous handoff (keep it, build on it).

Small, self-contained change. It gates writing the real GLM-5.2 stage files, so
it is the critical path.

## Why

GLM-5.2's MTP block (`blk.78`) is a full MoE transformer layer (3.498 GB of its
own 256 experts + 257.7 MB dense) plus the nextn glue. Its head is **tied**: the
GGUF has **no** `blk.78.nextn.embed_tokens` and **no**
`blk.78.nextn.shared_head_head` (both are declared `TENSOR_NOT_REQUIRED` in
`src/models/glm-dsa.cpp` and are absent from the file). So running MTP needs the
model-level `token_embd` **and** `output`.

`llama_pipeline_owns_tensor()` currently sends any `blk >= n_layer` tensor to the
tail, reasoning that the final hidden state lives there. For our deployment we
want MTP on the **head**:

- Head is mad-lab-main: R9700 32 GB + RX 6900 XT 16 GB, a 6.25 GB/s drive, and a
  much larger pager pool. The tail is mad-lab-2026: two 8 GB cards and a
  3.08 GB/s drive. MTP's 3.5 GB of experts should page from the fast side.
- The head already owns `token_embd`, so MTP's embedding input is free there.
  Hosting MTP on the tail would instead require duplicating `token_embd`
  (654 MB); hosting it on the head requires duplicating `output` (535 MB).
  Cheaper, and it keeps the driver/prefill on the strong GPUs.
- The hidden state MTP consumes travels tail -> head on the **existing**
  `PIPE_TOKEN` frame (`src/pipeline/pipe-protocol.h:109`), which already flows
  every decode step. No new round trip. (Actually carrying it is a LATER
  handoff -- not this one.)

## What to change -- behaviour only

In `llama_pipeline_owns_tensor()` (`src/llama-pipeline.cpp:68`):

1. **Tensors of the MTP/NextN layer (`blk >= n_layer`) belong to the head**
   (`first == 0`), not the tail. Replace the existing rule and replace its
   comment with one that states the new reasoning -- do not leave the old
   justification in place describing behaviour that no longer happens.
2. **`output.weight` and `output_norm.weight` must be owned by the head as well
   as the tail, when the model has an MTP layer.** The tail still needs them to
   sample the base model; the head now needs them to turn MTP's hidden state into
   draft logits. This is a deliberate duplication, not a move.
3. When the model has **no** MTP layer, behaviour must be **exactly** as today:
   `output*` tail-only, no duplication.

`llama_pipeline_owns_tensor` does not currently receive enough information to
know whether the model has an MTP layer. Adding it is your call (an extra
parameter, or deriving it from something already passed) -- but see the invariant
below about call sites.

## Invariants -- these are the failure modes that matter

- **The splitter's predicate and the loader's predicate must remain the same
  function producing the same answer.** Both `wp-stage-split-lib.cpp` and
  `src/llama-model.cpp` call it. If you change the signature, every call site
  must pass a consistent value. A splitter that writes a different tensor set
  than the loader expects yields a stage that loads and silently computes the
  wrong thing -- worse than a crash.
- `llama_pipeline_validate_stages()` must still accept the two-stage
  `[0,54] + [55,77]` configuration over `n_layer = 78`. The duplication is of
  global tensors, not layers; layer coverage stays a strict partition with no
  gaps or overlaps.
- Existing `token_embd` semantics are unchanged (`first == 0`, or the
  `duplicated_embd` tied-tail fallback).
- **Your previous handoff's partition check will now fail by design** -- the
  tensor set is no longer strictly disjoint, because `output*` is selected by
  both stages. Update that verification to assert "a partition of layer tensors,
  plus an explicitly reported set of intentionally duplicated global tensors",
  and have it print the duplicated names. Do not weaken it into something that
  would no longer catch a real double-selection.
- `tests/test-wp-stage-split.cpp` and `tests/test-wp-repack` must pass.

## Acceptance -- run these, paste output verbatim

```
cmake --build build-cpu --target llama-wp-stage-split -j 12
```

With `M=/home/kmbandy/models/GLM-5.2/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf`:

```
./build-cpu/bin/llama-wp-stage-split --model $M --first 0  --last 54 --dry-run
./build-cpu/bin/llama-wp-stage-split --model $M --first 55 --last 77 --dry-run
```

Required, and state explicitly whether each holds:

- head `[0,54]` now contains **all** `blk.78.*` tensors -- the four `nextn.*`
  AND the block body (`attn_*`, `ffn_*_exps`, `ffn_*_shexp`, `indexer.*`,
  norms, `exp_probs_b`) -- 27 tensors totalling 3.756 GB
- head also contains `token_embd.weight`, `output.weight`, `output_norm.weight`
- tail `[55,77]` contains `output.weight` and `output_norm.weight`, and **no**
  `blk.78.*` tensor
- layer tensors partition exactly; the only duplicated names are the `output*`
  pair, reported explicitly
- previous run for reference: head 161.00 GiB / 1251 tensors, tail 75.44 GiB /
  558 tensors, 1809 total. Expect head to grow by ~3.756 GB plus the `output*`
  pair and tail to shrink by ~3.756 GB. Report the actual numbers; if they do not
  move in that direction, something is wrong -- say so rather than adjusting.
- a single-file model still dry-runs unchanged (regression)

## Constraints -- hard

- **Do NOT touch a GPU. Do NOT run any inference of any kind.**
- **Do NOT write real stage files.** `--dry-run` only; the interactive Claude
  session writes them.
- CPU-only `build-cpu`. Do NOT touch `build-hip`, `build-vk`, `build-army`.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Leave everything in the working tree, including your previous multi-shard work.
- Do NOT touch `ggml/src/ggml-cuda/aiter-integration/`.
- Do NOT run `npx gitnexus analyze`. Do NOT restart any service.
- ASCII only.
- **Out of scope, do not start:** the nextn graph, dropping `TENSOR_SKIP` in
  `glm-dsa.cpp`, the `PIPE_TOKEN` hidden-state field, and any speculation
  wiring. Those are separate handoffs. This one is the ownership predicate and
  its verification only.

## Report back

- The rule you implemented and how the "does this model have an MTP layer"
  question is answered at each call site.
- How you kept the splitter and loader predicates in agreement.
- The two dry-run outputs verbatim, plus the duplicated-name report.
- Test results.
- Anything you could not verify without writing a stage or running a model.
