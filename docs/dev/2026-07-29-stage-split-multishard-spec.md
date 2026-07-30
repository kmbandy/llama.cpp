# Spec: make `wp-stage-split` work on split (multi-shard) GGUF models

Repo: `/home/kmbandy/GitHub/llama.cpp` on **mad-lab-main**, `master` at
`9475687be` (both machines and origin converged). Work in the working tree.

You are unblocking the GLM-5.2 cross-machine bring-up. Nothing else in the
pipeline path is missing -- this one tool is the blocker.

## The defect

`wp_stage_split::split_stage()` (`tools/wp-stage-split/wp-stage-split-lib.cpp:60`)
does exactly one `gguf_init_from_file(model_path)` and later reads tensor payload
with a single `std::ifstream fin(model_path)`. It has no concept of a split GGUF
set.

GLM-5.2-UD-Q2_K_XL is 7 shards. Shard 1 is metadata-only (9.4 MB); all tensor
data lives in shards 2-7. So today every band selects nothing:

```
$ ./build-cpu/bin/llama-wp-stage-split --model .../GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf \
      --first 55 --last 77 --dry-run
wp-stage-split: error: band selects zero tensors; refusing to write an empty stage
```

That error is correct behaviour reporting an incomplete tool, not a bug in the
predicate. `llama_pipeline_owns_tensor()` is fine and must not change.

## What to change

Teach the tool to treat a split set as one logical model:

1. **Accept the first shard and enumerate the set.** Read `split.count`; if it is
   greater than 1, require that the input is `split.no == 0` and derive the
   sibling filenames from the `-%05u-of-%05u.gguf` suffix.
   `tools/wp-repack/wp-repack.cpp:212-238` already does precisely this, including
   refusing a non-first shard. **Follow that existing pattern rather than
   inventing a second convention.** Reuse it if you can do so without dragging
   wp-repack's other dependencies into this target; otherwise mirror its logic.
2. **Build a combined tensor view.** Every tensor must resolve to
   `(owning shard file, data offset within that shard)`. Selection stays
   per-tensor-name via `llama_pipeline_owns_tensor()` with the band and
   `tail_tied` computed exactly as now.
3. **`bytes_in` / `n_tensors_in` must sum over the whole set,** not one shard, so
   `--dry-run` reports the real model totals.
4. **Read payload from the owning shard** when writing. This is the substantive
   part: the current single `ifstream` becomes per-shard access.
5. **Single-file models must keep working unchanged.** A model with no
   `split.count`, or `split.count == 1`, must behave exactly as today.

Note you will be holding one `ggml_context` of metadata per shard; the selected
tensors passed to `gguf_add_tensor` come from whichever context owns them. How
you manage those lifetimes is your call.

## The trap that will silently corrupt the output

`gguf_set_kv(ctx_out, ctx_in)` copies **all** KV from shard 1, which for a split
model includes `split.no`, `split.count` and `split.tensors_count`. The stage you
write is a **single self-contained file**. If those keys survive into it, a loader
opening the stage will try to find shards of the stage and fail (or worse,
misinterpret it).

The output stage must not advertise itself as part of a split set. Decide whether
to strip those keys or normalise them, do it deliberately, and say in your report
which you chose and why.

Also keep the existing non-default-alignment refusal, and apply it to **every**
shard you read, not just shard 1.

## Invariants

- `llama_pipeline_owns_tensor()` is not modified. It already places NextN/MTP
  tensors on the tail (`llama-pipeline.cpp:68-82`) and that is correct for our
  split -- do not "fix" it.
- The selection predicate must stay the loader's predicate. Divergence between
  what the splitter writes and what the loader expects is the one failure mode
  that produces a stage that loads and computes the wrong thing.
- `--dry-run` writes nothing and must remain safe to run repeatedly.
- Refusing to overwrite an existing output file stays.
- `tests/test-wp-repack` must still pass; if a stage-split test exists it must
  still pass.

## Acceptance -- run these, paste the output

Configured CPU build already exists on main. Build only your target:

```
cmake --build build-cpu --target llama-wp-stage-split -j 12
```

Then, `M=/home/kmbandy/models/GLM-5.2/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf`:

```
./build-cpu/bin/llama-wp-stage-split --model $M --first 0  --last 54 --dry-run
./build-cpu/bin/llama-wp-stage-split --model $M --first 55 --last 77 --dry-run
```

Expected, from the model's own tensor tables (independently computed; treat as a
target to explain, not a number to hit by construction):

- model total ~253.9 GB across 79 blocks; `n_layer` resolves to **78**
- band `0-54`  -> ~173 GB, and must include `token_embd.weight`
- band `55-77` -> ~81 GB, and must include `output.weight`, `output_norm.weight`,
  and the `blk.78.nextn.*` tensors (blk 78 >= n_layer, so it lands on the tail)
- the two bands' tensor counts must sum to the model's total tensor count with no
  tensor selected twice and none dropped

A band that reports zero tensors, or bands that do not partition the set, means
it is not working -- report that rather than adjusting the numbers.

Also confirm a single-file model still splits: any small GGUF in
`/home/kmbandy/models/` is fine for a `--dry-run` regression.

## Constraints -- hard

- **Do NOT run any model, any inference, `llama-cli` / `llama-server` /
  `llama-completion` / `llama-perplexity`, and do NOT touch a GPU.** The
  interactive Claude session runs all GPU and model work; it holds the board
  claims and can see live services and VRAM headroom. You cannot.
- **Do NOT write the actual stage files.** `--dry-run` only. Writing 173 GB is
  the interactive session's job -- disk sequencing matters (main has 320 GB free
  and the two stages must be produced in a specific order).
- CPU-only: `build-cpu` target above. Do **NOT** touch `build-hip`, `build-vk`,
  `build-army`, or any other build dir.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Both machines and origin are at `9475687be`; a stray commit desynchronises
  them. Leave your changes in the working tree.
- Uncommitted DSWS kernel work under `ggml/src/ggml-cuda/aiter-integration/` is
  not yours -- do not touch it.
- Do **NOT** run `npx gitnexus analyze`.
- Do **NOT** restart any service. `llama-router.service` is live on both boxes,
  and mad-lab-2026 hosts the MCP, dashboard and mneme daemon.
- ASCII only in code and comments.

## Report back

- How you enumerate the shard set, and whether you reused wp-repack's helper or
  mirrored it (and why).
- What you did about `split.*` KV in the output, and how you verified the stage
  does not present itself as a shard.
- The four `--dry-run` outputs above, verbatim.
- Whether the two bands partition the tensor set exactly, and how you checked.
- Anything you could not verify without writing a stage or running a model. Do
  not describe untested code as working.
