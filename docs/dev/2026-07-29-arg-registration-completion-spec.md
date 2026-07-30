# Spec: register the weight-paging + pipeline args for LLAMA_EXAMPLE_COMPLETION

Repo: `/home/kmbandy/GitHub/llama.cpp` on **mad-lab-main**, `master` at
`9475687be` plus your two uncommitted changesets (multi-shard `wp-stage-split`
and the MTP-head ownership predicate). Keep both.

Tiny change, but it is a hard blocker: without it the GLM-5.2 pipeline
correctness gate cannot run at all.

## The defect

`tools/pipeline/pipeline.cpp:550` parses with `LLAMA_EXAMPLE_COMPLETION`:

```cpp
if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, nullptr)) {
```

Every weight-paging option in `common/arg.cpp` is registered only for
`{LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_PERPLEXITY}`:

```
--weight-paging, --weight-paging-slots, --weight-paging-prefetch,
--weight-paging-resident-device, --weight-paging-ffn-island-device,
--weight-paging-resident-experts, --weight-paging-device-layers,
--pipeline-layers
```

`add_opt` (`common/arg.cpp:1211`) only pushes an option into `ctx_arg.options`
when `arg.in_example(ex)`, so for COMPLETION these options are never registered.
Consequence: `llama-pipeline --weight-paging` is rejected, and because the option
object is never added, **its `set_env` is never consulted either** -- so
`LLAMA_ARG_WEIGHT_PAGING=1` is not a workaround. Confirmed: 411 lines of
`llama-pipeline --help` contain no "paging".

This matters because a 2-stage split of a 254 GB model always leaves one stage
far larger than the 46 GB of VRAM on this box. Paging is mandatory, not optional.

Note this is the third instance of this same gap: `llama-completion` also lacks
`--weight-paging` (which is why `llama-server` was the only usable harness for
tonight's A/Bs), and now `llama-pipeline`. Consider whether these options belong
on a shared example set rather than an enumerated list that each new tool has to
be added to by hand -- but keep the change minimal if a broader refactor would
touch unrelated tools.

## What to change

1. Make the eight options listed above available to `LLAMA_EXAMPLE_COMPLETION`,
   so `llama-pipeline` (and `llama-completion`) accept them. Do not remove any
   existing example.
2. `tools/pipeline/README.md` documents `--pipeline-layers FIRST-LAST` as a
   supported way to set the band. That flag is currently unreachable for this
   binary; after the change it must actually work. If you conclude the README is
   the thing that is wrong instead, say so rather than silently changing one to
   match the other.

## Invariants

- Behaviour for SERVER / CLI / PERPLEXITY is unchanged.
- No option's semantics, default, or env name changes -- this is registration
  scope only.
- Do not touch `LLAMA_EXAMPLE_DOWNLOAD` handling (`add_opt`'s `inherit_common`
  special case).
- `llama-pipeline`'s manual pre-parse of `--pipeline-peer` / `--pipeline-listen`
  (`tools/pipeline/pipeline.cpp:79`, stripped from argv before
  `common_params_parse`) must keep working exactly as-is.

## Acceptance -- paste output

CPU build is enough to prove registration:

```
cmake --build build-cpu --target llama-pipeline -j 12
./build-cpu/bin/llama-pipeline --help 2>&1 | grep -c weight-paging
./build-cpu/bin/llama-pipeline --help 2>&1 | grep -E "weight-paging|pipeline-layers"
```

Required:

- all eight options appear in `llama-pipeline --help`
- `llama-cli --help` and `llama-server --help` still show them (no regression)
- state whether any tool's help gained an option it should not have

Do **not** run `llama-pipeline` against a model -- that is a GPU/model run and
belongs to the interactive session, which holds the board claims.

## Constraints -- hard

- **Do NOT touch a GPU. Do NOT run any inference or load any model.** The
  interactive Claude session holds claims `31850c89` (R9700) and `83b8aa94`
  (RX 6900 XT) and is about to run the gate.
- CPU-only `build-cpu`. Do **NOT** touch `build-hip`, `build-vk`, `build-army` --
  `build-hip` was just rebuilt for the gate and must not be disturbed.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Leave everything in the working tree, including your two prior changesets.
- Do NOT touch `ggml/src/ggml-cuda/aiter-integration/`.
- Do NOT run `npx gitnexus analyze`. Do NOT restart any service.
- ASCII only.
- Out of scope: the nextn graph, `TENSOR_SKIP` in `glm-dsa.cpp`, the `PIPE_TOKEN`
  hidden-state field, speculation wiring.

## Report back

- Which options you exposed and how.
- Whether you judged the README or the code to be at fault for
  `--pipeline-layers`, and why.
- The acceptance output verbatim.
- Anything you could not verify without running a model.
