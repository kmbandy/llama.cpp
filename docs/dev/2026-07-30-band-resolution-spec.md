# Spec: resolve a stage's band from the stage GGUF, not only from the CLI

Repo: `/home/kmbandy/GitHub/llama.cpp` on **mad-lab-main**. Keep every prior
uncommitted changeset in the working tree.

Small fix. It removes a landmine: today a stage launched without
`--pipeline-layers` fails with an error that points nowhere near the cause.

## The defect

`tools/pipeline/pipeline.cpp` (~line 596) resolves the band like this:

```cpp
// resolve the band: explicit flag, else stage-GGUF metadata (already
// adopted into mparams by the loader), else the full range.
band = llama_pipeline_resolve_band(
    params.pipeline_layer_first, params.pipeline_layer_last, s.n_layer);
```

The comment describes three sources. The code reads exactly one: the CLI values
in `common_params`. The stage-GGUF band that the loader adopts goes into
**`mparams`** (the model params) -- a different struct -- and `pipeline.cpp`
never consults it. So with no `--pipeline-layers`, `params.pipeline_layer_*` stay
unset and `resolve_band` returns the FULL range.

Observed on GLM-5.2 with correct stage files whose metadata was adopted fine:

```
load_hparams: pipeline band adopted from GGUF metadata: layers [55, 77]   <- loader: right
pipe: tail stage listening, band [0, 77] of 78 layers                     <- pipeline.cpp: wrong
pipe: HELLO rejected: pipeline: layer band [78, 77] is empty (first > last)
```

The `[78, 77]` is the complement computed from the bogus full band: a stage that
believes `s.first == 0` computes its peer as `{s.last + 1, n_layer - 1}` =
`{78, 77}`. The handshake error is real but three steps downstream of the cause,
which is what makes this expensive to debug.

## What to change

Make the resolution actually match its comment: **explicit `--pipeline-layers`
wins; otherwise use the band the model adopted from the stage GGUF; otherwise the
full range.** The model already exposes this -- see `llama_model::pipeline_band_enabled()`,
`pipeline_layer_first()`, `pipeline_layer_last()` in `src/llama-model.cpp` (~2553-2566),
and note there is already a C accessor convention in `include/llama.h` for
model-level queries if one is needed.

Also fix the pre-load `band_from_cli` warmup decision. Right now
`tools/pipeline/pipeline.cpp` (~561) computes:

```cpp
const bool band_from_cli = llama_pipeline_band_enabled(
    params.pipeline_layer_first, params.pipeline_layer_last);
if (band_from_cli && params.pipeline_layer_first != 0) { params.warmup = false; }
```

so a non-head stage relying on GGUF metadata does **not** get `--no-warmup`
forced, and there is a later warning acknowledging exactly that gap ("non-head
stage adopted from GGUF ran a token warmup; this should have failed by design").
That ordering problem is real: the band is only known after load, but warmup
happens during load. Handle it however is cleanest -- either force warmup off
before load whenever the model file declares a non-zero `pipeline.layer_first`,
or keep the post-hoc warning but make it accurate. Say which you chose and why.

## Invariants

- Explicit `--pipeline-layers` must still win over the GGUF metadata.
- A model with no band metadata and no flag must still resolve to the full range,
  i.e. the legacy single-process path is unchanged.
- The band that `pipeline.cpp` uses for role selection, the complement it
  computes for the peer, and the band the loader adopted must all agree. A
  disagreement here is what produced the `[78, 77]` above.
- Do not change `llama_pipeline_resolve_band`'s own semantics for the arguments
  it is given.

## Acceptance

CPU build, then demonstrate by reading the code (do NOT run a model) that:

- `--pipeline-layers 55-77` behaves as today
- a stage GGUF carrying `pipeline.layer_first=55 / layer_last=77` with NO flag
  now resolves to `[55, 77]` rather than `[0, 77]`
- a plain single-file model with neither resolves to the full range

```
cmake --build build-cpu --target llama-pipeline -j 12
```

If a unit test can cover the resolution precedence without loading a model, add
one; if not, say why.

## Constraints -- hard

- **Do NOT touch a GPU. Do NOT run any inference or load any model.** The
  interactive session holds the board claims and runs the gate.
- CPU-only `build-cpu`. Do **NOT** touch `build-hip`, `build-vk`, `build-army`.
- **Do NOT commit, stash, revert, `git checkout`, `git reset`, `git add -A`.**
  Preserve all prior uncommitted changesets.
- Do NOT touch `ggml/src/ggml-cuda/aiter-integration/`.
- Do NOT run `npx gitnexus analyze`. Do NOT restart any service.
- ASCII only.
- Out of scope: the nextn graph, `TENSOR_SKIP` in `glm-dsa.cpp`, the
  `PIPE_TOKEN` hidden-state field, speculation wiring.

## Report back

- How you obtain the adopted band and where precedence is decided.
- What you did about the warmup ordering problem, and why.
- The three acceptance cases, argued from the code.
- CPU build output, and anything unverifiable without a model run.
