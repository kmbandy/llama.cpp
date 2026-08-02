# DS4 DSpark: dual-path draft head (in-model + sidecar)

Spec/design: Claude (mad-lab-2026 session 87d16c2e), 2026-08-02.
Implementation: Codex gpt-5.6-luna.
Review + all builds + all measurement: Claude.

---

## 0. Goal

Make DSpark speculative decoding work for DeepSeek-V4-Flash-0731 **with the
DSpark head living inside the target GGUF**, while keeping upstream's
**sidecar** path fully working for externally-trained heads (Kimi-K3 and
community heads ship as sidecars).

Phase 1 — the scope of this handoff — is exactly:

> `--spec-type draft-dspark` with no `-md` produces one correct draft token
> from the in-GGUF DSpark head, and the dispatcher issues at least one expert
> request for layer 43.

Nothing else. Explicitly **out of scope**: the confidence scheduler, the
cumulative-product gate, STS calibration, prefetch, and any tuning.

## 1. THE OVERRIDING CONSTRAINT: upstream syncs must stay cheap

This fork is ~977 commits ahead of upstream and syncs regularly. The single
most important property of this change is that it **does not turn future
merges of `src/models/deepseek4.cpp`, `src/models/dflash.cpp`,
`common/speculative.cpp` or `include/llama.h` into a nightmare.**

Rules, in priority order:

1. **Every hunk added to an upstream-owned file must be marked.** Open with
   `// MAD-LAB: <one-line why>` and, if longer than ~5 lines, close with
   `// MAD-LAB: end`. A future sync must be able to `git grep -n "MAD-LAB:"`
   and see the complete set of fork deltas in these files. This is not
   decoration — it is the mechanism.
2. **Prefer adding over editing.** New helper functions, new subclasses, new
   files. Never restructure an upstream function to accommodate us.
3. **Never reformat, reorder, rename, or re-indent upstream code.** Not even
   to fix obvious style. Whitespace churn is what makes merges unresolvable.
4. **Keep per-file hunk COUNT low, even at the cost of a slightly larger
   single hunk.** Three scattered one-line edits conflict three times; one
   six-line block conflicts once.
5. **Append to enums, never insert.** Renumbering breaks ABI and every
   downstream switch.
6. **Default-valued parameters over new overloads** when extending an
   upstream signature — the existing call sites stay byte-identical, so they
   never conflict.

If you find yourself wanting to refactor an upstream function so the new code
fits nicely: don't. Duplicate a little, mark it, and note it in your report.
Say so explicitly rather than doing it.

## 2. State of the tree (verified, do not re-derive)

`mad-lab-main:/home/kmbandy/GitHub/llama.cpp` — the upstream merge is **DONE**.
Both boxes are at `1bb65b7d9` ("Merge upstream/master (through bb4e0e1b3)"),
clean trees, 0 behind upstream / 989 ahead. `libllama`, `llama-server` and
`llama-wp-expert-worker` all build and link.

If `git log --oneline -1` does not show `1bb65b7d9`, stop and report rather
than proceeding — someone has moved the tree under you.

Do not merge, rebase, or cherry-pick anything.

The tree contains upstream's:

- `src/models/dflash.cpp` — `llama_model_dflash::graph_dsv4`, the DSV4-flavoured
  DSpark draft graph (3 full DSV4 stages: hc + MLA + MoE, then markov +
  confidence heads via `build_dspark_markov_head`). Selected by
  `hparams.dsv4_hc_mult > 0`.
- `src/models/models.h:~1295` — `struct graph_dsv4 : public llama_model_deepseek4::graph`
- `src/models/deepseek4.cpp` — `load_arch_hparams` with the `n_layer_nextn`
  probe, `n_layer_all` handling, `graph_mtp`, `dsv4_hc_mean`, the recurrent
  state restore/snapshot machinery.
- `include/llama.h:217` — `LLAMA_CONTEXT_TYPE_DEFAULT = 0`, `..._MTP = 1`
- `common/speculative.cpp` — `common_speculative_init_result` ctor with
  `GGML_ASSERT(has_draft || spec_mtp)`, and
  `common_speculative_impl_draft_dflash` with `is_dspark`.

Facts about our model, measured from the GGUF — take these as given:

```
deepseek4.block_count           46      = 43 trunk + 3 DSpark stages
deepseek4.nextn_predict_layers  3
deepseek4.hash_layer_count      3
deepseek4.block_size            5
deepseek4.markov_rank           256
deepseek4.target_layers         [41, 42, 43]
blk.43/44/45   attn_* + hc_* + ffn_norm + ffn_gate_inp + exp_probs_b + *_shexp
blk.45  ALSO   nextn.hc_head_{fn,base,scale}, nextn.shared_head_norm
root           fc.weight, enc.output_norm.weight, markov_w1, markov_w2,
               conf_proj.weight, output.weight, token_embd.weight
```

**There is no `nextn.eh_proj`, `nextn.enorm` or `nextn.hnorm` anywhere in this
GGUF.** Those are MTP tensors; DSpark stages do not have them. This matters —
see §3.1.

Expert shards already cover blocks 0..45 on both sides (46 shards each,
`/mnt/nvme/models/DS4-eshard` on 2026, `~/models/DS4-eshard-main` on main). No
re-sharding. Do not touch the shards or the converter.

## 3. Design — five seams

### 3.1 `deepseek4.cpp` — the nextn probe must accept a DSpark head

Upstream `load_arch_hparams`:

```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
if (hparams.n_layer_nextn > 0 && hparams.n_layer_nextn < hparams.n_layer_all) {
    const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
    const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
    if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
        hparams.n_layer_nextn = 0;
    }
}
```

On our GGUF the probe fails, `n_layer_nextn` becomes 0, `n_layer()` becomes 46,
and the trunk runs the three DSpark stages as ordinary layers. **It fails
silently into a plausible-looking model** — wrong output, and `compress_ratios`
indexing shifts too.

**ALREADY DONE IN THE MERGE — DO NOT REDO.** The *tensor loader* half of this
same bug is fixed: `layer.nextn.eh_proj/enorm/hnorm` are now
`TENSOR_NOT_REQUIRED | flags` in `load_arch_tensors`, because upstream marked
them required and our DSpark stages have none of them. Leave that alone.

**STILL TO DO — the probe itself, verified unfixed at `deepseek4.cpp:55`.**
The probe must accept *either* head. Add a file-static helper above
`load_arch_hparams` and change exactly one condition:

```cpp
if (ml.get_weight(mtp_probe.c_str()) == nullptr && !dsv4_has_dspark_head(ml)) {
```

`dsv4_has_dspark_head` returns true when `markov_w1.weight` is present (that
tensor exists only on a DSpark-bearing checkpoint; the name is confirmed at
`src/llama-arch.cpp:627`, `{ LLM_TENSOR_DSPARK_MARKOV_W1, "markov_w1" }`).
Keep it to one condition change plus one added static function, both marked.

Add an `LLAMA_LOG_INFO` line reporting which head was detected and the
resulting `n_layer_nextn` / `n_layer()`. A silent mis-detection here is the
single most expensive failure mode in this whole change; make it loud.

### 3.2 `graph_dsv4` gains a stage base + count (defaulted)

`graph_dsv4` walks `model.layers[il]` for `il` in `0..n_layer-1`. That is
correct for a sidecar, where the stages *are* the model. In-model, the same
stages are `model.layers[43..45]` of the target.

Extend the constructor with **defaulted** parameters so every existing call
site stays byte-identical:

```cpp
// models.h
graph_dsv4(const llama_model & model, const llm_graph_params & params,
           int stage_base = 0, int n_stages = 0);   // MAD-LAB: in-model stages
```

`n_stages <= 0` means "use `n_layer`" — i.e. exact current behaviour.

Inside the ctor, introduce **one** alias at the top of the stage loop and use
it for tensor lookups:

```cpp
const int n_st = n_stages > 0 ? n_stages : n_layer;
for (int il = 0; il < n_st; ++il) {
    const int il_m = stage_base + il;          // MAD-LAB: in-model stage offset
    const auto & layer = model.layers[il_m];
    ...
```

Be careful to distinguish the two indices:
- **`il_m`** — indexes `model.layers[]` and the `cb()` debug label.
- **`il`** — the *position within the draft block*. Anything that indexes a
  per-draft-layer structure (the draft KV ring / `inp_attn` cache slots) must
  keep using `il`, because the draft cache has `n_stages` layers, not 46.

Getting this wrong will not crash; it will read the wrong KV. Please state
explicitly in your report which index you used at each site and why.

### 3.3 New context type

`include/llama.h`, **append only**:

```cpp
LLAMA_CONTEXT_TYPE_DSPARK = 2,   // MAD-LAB: in-model DSpark draft context
```

and the matching `llm_graph_type` value (follow whatever pattern
`LLM_GRAPH_TYPE_DECODER_MTP` uses — append, do not insert).

`llama_model_deepseek4::build_arch_graph` dispatches it:

```cpp
// MAD-LAB: in-model DSpark draft graph -- reuse upstream's DSV4 DSpark body,
// pointed at this model's stage blocks instead of a sidecar's layers 0..N.
if (params.gtype == LLM_GRAPH_TYPE_DECODER_DSPARK) {
    return std::make_unique<llama_model_dflash::graph_dsv4>(
        *this, params, (int) hparams.n_layer(), (int) hparams.n_layer_nextn);
}
```

Note this deliberately reuses upstream's graph body rather than our
`529e18b73` implementation. Ours is superseded; do not port it.

### 3.4 `common/speculative.cpp` — generalise the self-draft gate

Today the ctor knows two worlds: a real draft file (`has_draft`) or MTP-on-
the-target (`spec_mtp`). Add a third that shares the second's shape. Keep the
edit to one contiguous marked region where possible:

```cpp
// MAD-LAB: DSpark may live INSIDE the target GGUF (DeepSeek ships it that way)
// or as a sidecar (Kimi-K3 and community heads). Sidecar => has_draft, handled
// by the existing branch. In-model => a second context on the target, exactly
// like MTP.
const bool spec_dspark = std::find(params.speculative.types.begin(),
                                   params.speculative.types.end(),
                                   COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params.speculative.types.end();
const bool spec_dspark_self = spec_dspark && !has_draft &&
                              llama_model_n_layer_nextn(model_tgt) > 0;
const bool spec_self = spec_mtp || spec_dspark_self;
```

- `GGML_ASSERT(has_draft || spec_mtp)` → `GGML_ASSERT(has_draft || spec_self)`
- ctx_type selection sets `_MTP` for `spec_mtp`, `_DSPARK` for `spec_dspark_self`
- the `else if (spec_mtp)` branch that builds a context from `model_tgt`
  becomes `else if (spec_self)`

**Precedence must be: an explicit `-md` sidecar always wins.** `spec_dspark_self`
is false whenever `has_draft` is true. Both paths must remain reachable; a
sidecar run must behave exactly as it does today.

### 3.5 Dispatcher: layers 43–45 have never been dispatched

Per-request traces from the 4.231 tok/s run show `layers seen: 0..42`. The
shard data for 43/44/45 exists and is correctly indexed, but no expert request
has ever been issued for those layers.

`src/pipeline/*` is **fork-owned** — no upstream conflict risk, so normal code
quality rules apply rather than the minimal-diff discipline above.

Do not redesign anything here. Audit only, and report:
- any place a layer index is bounded by `n_layer()` rather than `n_layer_all`
  in a way that would drop or misroute a layer-43 request;
- whether the worker's layer-set advertisement (`pipe_hello` expert payload)
  covers 43..45.

Fix only what is provably wrong. If it already works, say so — do not change
working code. (We have burned a handoff before on renaming a working safety
flag because a design doc asserted a bug that wasn't there. Read the full call
chain before concluding something is broken.)

## 4. Explicitly NOT in this task

- No confidence scheduler, no `∏cᵢ`, no STS, no prefetch, no protocol frame.
- No converter changes. No re-shard. No GGUF regeneration.
- No changes to `conf_min` semantics.
- Do not port anything from commit `529e18b73` (our superseded implementation).
- Do not merge, rebase, cherry-pick, or otherwise move branch state.

## 5. Hard constraints on the implementing agent

- **Do NOT run any GPU work or LLM inference.** Not `llama-cli`, not
  `llama-server`, not a worker, not a benchmark. Standing fleet rule, no
  exceptions.
- **Do NOT run builds.** Claude builds and measures. Write the code.
- **Do NOT run `git checkout/restore/stash/reset`, `git add -A`, or
  `git commit -a`.** The tree is shared and currently has ~31 dirty files
  belonging to other sessions. Stage nothing. Commit nothing. Leave your work
  as unstaged edits and list the files you touched.
- **Do not touch** `ggml/src/ggml-cuda/aiter-integration/` (another session).
- **Do not run `npx gitnexus analyze`.**

## 6. Acceptance criteria

Claude verifies all of these; you cannot.

1. Sidecar path unchanged: a `-md <dspark gguf>` run behaves byte-identically
   to pre-change. This is the regression that matters most.
2. In-model path: `--spec-type draft-dspark` with no `-md` creates a non-null
   `ctx_dft` and logs the detected head + `n_layer_nextn=3`, `n_layer()=43`.
3. One draft token emerges and is not garbage.
4. At least one expert request for layer 43 round-trips to a worker.
5. `git diff --stat` on the four upstream-owned files stays small, and
   `git grep -n "MAD-LAB:"` enumerates every fork delta in them.

## 7. What to report back

- Files touched, with a one-line why each.
- For §3.2: which index (`il` vs `il_m`) you used at each site, and why.
- Anything in §3.5 you found already correct (name it, so I don't re-audit it).
- **An explicit "could not verify" list.** You cannot build or run; say so
  plainly and enumerate what that leaves unproven. A previous handoff's honest
  could-not-verify list was the most useful part of its report — do that again.
- Anywhere you thought the design was wrong. Say so rather than working around
  it silently.
