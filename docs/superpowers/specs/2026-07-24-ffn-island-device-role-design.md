# FFN-Island Device Role — shared experts and router on the second GPU

**Date:** 2026-07-24
**Status:** design / pre-build. Gating validation in §7.
**Scope:** increment A of Phase 3 of `docs/dev/2026-07-21-tiered-dual-gpu-expert-feeding-design.md`.
Hot-expert pinning and per-token intra-layer dispatch are **out of scope** here; see §9.

---

## 1. Goal

Give the weight pager a third device role. Today it knows a **paging** device (routed
experts, NVMe-fed VRAM pool) and one or more **resident** devices (attention, KV, dense,
lm_head). This adds an **FFN-island** device that owns the shared experts and the FFN
island — `ffn_*_shexp`, `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b`,
`ffn_gate_tid2eid`, `hc_ffn_*` — so that the always-active shared-expert GEMV computes on
a second GPU in parallel with the paging device's routed-expert compute.

On mad-lab-main the intended layout is: R9700 = paging device (cold routed experts),
6900XT = resident device (attention + KV) **and** FFN-island device.

The roles are independent by construction. The island defaults to the first resident
device but is named explicitly, so it can later move to a different card without rework.

## 2. Why this is worth building, stated honestly

Measured from the GGUF (`~/models/ds4/DeepSeek-V4-Flash-Q8-MTP`, 44 layers, 256 experts,
6 used, n_embd 4096):

| group | size |
|---|---|
| shared experts `ffn_*_shexp` | **1.03 GiB** |
| FFN island (norm, gate_inp, exp_probs_b, hc_ffn…) | **0.06 GiB** |
| routed experts `ffn_*_exps` | 264 GiB |
| **one routed expert** (up+gate+down) | **25.2 MiB** |
| attention / other | 5.38 GiB |
| token_embd | 0.49 GiB |

So this increment moves **1.09 GiB** of weights. Two effects, both modest and neither
oversold:

1. It frees ~1.1 GiB on the paging device for a larger cold pool (against a ~23 GiB pool,
   marginal).
2. It moves ~1.03 GiB/token of shared-expert GEMV onto the otherwise-idle second card —
   roughly **13%** of the ~6.6 GiB/token routed-expert bandwidth, *if and only if* the two
   devices' compute actually overlaps (§7).

This is not the aggregate-bandwidth unlock the tiered design is ultimately after. Its
strategic value is that it lands the three-role placement plumbing and the VRAM budget
accounting that hot-expert pinning will reuse, and it answers the overlap question
cheaply, before the expensive half is built.

### 2a. Correction to the 2026-07-21 design's sizing

That doc's §3 hot-set coverage table ("8 GB → 31% coverage", "~15 GB for the 44% balance
point") assumes **~13.4 MB per expert**. The measured Q8 expert is **25.2 MiB** — about
2× larger. An 8 GB hot set therefore holds ~318 experts, not ~600, and the coverage at any
given VRAM budget is materially lower than that table states. The "44% balance point"
would need roughly 30 GB, which the 6900XT does not have after attention and KV. This
does not affect the present increment, but it must be re-derived before any hot-set sizing
decision. It compounds the already-open §7.1 caveat that coverage was measured on a single
non-representative prompt.

## 3. Non-goals

- No hot/highly-reused routed experts pinned on the second card. §9.
- No per-token intra-layer split of a layer's routed experts across devices. §9.
- No change to pager behavior, page catalog, eviction, prefetch, or HostTier. The pager
  continues to see exactly the routed-expert tensors it sees today.
- No change to any default. With the flag unset, output must be byte-identical.

## 4. Placement model

`build_router_overrides` currently emits, in first-match-wins order: routed experts →
paging; shared experts → paging; FFN island → paging; `token_embd` → CPU; user overrides;
dense catch-all → resident.

The change: when an island buft is supplied, the **shared-expert** and **FFN-island**
patterns point at it instead of the paging device. Everything else is untouched. The
expert and shared-expert patterns are already disjoint (`_exps\.` vs `_shexp\.`), so
relative ordering is unaffected.

When no island buft is supplied the emitted list must be byte-identical to today's.

## 5. Units

Five units, each independently reviewable.

1. **Island device selection.** A sibling to the existing `wp_select_paging_device_index`
   and `wp_select_resident_device_indices` in `src/llama-model.cpp`. Given the device list
   and the already-resolved paging and resident indices, resolve an explicit device name,
   `auto` (→ first resident device), or nothing. Must reject an island that resolves to the
   paging device, since that is the current behavior and the role would be meaningless.

2. **Router override construction.** `wp::build_router_overrides` takes an optional island
   buft and routes the two pattern groups to it. Pure function; no device or GPU
   dependency; fully unit-testable.

3. **Preflight VRAM accounting.** Before committing the placement, sum the island tensors'
   bytes from the model loader and compare against free VRAM on the target minus a reserve
   for KV and compute buffers. The reserve is a single explicit value with a documented
   default, overridable by env, not an implicit fudge factor buried in the comparison. On
   failure, log loudly and place the island back on the paging device. This is nearly moot
   for this model at 1.09 GiB, and exists to protect other models and smaller cards.

4. **Flag and params plumbing.** One CLI flag plus an env override, defaulting to **off**.
   Registered for the same tools as the existing `--weight-paging*` flags (SERVER / CLI /
   PERPLEXITY — note `llama-completion` rejects those flags).

5. **Observability.** One load-time line reporting the resolved role per device and the
   byte total placed on each, so a later session can read the layout without re-deriving it.

## 6. Data flow, per MoE layer during decode

Attention produces the hidden state on the 6900XT. `ffn_norm` and the router
(`ffn_gate_inp` → logits → top-k) run there as well, so no transfer is needed. The shared
expert computes in place on the 6900XT. The normed activation and the `ids` tensor cross to
the R9700, where the pager ensures pages, arms the routed-expert pointer channel, and the
three `MUL_MAT_ID` nodes execute; results are weighted and summed there. The routed result
crosses back and is added to the shared-expert output into the residual.

Transfer count is **unchanged** from today: the activation crosses to the paging device and
the result crosses back, exactly as now, plus a ~24-byte `ids` transfer. Today the norm and
router simply happen on the far side of the same crossing. This increment does not add TB3
round trips; it relocates which side of the existing crossing the router and shared expert
sit on.

## 7. The gating validation: does the compute actually overlap?

Established from `ggml/src/ggml-backend.cpp`:

- `ggml_backend_sched_compute_splits` is a single sequential CPU loop over splits, but it
  submits compute with `ggml_backend_graph_compute_async` and inserts **no barrier between
  consecutive splits on different backends**. Overlap is therefore structurally possible.
- Cross-backend split inputs are copied with `ggml_backend_cpy_tensor_async` when the
  backend pair supports it. When it does not, the code falls back to a **CPU-blocking**
  `ggml_backend_synchronize` of the producer backend (ggml-backend.cpp:1788).
- The blocking MoE ids read at ggml-backend.cpp:1701/1720 is **not** implicated here: it is
  guarded on the expert weights living in a **host** buffer with `USAGE_WEIGHTS`
  (ggml-backend.cpp:1692-1694), the CPU-offload path. Our routed experts are in the paging
  device's VRAM, so that branch does not fire.
- Splits are emitted by a linear scan over the node array in graph order, with no
  dependency-aware reordering, so two independent subgraphs overlap only if their nodes are
  contiguous in that order.

**Therefore the increment's entire value rests on one empirical question: does
`cpy_tensor_async` work between the R9700 and the 6900XT across Thunderbolt 3?** If peer
async copy is unsupported for that pair, every cross-device input copy degrades to a
CPU-blocking synchronize, the two cards serialize, and this increment costs transfers while
buying nothing.

This must be answered on hardware **before** the hot-expert half is designed, and it is a
legitimate reason to stop after this increment. Any GPU run requires kmbandy's explicit
go-ahead.

## 8. Failure handling and verification

Every failure path degrades to today's behavior; none may fail the load.

- Flag unset → byte-identical override list. Default.
- Named device does not resolve, or resolves to the paging device → log, disable the role.
- Island bytes exceed the target's free VRAM minus reserve → log, fall back to paging.
- Island device is also a resident device → supported; this is the default arrangement.

Verification in three tiers:

1. **No GPU.** `build_router_overrides` is pure. A unit test pins the flag-off output as
   byte-identical to the current output, and checks the flag-on output routes exactly the
   shexp and island patterns to the island buft with ordering preserved. This is the real
   correctness gate for the placement logic.
2. **Correctness on hardware.** Greedy decode of the established prompt must stay coherent,
   then a wikitext perplexity run compared against the known paged baseline of **4.1524**.
   A placement bug is the silent-wrong-weights class, so perplexity is the arbiter, not
   eyeballed output.
3. **Performance.** Decode t/s at a fixed pool size, flag off vs on, interleaved twice,
   following the `~/host_cache.sh` convention already staged for this kind of A/B. This is
   also where §7 is answered.

## 9. Deferred: the hot-expert half

Pinning highly-reused routed experts on the second card and computing them there requires
splitting one layer's routed compute across two devices per token. Two structural
obstacles, both established from the code:

- A layer's experts are **one consolidated 3D tensor** (`ne[2]` = expert index), and
  `ggml_cuda_mul_mat_id` asserts against split buffers, so two devices cannot hold views
  into different expert slices of the same tensor.
- The routed-expert pointer side channel is a **single thread-local slot**
  (`ggml/src/ggml-cuda/mmq.cu:23`), taken once by the next MMQ launch (`mmq.cu:270`). Two
  concurrent device dispatches would race on it; it would have to become per-device or
  per-stream.

Neither is fatal, and both are cheaper to judge once §7 is answered and the hot-set
coverage curve has been re-derived against the corrected 25.2 MiB expert size.

## 10. Build hazard

`common/arg.cpp` currently holds **uncommitted pre-existing edits** belonging to another
line of work, as do `tools/server/server-models.{cpp,h}` and
`docs/examples/router-fleet-main.ini`. The flag plumbing touches `common/arg.cpp`. Edits
there must be strictly additive, and unrelated hunks must not be reverted or swept into a
commit.
