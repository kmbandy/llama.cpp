# The distributed-MoE f16 partition bug, and the 16-run validation matrix

**Date:** 2026-08-04 (night)
**Status:** fixed, validated, uncommitted on both trees
**KG note:** the memory MCP was down when this was written — this file is the record of
record until it can be mirrored into the repo KG.

---

## 1. The headline

**Before the fix, the model had lost its own context.** Every prose run opened
`**Chapter 1**` with a story *unrelated* to the 663-token prompt. After the fix, every
prose arm opens `**Chapter 2**` and correctly continues the Kestrel/Aluna story with the
right characters, distinct-ratio 0.615–0.629, and **not one arm in 16 shows the CJK
(U+5768) collapse** that appeared four times earlier the same day.

This was a **quality defect that was invisible in every metric we track.** Throughput,
acceptance, and mean accepted length all looked plausible while the output was wrong.

## 2. The bug

Two independent width-dependent channels, compounding.

**(a) f16 subtotals at the partition boundary** — `wp-expert-worker.cpp`. Each worker
summed *its own subset* of a layer's experts in f32, then rounded **the whole subtotal**
to f16 for the wire; the spine converted back and added. The arithmetic was

```
sum_workers  f16( sum_experts_assigned_to_that_worker )
```

f16 has an 11-bit mantissa, so that placed a **~5e-4 relative error exactly at the
expert→worker partition boundary**.

**(b) history-dependent assignment** — `pipe-expert-dispatcher.cpp:choose_worker()` chose
from residency + in-request `assigned_counts` + a rotating `machine_cursor`, all of which
move with batch width and request history. **Three workers on three different backends**
(CUDA 1070, Vulkan RX480, CPU) all advertise experts 0..84, so the same expert could
execute on a different backend from one run to the next.

Our own `harvest_partials` comment already said *"Worker ASSIGNMENT is already
timing-dependent — ~35% of requests differ between identical runs"* and reasoned past it.

**Together:** changing the speculative draft length (`conf_min`) changed the verify batch
width → changed the expert union → moved an expert between workers → re-rounded two
subtotals → perturbed the layer output ~1e-3, which the hyper-connection gates and the
discontinuous router top-k amplified into a different trajectory, sometimes a repetition
attractor. Four orders of magnitude above ordinary f32 reordering noise (~1e-7).

## 3. The fixes

- `pipe_expert_partial::partial` is now `std::vector<float>`; `wr_f32`/`rd_f32`, length
  check 4 bytes/value. **`PIPE_VERSION` 1 → 2** so a stale binary *refuses* rather than
  mis-decoding a 2-byte stream as 4-byte values.
- `choose_worker()` returns `splitmix64(layer, expert) % candidates.size()` — a pure
  function. `WP_DISPATCH_STATIC_ASSIGN=0` restores the old balancing path, documented as
  **not reproducible**.

## 4. Validation matrix — 16 runs, 2 reps per config

| gate | result |
|---|---|
| rep-to-rep determinism (record) | **PASS** |
| conf 0.5 rep-to-rep | **PASS** |
| gather bypass vs record — text identity | **PASS** |
| conf_min invariance 0.9 / 0.5 / 0 | **FAIL** |
| ubatch 512 vs 2048 text identity | FAIL |
| overlap=1 rep-to-rep | FAIL |

**The conf_min prediction FAILED and that is the honest result.** I predicted the two
closed channels would make output invariant to `conf_min`; three configs gave three SHAs.
What remains is **inherent batch-width FP non-associativity in the spine's own GEMMs**
(M = n_draft+1 ∈ {2..6}), which no dispatcher fix can reach. The ubatch and overlap FAILs
are expected for the same reason / the untouched two-group bug.

### Numbers (prose739, config of record unless noted)

| config | prefill | decode | acceptance | mean len | text SHA |
|---|---|---|---|---|---|
| record ub2048 conf 0.9 | 19.20 / 20.45 | 2.49 / 2.34 | 0.67105 | 1.67 | `404c672900` |
| conf_min = 0.5 | 18.40 / 21.26 | 2.59 / 2.50 | 0.58163 | 1.58 | `e2b4814a3e` |
| conf_min = 0 | 21.04 / 20.64 | 1.85 / 1.85 | 0.12152 | 1.60 | `5baf18dd84` |
| UBATCH = 512 | 16.67 / 16.26 | 2.59 / 2.64 | 0.67045 | 1.68 | `0683e359ac` |
| **GATHER_MIN_TOKENS = 2** | 21.46 / 19.02 | **2.62 / 2.66** | 0.67105 | 1.67 | **`404c672900`** |
| WP_EXPERT_OVERLAP = 1 | 20.27 / 20.45 | 2.82 / 2.74 | 0.729 / 0.793 | 1.76 / 1.92 | *differs* |
| IGNORE_EOS=0 NPRED=512 | 20.37 / 20.43 | **2.84 / 2.86** | 0.71429 | 1.71 | `e7420f6caf` |
| SHORT prompt CTX1024 NPRED512 | — | 5.54 / 5.42 | 0.95640 | 2.84 | `b6c035254b` |

## 5. Adoptable

- **`WP_EXPERT_GATHER_MIN_TOKENS=2` (#25): +9.3% decode with byte-identical output.**
  Same SHA, same acceptance to five decimals, same mean length. Provably identical work,
  just faster — at `n_tokens == 1` decode routing density is 100%, so gather compacted
  nothing and only added graph nodes. *Confirm once on a fresh run before banking:* 2 reps,
  and the record arm's own decode spread is a few percent.
- **UBATCH 2048 over 512** for prefill (+20%); decode a wash.
- **`WP_EXPERT_OVERLAP` reopens as a speed-vs-determinism tradeoff** (+15% decode,
  coherent now, still irreproducible) rather than the correctness hazard #23 rejected.

## 6. Two things that change how we measure

- **NPRED=256 under-reports decode.** 2.84–2.86 at NPRED=512 vs 2.34–2.49 at 256 on the
  same config — the ~21 s decode warm-up amortising. Prefer 512 for any decode figure
  worth banking.
- **mean length is NOT set by ubatch.** ub512 gives 1.68, ub2048 gives 1.67. So ubatch is
  not what dropped the short prompt from 4.89 to 2.84. **`DSPARK_TAP` is the only
  remaining variable and is still unvalidated in the config of record.**

## 7. Still open

- **#27** the token-merge artifact (`containercars`, `grievingcars`, `tooholed`) — recurs
  across independent runs with *different* SHAs, so a specific token at a specific
  boundary, not noise. Cheapest discriminator: one run with `SPEC` off.
- **`DSPARK_TAP`** — in the record on faith, never validated on throughput or text.
- **#26** the degeneracy gate, which passed three corrupt runs today.

## 8. How it was found

Two external reviews converged independently on `choose_worker`; I verified both halves in
source before accepting either.

- **Codex (gpt-5.6-sol)** found the f16 subtotal — the amplifier that makes a partition
  change material rather than last-bit.
- **Kimi K3** found that my `WP_SELFCHECK` probe computed gather vs dense *at the same
  n_tokens* and never varied batch width, so my "the divergence is above the expert path"
  conclusion was reasoning I had not earned. It also killed the `split_immediate_deferred`
  suspicion (`may_defer` requires `n_tokens == 1`) and confirmed Sinkhorn has no
  cross-token coupling.

**Method note.** I produced six mechanisms before this one — gather corrupting the draft,
ubatch overrunning the injection buffer, the conf_min gate mis-indexed, rollback broken,
`ignore_eos` filler, and a half-applied `n_embd_out` widening that shipped an
out-of-bounds read into the config of record before I reverted it. Each came from a
correlation plus a plausible story, and each was acted on before it was checked. The rule
that survives: **a mechanism is a hypothesis until a measurement or a code path proves it;
it does not get acted on before that, and it never gets written down as a finding before
that.**
