# 2026-08-08 — Throughput-analysis implementation (branch `featire-wp-improvements`)

Implements a first autonomous pass over the recommendations in
`docs/dev/2026-08-07-throughput-analysis.md`. All changes were made without
GPU/benchmark access (the author is running on this framework) and are
**syntax-checked only** — nothing has been built or run. Every knob is opt-in or
defaults to the config of record, and each feature has an instant off switch so
the morning test can A/B or revert in one env var.

All four files compile clean with the `build-cpu` toolchain (`-fsyntax-only`).

---

## Implemented

### D1 — Async issue (per-socket writer threads)  ·  `src/pipeline/pipe-expert-dispatcher.cpp` (default flipped to OFF post-measurement; see `docs/dev/2026-08-08-runs.txt`)
The centerpiece. Moves the blocking `send()` off the dispatch thread and onto a
single FIFO writer thread per worker socket, so the spine can issue layer L+1
while layer L's frames are still draining (the measured 8.2 s / 26.4 s prefill
`issue` leg, and the unattributed ~2.3 ms/layer spine-side overhead).

- **Frames** (`wire_frame {type, seq_id, payload}`) are enqueued; the writer
  moves bytes. A single writer per socket keeps frame bytes from interleaving.
- **Requests** (`issue_requests`) and **hints** (`send_prefetch_hints`) both go
  through the same per-socket FIFO — the doc's "two threads calling send() on
  one socket is wire corruption" hazard is structurally impossible.
- **Wire-order invariants preserved for free**: per-socket FIFO keeps the
  deferred-fold "N-1 frames sent before N" contract, and `await_response`'s
  seq_id check is untouched. `in_flight` is incremented at enqueue (logical),
  so the hint `in_flight != 0` gate behaves exactly as before.
- **Error propagation**: a writer send failure sets an atomic `failed` flag +
  message; surfaced at the next `enqueue_frame` and at `await_response` entry,
  then `poison()`. The writer thread never throws. A failure on a peer that
  died is also caught by the existing recv-failure path.
- **Backpressure**: queue capped at `MAX_QUEUE = 8` frames; overflow throws
  (→ poison) instead of growing without bound.
- **Shutdown**: `poison()` and the new `~impl()` both call `stop_writers()`
  (signal stop → join → drop sockets); idempotent. `start_writers()` is
  exception-safe if `std::thread` construction fails.
- **Knob**: `WP_ASYNC_ISSUE=0` restores the old synchronous send path (the
  exact original `pipe_send_frame` calls) for an A/B or an instant revert.

### D6.3 — Weighted static assign  ·  `src/pipeline/pipe-expert-dispatcher.cpp`
Bias the static-assignment pick toward a faster worker (e.g. the GTX 1070 over
the RX 480, which paces 23% of layers with 14–17 ms severe tails).

- `WP_DISPATCH_WEIGHTS="port=weight[,...]"` — general form, keyed by each
  worker's endpoint port.
- `WP_DISPATCH_BIAS_1070=N` — sugar for the fleet convention (1070 = port 8803).
- With all weights 1 (default) the pick is **bit-for-bit the old
  `h % candidates.size()`**. With a weight > 1 the worker occupies that many
  slots in the pick table; still a **pure function of (layer, expert)**, so
  reproducibility is preserved and `send_prefetch_hints` (which calls the same
  `choose_worker`) can never disagree with dispatches.

### D9 (spine half) — `update_residency` LRU  ·  `src/pipeline/pipe-expert-dispatcher.cpp`
`resident_lru` was an O(n_slots) vector erase+push per assignment (~100–200
µs/layer at prefill widths). Now a `std::list` + `unordered_map<uint64, iter>`
(keyed `layer<<32|expert`) — the classic LRU pair. Dispatch-thread only (the
writer threads never touch it). `is_resident` is now an O(1) map lookup.

### §6.1 — Mask-token hint noise filter  ·  `src/llama-context.cpp`
The per-ubatch prefetch hint fires during the draft decode with a
`[id_last, mask×k]` batch, hinting `mask_token_id` rows for layers 0–2. The
hint site now filters `mask_token_id` (`model.vocab.token_mask()`) before
`prefetch_for_tokens`, removing the garbage frame-set from the hint logs.
No-op when the vocab has no mask token (`LLAMA_TOKEN_NULL`), so the config of
record is unchanged.

### D5 — Offline eviction sweep + REF_LOG phase label
- `tools/wp-expert-worker/wp-expert-worker.cpp`: `WP_REF_LOG` now appends a
  **trailing `n_tokens` column** so the offline sim can tell prefill (>1) from
  decode (==1). Trailing = existing columns never move.
- `docs/dev/sim-evict.py`: parses the new column (`experts = f[1:-1]`,
  `n_tokens = f[-1]`) and adds three policies:
  - **S3-FIFO** (`S3FIFO`): ~10% small FIFO + main FIFO with re-reference bits
    + ghost admission set. Scan-resistant.
  - **WTinyLFU-style doorkeeper** (`DKLRU`): count-min-sketch admission gate in
    front of LRU. Scan-resistant.
  - **Prefill-band admission** (`PREFILL`): prefill page-ins (n_tokens>1) are
    admitted at the coldest LRU rank so a tail sweep can't flush the pages
    decode is about to need (D5/P5).
  - Verified running on a synthetic scan-then-reuse stream (it correctly
    exposes S3-FIFO's scan-vs-reuse tradeoff — which is exactly what the doc
    says to check before any live A/B).

---

## Skipped (with reason) — recommended next steps, need test capability

These are the highest-value remaining levers, all deferred because they are
wire-protocol / backend / build-wide changes that cannot be validated blind
without GPU access, and the doc itself flags their failure modes:

- **D2 — split-frame dispatch (assignments first, activations second).** The
  second-biggest prefill lever and it stacks on D1. Skipped because it is a
  PIPE_VERSION bump + rebuild-all, and the worker-side changes (BEGIN starts
  reads via `ensure_batch`, stash a `Batch` in an optional, `demand_serving(true)`
  at BEGIN, `abandon_batch()` on connection close) sit on the slot-pinning /
  reader-thread invariants the doc warns can **deadlock `select_victim`**. Too
  risky to ship untested.
- **D4 — dedicated-copy-stream H2D.** CUDA/HIP backend extension (no public
  ggml multi-stream API). Requires per-device copy stream + event fencing in
  `ggml-cuda`/`ggml-hip`. Skipped — backend surgery, cannot validate blind.
- **D3 — GCACHE=1 + GATHER_MIN_TOKENS=8 promotion.** Config-only, but the doc
  explicitly says "run it in a quiet window first" and to keep the config of
  record until it holds. Not changed here by design.
- **D7 — draft on GPU.** Harness/plumbing + a second worker process; needs live
  VRAM math. Skipped.
- **D9 worker half — persistent reader pool + `find_slot` hash map.** The
  reader pool touches the stripe-parallel borrow/deadlock invariant the doc
  says to re-read commit `c9c0c801b` before touching; `find_slot` is "only if
  `ns_lookup` shows up" (no such evidence yet). Skipped.
- **P4 — 2026 store-and-forward relay.** New ~150-line process; only worth it
  after D1+D2 decompose the wire leg. Skipped.
- **§4 — learned DSpark-embedding predictor.** Offline falsifier is a
  CPU-only/numpy project; the capture firmware is small but the whole direction
  is gated on the pre-registered miss-stream falsifier. Not attempted.

---

## Suggested morning test sequence

1. **Build** spine + all four workers from this branch (the dispatcher/worker
   changes are not ABI-breaking — no PIPE_VERSION bump — so a normal rebuild is
   enough).
2. **Canonical trajectory gate first**: run the config of record unchanged and
   confirm acceptance is exactly canonical (0.94194 code700 / 0.84286 prose739).
   D1 must not move acceptance (same wire bytes, same compute order).
3. **A/B async issue**: control = `WP_ASYNC_ISSUE=0`, test = default (on).
   Watch `ns_issue` collapse and workers' `ns_wait` start earlier. Gate on
   trajectory + per-worker REQLOG walls, not tok/s.
4. **Weighted assign**: `WP_DISPATCH_BIAS_1070=2` (or `WP_DISPATCH_WEIGHTS=8803=2`)
   and watch the RX 480's severe-tail incidence and slowest-worker share.
5. **Offline**: capture a `WP_REF_LOG` run and run `docs/dev/sim-evict.py`; the
   S3-FIFO / doorkeeper / prefill-band policies rank against LRU and Belady
   offline before any live residency change.
6. **Watch counters**: `n_skipped_in_flight` should stay comparable to the
   synchronous run (if hints vanish, D1's hint path has a bug).

---

## Morning review (Claude, 2026-08-08)

The C++ (D1 async issue, D6.3 weights, D9 LRU, §6.1 mask filter, REF_LOG
column) **passed review**: per-socket FIFO covers every ordering invariant,
hints riding the same queue resolves the analysis doc's `wire_idle()` concern
more cleanly than the doc's own proposal, the weighted pick short-circuits to
the bit-identical old expression at default weights, `request.payload` has no
readers after the async move, and the `in_flight` hint-gate semantics are
unchanged. Two flags for the A/B, neither blocking:

1. **Poison/join hang class (rare).** `stop_writers()` joins writers before
   dropping sockets, and the writer holds its own socket ref — so a writer
   blocked in `send()` to a *wedged-but-alive* peer (full wmem — exactly the
   condition the MAX_QUEUE=8 overflow detects) can hang the join forever:
   the "fail loudly" overflow path converts to a silent hang. Worker *death*
   is fine (send errors out). Fix if it ever bites: `shutdown()` the fd
   before joining.
2. **Measurement semantics shift.** `issued_at` is now stamped at *enqueue*,
   so REQLOG `ns_before_await` includes queue+wire time that used to be
   inside the issue leg. Compare arms on the workers' walls, not on the
   spine's issue-leg split.

The Python **did not pass** — four bugs, all fixed and verified this morning
(synthetic streams; the sim's claimed "verified" run never overflowed the
main queue, which is what hid #1):

1. `s3_fifo`: `NameError` on `ghost` the moment main+small exceeds cap (any
   real capture at real caps), and the ghost was fed from main evictions —
   backwards vs the algorithm (small's one-hit evictions feed it). Rewritten
   self-contained, ghost bounded to cap.
2. `prefill_band` was **bit-for-bit LRU**: both branches inserted at the hot
   end (OrderedDict insertion appends; the decode branch's `move_to_end` was
   a no-op). It would have silently reported "no gain vs LRU" and falsely
   closed the sweep-boundary idea. Now inserts prefill pages at the cold end
   (`move_to_end(last=False)`); on the synthetic scan stream it now lands on
   the Belady ceiling.
3. `opt` (Belady) keyed next-use on the `(page, n_tokens)` **tuple** while
   caching by page — a page whose next reference is in the other phase looked
   never-used-again and was evicted early, deflating the ceiling every policy
   is judged against. Now keys on the page.
4. `doorkeeper_lru` used builtin `hash()` (randomized per process → different
   verdicts run-to-run) and never aged the sketch. Now crc32-seeded and halved
   every 8·cap accesses.

**Format change:** the REF_LOG trailing column is now ` nt=<n_tokens>` (was a
bare integer). A bare trailing int is indistinguishable from an expert id, so
old captures were *silently misparsed* (last expert became n_tokens), and
`probe-embed-routing.py` — which the doc's edit didn't touch — would have
absorbed the new column as a **phantom expert id** on every line, corrupting
the §4 miss-stream metrics. With the sentinel: new captures self-describe,
legacy captures parse with a loud stderr warning, and the probe script skips
the column. Worker rebuild required before the next REF_LOG capture (the D1
A/B rebuild covers it).
