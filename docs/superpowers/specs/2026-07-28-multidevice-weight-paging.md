# Spec: multi-device weight paging (make the 6900 XT a paging device)

## Goal

Today exactly one GPU can page. `llama.cpp:145` throws if paged weights span more than one GPU
device. On mad-lab-main that means the R9700 (ROCm0) pages and the 6900 XT (ROCm1) only holds
resident tensors (attention island, lm_head, shared experts, FFN island, draft model), leaving
~10.8 GB of its VRAM unused.

Make **each GPU page its own band of layers from its own pool**, computing those layers in place.

## Design decision: N pagers, NOT one pager with N pools

`WeightPager` holds a single `PoolAllocator pool_` and a single `GpuTransport transport_`
(`wp-pager.h:773-774`). There are **123** `pool_.` / `transport_.` call sites in `wp-pager.cpp`.

- **Rejected — device-index every call site.** 123 edits in the code where a wrong pointer means a
  read from the wrong device's VRAM: silently wrong weights, not a crash.
- **Chosen — partition the catalog by device and give each partition its own complete
  `WeightPager`.** Every one of those 123 sites keeps working unmodified, because each pager still
  sees exactly one pool and one transport. Routing happens at the single existing seam,
  `find_page()`.

This also matches the eventual cross-machine structure, where each machine runs one pager over its
own shard.

## Already in place

`page_buft_` (`wp-pager.h:799`, written at `wp-pager.cpp:806`) already records each page's real
buffer type, resolved from the tensor's actual `t->buffer` in `llama.cpp:178-183`. **It is never
read.** The per-page device assignment is already computed correctly and thrown away. Use it.

## The partition rule

A page belongs to the device of its tensor's buffer type. That is decided by the buft overrides in
`wp::build_router_overrides`, so the layer→device assignment is expressed exactly the way the
existing `--weight-paging-resident-experts <blocks>` flag expresses it: by tensor-name pattern.

Add a companion flag for *paged* bands (name it `--weight-paging-device-layers`, or reuse the
existing convention — implementer's call, but it MUST be explicit block ranges per device, never an
auto-fill; auto-fill was deliberately removed from the resident-experts flag for being a
few-percent no-op that looks configured).

Partition invariant: **every catalog page lands in exactly one pager.** Assert it — sum of per-pager
page counts equals the total, and no page index appears twice.

## What changes

1. **`llama.cpp` `init_weight_pager`** — instead of throwing on `gpu_bufts.size() > 1`, build one
   pager per distinct GPU buft. Slot budgeting (step 4) already queries `ggml_backend_dev_memory`
   for one device; it must now do so per device, against that device's own free VRAM and its own
   `max_page_size()` (which differs per partition, since the max page in a band is not the global
   max).
2. **`llama_model`** — `unique_ptr<wp::WeightPager> wp_pager` becomes a small owner holding N
   pagers plus the name→(pager, page_idx) routing map. Keep a `primary()` accessor so the many
   existing single-pager call sites (`llama-context.cpp:406/548/1389/2538/2557`,
   `llama-model.cpp:3698-3733`) keep compiling; migrate them deliberately, not by search-and-replace.
3. **`wp-eval-cb.cpp`** — `pager->find_page(name)` becomes a lookup that yields both the owning
   pager and the local page index. Sites: `:687`, `:1433`, `:1434`, `:1451`, `:1453`. Every
   subsequent `pager->` call in that scope must use the pager the lookup returned, not a captured
   one.
4. **Per-pager subsystems** — host (RAM) tier, prefetcher, router predictor are per-pager. Split the
   RAM tier budget across pagers rather than giving each the full amount.
5. **Stats** — aggregate across pagers for reporting, but keep per-device breakdowns. `page_ins` is
   the project's determinism instrument; it must stay meaningful (report both total and per-device).

## Invariants that must not break

- **A page's data pointer must land in the pool of the device that tensor lives on.** This is the
  silent-wrong-weights failure. Add a debug assert comparing the resolved slot's device against
  `page_buft_[page_idx]`'s device on every `ensure()`.
- **The existing single-GPU path must be bit-identical.** With one GPU buft the partition is
  trivial (one pager) and behaviour must be exactly as today. Verify by `page_ins` count: at
  temperature 0 it is deterministic and must match the pre-change binary run-for-run.
- **`is_paged_weight()` / catalog `page_this` / buft override must still agree** (see
  `llama-model.cpp` comments and `wp::ResidentExpertPlan`). Multi-device does not relax this — it
  adds a *third* thing they must agree on, namely which device.
- Resident-expert blocks (`--weight-paging-resident-experts`) are excluded from paging entirely and
  must not appear in any partition.

## Known constraints (do not re-derive)

- The 6900 XT is on **Thunderbolt 3 at ~2.7 GB/s** (KG `8aebec54`). Its pool fills slower than the
  drive can read. This is expected, not a bug.
- A **VRAM victim tier** on that card is dead for the same reason — promotion over TB3 is no faster
  than re-reading from NVMe. Do not add one.
- `direct_to_device()` is false on this config, so transport falls to the serial path. Expected.
## Bandwidth: the drive is NOT the fixed pie

An earlier draft of this spec said both GPUs share one NVMe so this adds "pool capacity, not read
bandwidth". **That was wrong** and is corrected here, because it points at the opposite design.

Measured (KG `5d3e38fa`): the production pager achieves **~0.8-0.9 GB/s**, which is essentially
main's SN850X **QD1** rate (0.74-0.91). The same drive does **2.38-2.62 at QD4**, **2.84-2.95 at
QD16**, and sustains **6.2 GB/s** in prefill. The pager runs at an effective queue depth of ~1
against a 16-deep ring, at a **~38% duty cycle**.

So the drive is not saturated and there is no pie to split. It is **under-requested**. Two
independent pagers are two independent request streams, and raising effective queue depth is the
axis the KG values at 3-4x on the I/O path.

**Design consequence, and this is binding:** the pagers must be able to have I/O in flight
*simultaneously*. Do NOT share one io_uring ring, one worker pool, or one submission lock between
pagers, and do not serialise them behind a global mutex. Each pager owns its own I/O path. Anything
that funnels both pagers through a single submission point destroys the main reason to do this.

Open and NOT yet established: whether the two devices' I/O actually overlaps in time. With a
contiguous layer split and strictly sequential decode, device B is idle while the token is in
device A's band, so no second stream exists at that moment. The likely mechanism is instead that the
idle device fills its pool from the drive's 62% idle time. Instrument for it: per-device
`io_seconds/wall`, and `ensure_batch_avg_n` before and after.

- On one machine both GPUs share one NVMe. This is NOT a reason to treat bandwidth as fixed (above).

## Tests

- Unit: partition function — every page assigned exactly once; single-buft case yields one pager;
  pages of a resident block appear in no partition.
- Unit: routing — a name resolves to the pager whose partition contains it, and to a valid local
  index within that pager's catalog.
- Regression: single-GPU config produces an identical partition to today's behaviour.
- No GPU work. Do not build with HIP, do not run any model.

## Non-goals

- Cross-machine paging. That needs two processes and pipeline parallelism; this does not unlock it.
- Auto-sizing which layers go where. Explicit ranges only.
- Any change to eviction policy, prefetch, or the transport ladder.
