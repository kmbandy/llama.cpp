# Spec: pre-carved size-class slots for the weight-pager pool

## The bug, reproduced today

`WP_SIZE_CLASS_SLOTS=1` **crashes** on GLM-5.2. Measured on mad-lab-main:

```
ARM ctl (WP_SIZE_CLASS_SLOTS=0): PPL 2.7266, page_ins 114432, evictions 111936,
                                 io_gb_read 465.769  -- works
ARM test (WP_SIZE_CLASS_SLOTS=1): Aborted (core dumped), exit 134
    "no unpinned size-class slot can fit ... allocated_slots=4096, high_water=..."
```

## Root cause

`PoolAllocator::alloc_slot_size_class_` (src/weight-pager/wp-pool.cpp:466) carves slots from
the arena **on demand, first-come**, bumping `high_water_`. There is no splitting and no
coalescing, and `high_water_` only grows.

GLM's demand is 65% small pages, so the arena gets consumed by small slots. When a large page
finally arrives there is no free slot of an adequate class AND no *used* slot with
`slot_class >= requested_class` to evict either -- the eviction scan at wp-pool.cpp:597 requires
an adequate class, and none exists. `alloc_slot` returns -1 and that propagates to a
`GGML_ABORT` in wp-eval-cb.

**The allocator cannot discover the right class mix at runtime.** It commits arena space
irreversibly and then wedges. The mix must be decided UP FRONT.

## The measured distribution (GLM-5.2 UD-Q2_K_XL)

58,368 expert sub-pages, 221.8 GiB payload, **5 distinct sizes**:

```
   size MiB     pages   % pages   payload GiB   roles
      6.375      1024      1.8%          6.4    down
      5.156       256      0.4%          1.3    down
      4.594     18688     32.0%         83.8    down, gate, up
      3.938       512      0.9%          2.0    gate, up
      3.469     37888     64.9%        128.3    gate, up
```

Uniform slots at 6.375 MiB need **363.4 GiB** to hold 221.8 GiB of payload -- **39% wasted**,
effective pool factor 0.610.

Three classes (6.375 / 4.594 / 3.469, each absorbing the size just below it) need **222.5 GiB**
-- **0.3% waste, a 1.63x larger effective pool for the same VRAM.**

Note `ffn_down` spans three sizes and gate/up span three, so **classes cannot be keyed on role**.
They must come from the measured per-tensor sub-page size, which the catalog already computes in
`add_consolidated_experts` (`per_expert_size = total_size / n_experts`).

## What to build

### 1. Expose the histogram

`PageCatalog` knows every page's size. Add something like:

```cpp
// size in bytes -> number of pages of exactly that size
std::map<size_t, int> page_size_histogram() const;
```

Count only slottable pages (`is_expert`, not `is_pinned`, not `is_consolidated`) -- the same set
that can occupy a slot.

### 2. Pre-carve the arena per class

`PoolAllocator::init` takes the histogram. When size classes are enabled, instead of carving on
demand:

- Let `f_c = n_c / total_pages` be each class's demand fraction.
- Solve total slots `K` from the arena budget `A`: `K = A / sum(f_c * s_c)`.
- Give class `c` exactly `k_c = max(1, round(f_c * K))` slots, carved contiguously.
- Trim if rounding overshoots `A`; never exceed the arena.

Log the resulting per-class slot counts and total bytes. This is the number a human needs to
sanity-check the config, so make it readable.

### 3. Allocation never carves

`alloc_slot_size_class_` becomes: take a free slot of the requested class, else evict LRU
**within that class**. Keep the existing speculative-first pass. Remove the `high_water_`
carve path when pre-carving is active.

Deciding whether a request may use a LARGER class when its own is exhausted is a judgement call:
it wastes space but avoids a stall. Either is acceptable -- **state which you chose and why**.
What is NOT acceptable is returning -1 and aborting when the arena has usable space.

### 4. Fix the auto-sizer

`src/llama.cpp:272` sizes the pool as `usable_vram / max_page_size()`. That is the uniform
assumption baked into auto-sizing -- it computes how many 6.375 MiB slots fit when 97% of pages
are 3.469 or 4.594, and so under-provisions by the whole waste factor. Use the histogram to
compute how many pages actually fit.

Leave the explicit `--weight-paging-slots` override semantics alone for now.

## Invariants

- **`WP_SIZE_CLASS_SLOTS` unset must be byte-identical to today.** That is the regression that
  matters most; the uniform path is what currently works.
- A model with ONE page size must behave exactly like the uniform path.
- **Never abort when the arena has usable space.** Exhaustion of a class is a policy decision,
  not a crash.
- PPL must be unchanged. A pool change alters *which* pages are resident, never *what* they
  contain. If PPL moves at all, something is wrong.

## Tests

- Histogram from a synthetic catalog: sizes and counts correct, non-slottable pages excluded.
- Pre-carve solver: proportional counts, total within budget, every class gets >= 1 slot,
  single-size input reproduces the uniform layout.
- Allocation: a request for the largest class succeeds after the arena is full of small slots
  (this is exactly the crash above -- it must not return -1).
- Eviction stays within class and respects pins.

## Constraints

- Do NOT run GPU work, any model, any inference, llama-cli/llama-server/llama-perplexity, or any
  cmake/make/ninja build. A GPU A/B is running on this machine right now.
  `g++ -fsyntax-only` and standalone CPU unit tests are fine.
- Do NOT run `npx gitnexus analyze` or any gitnexus tooling; the repo CLAUDE.md says to, ignore it.
- Do NOT commit, stash, revert or `git checkout`. The tree has uncommitted work from three other
  agents today. Build on top of it.
- ASCII only in code and comments.
