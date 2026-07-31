# Design: expert-dispatch failure isolation (worker loss must not kill the spine)

Status: design, ready for implementation
Author: Claude (design/review). Implementation handed to gpt-5.6-terra.
Date: 2026-07-30

## 1. The defect

`graph_dispatcher::compute` (`src/pipeline/pipe-expert-dispatch-graph.cpp:96`) is
handed to `ggml_map_custom3` as a **C function pointer** (line 89) and is invoked by
the CPU backend on a **threadpool worker thread**.

Its body can throw from at least five places:

| site | condition |
|---|---|
| line 104 | `ith != 0 \|\| nth != 1` |
| line 108 | null context / owner |
| line 123 | input shapes do not match |
| line 138 | repeated expert for one token |
| line 158 → `impl::await_response` line 275 | **a worker died mid-request** |

Nothing catches. The exception unwinds across the C ABI boundary from a thread whose
entry point has no handler, so the runtime calls `std::terminate`. **The whole spine
process dies, with no diagnostic naming the worker that went away.**

Observed: killing any one of the three workers kills the spine.

## 2. Invariant to establish

> **No exception may cross `ggml_map_custom3`.** `compute` must behave as if `noexcept`.

Everything below follows from that.

## 3. Required changes

### 3.1 Sticky failure latch on `graph_dispatcher`

`compute` is `static` and reaches state only through `op_context->owner`, so the latch
lives on `graph_dispatcher` (`src/pipeline/pipe-expert-dispatch-graph.h`):

```cpp
std::atomic<bool> failed_{false};       // atomic: set from a ggml threadpool thread
mutable std::mutex failure_mutex_;
std::string        failure_message_;    // guarded by failure_mutex_

void        latch_failure(const std::string & what) noexcept;  // FIRST writer wins
bool        failed() const noexcept { return failed_.load(std::memory_order_acquire); }
std::string failure_message() const;
```

First writer wins: the first failure is the root cause; later ones are consequences of
it and must not overwrite the useful message.

### 3.2 Wrap the entire body of `compute`

```cpp
void graph_dispatcher::compute(...) {
    op_context * context = static_cast<op_context *>(userdata);
    graph_dispatcher * owner = (context != nullptr) ? context->owner : nullptr;
    try {
        if (owner == nullptr) { /* cannot latch; zero-fill and return */ }
        // *** SHORT-CIRCUIT: see 3.3 ***
        if (owner->failed()) { zero_fill(dst); return; }
        ... existing body unchanged ...
    } catch (const std::exception & e) {
        if (owner) owner->latch_failure(e.what());
        zero_fill(dst);
    } catch (...) {
        if (owner) owner->latch_failure("expert dispatch: unknown exception");
        zero_fill(dst);
    }
}
```

`zero_fill(dst)` writes 0.0f across `dst` so the remainder of the graph reads
initialised memory. The result is **numerically wrong and must never be sampled** —
§3.5 guarantees the decode is rejected before that can happen. Zero-filling is purely
to keep the rest of graph execution memory-safe, not to "carry on".

### 3.3 Short-circuit once failed — this is a performance requirement, not a nicety

GLM-5.2 has 76 routed layers, so one dead worker means **76 dispatch attempts per
token**, each blocking until its socket read fails or times out. Without the
short-circuit at the top of `compute`, a single worker loss turns every subsequent
token into minutes of serial timeouts. Check `owner->failed()` and return immediately.

### 3.4 Poison the dispatcher — a CORRECTNESS requirement

This is the subtle one and must not be skipped.

`impl::await_response` (`pipe-expert-dispatcher.cpp:265`) decrements `in_flight` **only
on the success path** (line 281). A throw at line 275 leaves:

- `in_flight` non-zero, and
- the worker socket **positioned mid-frame**, with an unconsumed or partial frame.

Reusing that connection would desync sequence ids. Since `dispatch()` matches responses
by `seq_id`, a desynced socket can return **another expert's output** for a request that
appears to succeed — silent numerical corruption, not a crash. That is far worse than
the terminate we are fixing.

Therefore add to `impl`:

```cpp
bool poisoned = false;   // set in the catch path; never cleared
```

- Set it wherever a dispatch fails part-way.
- Check it at the **top of `impl::dispatch()`** and throw immediately if set (that throw
  is then caught by §3.2 and simply re-latches).

**Recovery is explicitly out of scope.** A poisoned dispatcher stays poisoned for the
lifetime of the `llama_context`. Reconnect/failover is future work (§5).

### 3.5 Surface it as a decode error

`llama_context::graph_compute` is called at `src/llama-context.cpp:1448` and already
yields a `ggml_status` that the caller checks. Immediately after that check:

```cpp
if (expert_dispatch && expert_dispatch->failed()) {
    LLAMA_LOG_ERROR("%s: expert dispatch failed: %s\n",
                    __func__, expert_dispatch->failure_message().c_str());
    return -3;   // or the existing decode-failure convention at this call site
}
```

Verify the surrounding function's actual error convention before picking the value —
match what the neighbouring failure paths return rather than inventing a code.

Net effect: `llama_decode` returns non-zero, llama-server fails **that request** with a
500 naming the dead worker, and **the process stays up**.

## 4. Tests (required, not optional)

1. **Worker loss does not kill the spine.** Bring up the spine + ≥1 worker, issue a
   completion, kill a worker by PID mid-generation. Assert: process alive, HTTP error
   returned, log names the dead endpoint.
2. **No exception escapes `compute`.** Unit-test the callback directly with a stub whose
   `dispatch` throws; assert `compute` returns normally and `failed()` is set.
3. **Poison blocks reuse.** After a forced failure, assert a subsequent `dispatch()`
   fails fast and does **not** write to the socket.
4. **First-writer-wins.** Latch twice; assert the first message survives.
5. **Short-circuit.** After failure, assert subsequent `compute` calls perform **no**
   socket I/O.

## 5. Out of scope (future work)

`impl::build_routes()` already supports **multiple candidate workers per expert**, and
`choose_worker()` already filters by residency. So dropping a dead worker from `routes`
and re-routing its experts is feasible *where another worker holds that shard*. Today
that is true for the mad-lab-2026 pair (both hold experts 0–84) and **false** for the
R9700 (sole holder of 85–255). Real failover therefore needs a replication policy
decision first, and is deliberately not part of this change.

## 6. Note on the index

GitNexus does not know these symbols — `src/pipeline/*` is new and the index predates
it. Blast radius was derived directly:

- `graph_dispatcher::build` ← `src/llama-graph.cpp:2098`
- `graph_dispatcher` owned by `llama_context` ← `src/llama-context.h:291`, constructed
  at `src/llama-context.cpp:69`
- decode checkpoint ← `src/llama-context.cpp:1448`

Do **not** run `npx gitnexus analyze` to refresh it.
