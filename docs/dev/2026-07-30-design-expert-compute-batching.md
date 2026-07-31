# Design: batch expert compute into one graph per request

Status: design, ready for implementation
Author: Claude (design/review). Implementation to gpt-5.6-terra.
Date: 2026-07-30

## 1. The finding

Expert compute is **~80x above the hardware roofline**, on both machines. It is not a
GPU speed problem.

A decode-time expert is three mat**vec**s (n_tokens = 1) over ~12.2 MB of weights:

| | VRAM bandwidth | roofline for 12.2 MB | measured | ratio |
|---|---|---|---|---|
| GTX 1070 | ~256 GB/s | ~48 us | 3.93 ms/expert | **~80x** |
| R9700 | ~640 GB/s | ~19 us | ~1.2 ms/expert | **~63x** |

Both cards being that far above their *own* limit means the cost is overhead, not silicon.

## 2. Root cause

`compute_expert()` (`tools/wp-expert-worker/wp-expert-worker.cpp:1545-1602`) is called
**once per expert**, and each call performs a full standalone GPU episode:

```cpp
ggml_init(...)                        // fresh context + graph metadata
ggml_new_tensor_2d(...) x4            // gate, up, down, input
ggml_mul_mat x3 + ggml_swiglu_split
ggml_backend_alloc_ctx_tensors(...)   // *** DEVICE BUFFER ALLOCATION ***
ggml_backend_tensor_set(input, ...)   // upload the SAME activation, again
ggml_backend_graph_compute(...)       // *** SYNCHRONOUS SUBMIT + FULL SYNC ***
ggml_backend_tensor_get(output, ...)  // *** BLOCKING READBACK ***
~buffer_ptr()                         // *** DEVICE BUFFER FREE ***
```

At 8 experts/layer x 75 routed layers x 32 tokens that is roughly **19,200 device
allocations, submits, syncs, readbacks and frees per generation** — to perform ~48 us of
arithmetic each. `hipMalloc`/`cudaMalloc` are hundreds of microseconds and synchronize
the device.

This matches a pattern this fleet has hit before: a calibration job spent 97.3% of its
time emitting ~1.6M tiny HIP dispatches at ~250 us each; vectorizing the dispatch pattern
took it from 7-8 h to ~50 min. Same shape, different subsystem.

Three distinct wastes, each independently fixable:

1. **Device buffer alloc + free per expert.** Should be one persistent, reused buffer.
2. **One graph submit + full device sync per expert.** All experts in a request are
   independent; they belong in one graph with one submit.
3. **The activation is uploaded once per expert** (identical data every time), and each
   expert's full `[n_embd]` output is read back separately and summed **on the CPU**
   (`wp-expert-worker.cpp:1729-1741`). Upload once; do the weighted sum on the GPU as a
   graph node; read back one vector.

## 3. What to build

One graph per request covering **all** assigned experts:

```
input activation (uploaded ONCE)
  |
  +-- expert 0: mul_mat(gate) , mul_mat(up) -> swiglu -> mul_mat(down) -> scale(w0)
  +-- expert 1: ...                                                    -> scale(w1)
  ...                                                                       |
                                                       sum of scaled outputs (GPU)
                                                                            |
                                                        ONE readback of [n_embd, n_tokens]
```

Requirements:

- **One `ggml_backend_graph_compute` per request**, not per expert.
- **Persistent compute buffer** owned by the worker, sized for the worst case
  (max experts per request x working tensors) and reused. Grow-only; never free per
  request. Note `ggml_new_graph_custom(ctx, 64, false)` currently caps the graph at 64
  nodes — a batched graph needs roughly `5 x n_experts + n_experts` nodes, so size it
  from the real assignment count rather than a hard 64.
- **Upload the activation once** per request.
- **Weighted sum on the GPU.** The router weights are per (expert, token); apply them in
  the graph and reduce, so exactly one `[n_embd, n_tokens]` F32 buffer comes back.
- **Keep the hit/miss ordering.** Today hits compute first, then `batch.complete()`, then
  misses (`:1706-1726`) — this lets hit-experts compute while misses are still streaming.
  Do not collapse that into one barrier without measuring; if the batched graph forces a
  single submit, submit **two** graphs (hits, then misses) rather than losing the overlap.
  Two submits per request is still ~4x fewer than eight.

## 4. Mechanism counters — required, not optional

A throughput change without a mechanism counter is not a result. Add to
`WP_WORKER_STATS=1`:

```
n_graph_submits      graph computes per run   (expect ~n_requests or 2x, NOT 8x)
n_device_allocs      device buffer allocations (expect ~0 in steady state)
ns_graph_build       CPU time building graphs
ns_submit            submit + sync
ns_readback          device -> host
```

If `n_graph_submits` does not fall by roughly the experts-per-request factor, the batching
did not happen and any speedup came from somewhere else. This fleet has previously shipped
a "+3.7%" that turned out to be a cold-cache artifact with its mechanism counter reading
exactly zero.

## 5. Expected effect, written down in advance

- Per-expert marginal cost should fall toward the arithmetic (~48 us on the 1070) plus a
  shared per-request submit.
- The fitted model today is **1070: F = 1.96 ms/request + M = 3.93 ms/expert.** If the
  per-expert episode is the marginal term, M should collapse and F should rise slightly
  (one bigger graph). **The prediction is a large drop in M, not in F.** If M does not
  move, this diagnosis is wrong and we stop and re-measure rather than iterating.
- No claim is made about end-to-end tok/s here. Run-to-run variance is +/-3%, so report
  the per-leg numbers, not just throughput.

## 6. Correctness

- Output must stay coherent. Do **not** use output sha256 as the check: greedy argmax
  masks small numeric differences (output was bit-identical across moving 6 GiB of
  weights between devices).
- The weighted sum currently accumulates in F32 on the CPU. Moving it to the GPU changes
  summation order across experts. That is acceptable (the wire format is already F16
  activations with F32 accumulation) but must be stated, and the result must remain
  coherent across all three backends.
- All three backends must work: ROCm (R9700), CUDA (GTX 1070), Vulkan (RX 480). No
  compile-time `#if defined(GGML_USE_*)` in this path — that bug class has recurred six
  times in this codebase.

## 7. Build — enumerate every target explicitly

Rebuild **all** of these on **both** machines. A partial list caused tonight's outage:
inserting a field into `struct Options` while rebuilding only the library left the
worker executable on the old layout, so it read `test_hooks` from the wrong offset and
segfaulted on every request.

```
mad-lab-main   build-hip   : llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker
mad-lab-2026   build-army  : llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker   (-j2)
```

On mad-lab-2026, move the active `libllama.so*` chain aside before rebuilding so the live
services (pid 855466 nemotron embedder, pid 3025042 llama-router) keep their mapped inode.
Do not signal or restart them.
