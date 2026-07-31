# Design: page-locked staging buffers for the H2D upload path

Status: design, ready for implementation
Author: Claude (design/review). Implementation to gpt-5.6-terra.
Date: 2026-07-31

## 1. The target

The 1070's H2D upload runs at **1.29 GB/s**. Its link is **PCIe gen3 x4 = 3.94 GB/s**.
Get the upload to 3-4 GB/s. That is the whole job.

## 2. Why it is 1.29 GB/s

Two facts, both from source.

**Staging is pageable.** `wp-expert-worker.cpp:803` allocates the O_DIRECT read buffers with
`posix_memalign`. Correctly aligned, never registered with the driver.

**The upload is a synchronous `cudaMemcpy`** (`ggml/src/ggml-cuda/ggml-cuda.cu:749`). A
`cudaMemcpy` from *pageable* memory cannot DMA. The driver copies
host -> its own internal pinned bounce buffer -> device, in chunks, on the calling thread.
The bounce is invisible to our instrumentation because it happens inside the driver.

That is the entire gap. It is not the drive (SN750 does 2.13-2.20 GB/s at QD1), not the bus
(3.94 GB/s), not the arithmetic (0.54 ms of 12.1).

## 3. Topology, for the record

mad-lab-2026, Gigabyte Z170-HD3P, i7-6700K:

```
RX 480   01:00.0 -> 00:01.0  CPU root port   gen3 x16
1070     07:00.0 -> 00:1c.4  PCH root port   gen3 x4     <- slot limit, not card limit
SN750    08:00.0 -> 00:1d.0  PCH root port   gen3 x4
```

The 1070 and the NVMe are both behind the PCH, so each expert page crosses DMI twice
(SSD->RAM inbound, RAM->GPU outbound). **DMI 3.0 is full duplex at 3.94 GB/s per direction**,
and those two flows are in *opposite* directions, so they do not contend. There is no
DMI-imposed floor above ~3.1 ms/miss.

No hardware escape exists and none is needed: the 1070 has no resizable BAR (BAR1 = 256 MB),
Pascal GeForce exposes no P2P DMA interface, Intel client PCH does not route root-port-to-
root-port P2P, and the board has no second CPU-fed slot to move the card into.

## 4. What to build

**Allocate the staging pool from page-locked host memory.**

`StagingPool` (`wp-expert-worker.cpp:795-850`) currently does `posix_memalign` per buffer.
Allocate instead from the backend's host buffer type:

```
ggml_backend_dev_host_buffer_type(dev)   // ggml-backend.h:187
```

This is the portable entry point — CUDA maps it to `cudaHostAlloc`, HIP to `hipHostMalloc`,
Vulkan to host-visible memory. Do **not** call `cudaHostRegister` or any backend-specific
symbol directly; this worker serves ROCm, CUDA and Vulkan from one binary and the
`#if defined(GGML_USE_*)` bug class has recurred six times in this codebase.

Requirements:

- **Fall back to `posix_memalign` if the host buffer type is null** (some backends do not
  provide one) or if allocation fails. Log which path was taken, once, at startup.
- **Verify 4096-byte alignment of whatever comes back, and fall back if it is not aligned.**
  The reads are O_DIRECT; a misaligned buffer fails with EINVAL at read time rather than at
  allocation time, which would be a confusing runtime failure. `cudaHostAlloc` returns
  page-aligned memory in practice — verify it, do not assume it.
- **Do not add a field to `struct Options`.** Use an env kill switch `WP_STAGING_PINNED=0`
  read at startup. Inserting a field into `Options` is exactly what broke every worker on
  2026-07-30 (see section 7).
- Pool size is unchanged: `DEFAULT_STAGING_BUFFERS = 16` x max page (~16.3 MB) = **~261 MB**
  pinned. Page-locked memory is unswappable, so do not raise the count as part of this change.
  mad-lab-2026 has 15 GB and runs the fleet's MCP, mneme daemon and dashboard.

Nothing else changes. `complete_batch` keeps its single consumer, the upload stays a
synchronous `ggml_backend_tensor_set`, and MAD-114's device-wide ordering guarantee is
untouched — see section 5.

## 5. What NOT to touch

`ggml-cuda.cu:749` is synchronous **on purpose**. MAD-114 replaced
`cudaMemcpyAsync` + stream sync with a blocking `cudaMemcpy` because on HIP/RDNA (gfx1201,
ROCm 7.2.x) cross-stream visibility is unreliable even after host-side sync of the source
stream, letting graph kernels read stale input (ROCm/hip#3882, #3887).

Pinning the *source* buffer changes how the driver moves the bytes. It does not change the
synchronous semantics or the ordering guarantee. **Do not make this async as part of this
change.** If a single synchronous stream turns out not to saturate the link, that is a
separate change with its own correctness argument.

The comment on that line — "Cost is negligible - input tensor sizes are small" — was written
for KB-sized activations and is now false for 12.2 MB expert pages. Update the comment to say
so. Do not change the code.

## 6. Instrumentation — required

There is currently **no direct measurement of the upload at all**. The 1.29 GB/s figure is
derived, not observed. Add to `WP_WORKER_STATS=1`:

```
bytes_h2d        total bytes uploaded
ns_h2d           wall time inside ggml_backend_tensor_set, summed
gb_s_h2d         derived, printed explicitly
staging_kind     "pinned" | "pageable"   <- which path actually ran
```

`staging_kind` is not optional. A performance change whose mechanism counter is absent is not
a result: this fleet has already shipped a "+3.7%" that was a cold-cache artifact, and has
already spent a day measuring a gate that was structurally unreachable. Print what actually
happened, not what was configured.

## 7. Build - every target, both machines

```
mad-lab-main   build-hip    llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker
mad-lab-2026   build-army   llama  llama-server  llama-wp-expert-worker  test-wp-expert-worker  (-j2)
```

On 2026, move the active `libllama.so*` chain aside before rebuilding so the live services
(pid 855466 nemotron embedder, pid 3025042 llama-router) keep their mapped inode. Do not
signal or restart them.

## 8. The prediction, written down before the run

- 1070 `gb_s_h2d`: **1.29 -> >= 3.0 GB/s**. Anything under 2.5 means the bounce-copy diagnosis
  is wrong and we stop and re-measure rather than iterating.
- `staging_kind` must read `pinned` on all three workers. If Vulkan falls back to pageable,
  the RX 480 will not improve and that is expected, not a bug — report it.
- Per-request H2D on the 1070: 7.24 ms -> ~3 ms.
- No claim about end-to-end tok/s here. Run-to-run variance is +/-3%; report the per-leg
  numbers.

## 9. Correctness

- Output must stay coherent. Do **not** use output sha256 as the check — greedy argmax masks
  small numeric differences.
- This change moves no math. Byte-for-byte identical data reaches the device; only the
  transport differs. Any output change at all indicates a bug in the fallback or alignment
  logic, not a numerical tradeoff.
- All three backends must work: ROCm (R9700), CUDA (GTX 1070), Vulkan (RX 480).
