# RX480 Vulkan concurrency and paged-attention handoff

Date: 2026-07-11

## Goal

Improve LFM2.5-8B-A1B Q6_K throughput on the Radeon RX480, especially with
`--parallel 2-8`, tiered turbo4 KV, paged attention, and agent contexts around
8k tokens. Q6 weights must remain Q6 because Q5 produced unacceptable output.

## Hardware and production configuration

- AMD Radeon RX480 / Polaris10 / GCN, 8 GiB, RADV Mesa 26.0.6.
- Vulkan reports wave size 64, no FP16 compute, no integer dot product, and no
  matrix cores.
- PCIe is Gen3 x16.
- Production unit:
  `~/.config/systemd/user/llama-server-lfm25-8b-swarm-480.service`
- Model: `/home/kmbandy/models/LFM2.5-8B-A1B-UD-Q6_K.gguf` (6.7 GiB).
- Important current flags:

  ```text
  --ctx-size 524288
  --parallel 8
  --cache-type-k turbo4
  --cache-type-v turbo4
  --kv-tiered 60,40,0
  --kv-tier-paged-blocks
  --flash-attn on
  ```

The paged flag was briefly replaced with `--no-kv-tier-paged-blocks` during
testing, but it has been restored because paged/tiered KV is a required
long-running feature.

## Baseline findings

### Short-context Q6 weight throughput

`llama-bench`, Q6_K, Vulkan0, full offload:

- Prompt: approximately 460-468 tok/s at 256 tokens.
- Single-stream decode: approximately 95-98 tok/s in the isolated benchmark.
- Eight real short-context HTTP requests generated 2,048 verified tokens in
  20.7 seconds: 98.8 aggregate tok/s, 12.35 tok/s per request.

Therefore continuous batching and Q6 weight matmuls work correctly at short
context. The reported production symptom, about 3 tok/s per agent with eight
agents, is not caused by concurrency alone.

### 8k production context

With production paged/tiered turbo4 KV, one 8,192-token request plus 128 output
tokens took about 104 seconds. Before later experimental code changes, metrics
reported:

- Prompt: 76.5 tok/s.
- Decode: 15.6 tok/s.

Eight agents at roughly 3 tok/s each are about 24 aggregate tok/s. Relative to
the 15.6 tok/s single-stream 8k result, concurrency adds aggregate throughput,
but attention cost divides it among users.

The GTX 1070 comparison is important: it uses the same paged/tiered features,
runs eight agents, and gets roughly 7-8 tok/s each. The 6900 XT runs 24 agents
at roughly 10-15 tok/s each. The RX480 regression is therefore backend/
architecture-specific, not simply expected context scaling.

## Configuration and Q6 experiments already rejected

All of these used controlled A/B runs. Do not repeat without a new reason.

| Experiment | Result |
|---|---|
| `GGML_VK_FORCE_MMVQ=1` | About 2.2% slower decode |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | About 1.3% slower |
| Both together | No gain |
| Submit every 25/50/200 nodes | Within about 1% noise |
| `GGML_VK_DISABLE_ASYNC=1` | About 2.5% slower decode |
| RADV LLVM compiler (`RADV_DEBUG=llvm`) | About 27% slower than ACO |
| GCN K-quant rows/workgroup 1 | About 3% slower |
| Rows/workgroup 2 | Neutral |
| Rows/workgroup 8 | 6-8% slower |
| 256-thread large/hybrid Q6 matvec | Decode dropped from 97-98 to 58-60 tok/s |

The default Q6 matvec rows/workgroup value of 4, subgroup workgroup, ACO, async,
and compute queue are correct for Polaris.

Vulkan operation profiling at short context showed roughly:

- Q6 expert matvec: 43% of decode GPU time.
- Ordinary Q6 matvec: 14%.
- Q8 output projection: 12%.

Those kernels are already reasonably tuned; attention is the better target for
the real agent workload.

## KV/path isolation results

`llama-bench -pg 8192,64` was used to compare paths. Its combined figure mixes
prefill and decode, so use it for relative path comparisons, not as a pure
decode rate.

| Path | Combined 8k+64 rate |
|---|---:|
| F16 contiguous | 496.8 tok/s |
| turbo4 contiguous | 216.6 tok/s |
| turbo4 paged | 123.2 tok/s |
| F16 paged | Unsupported/aborts on Vulkan `PAGED_ATTN_MT` |

Real HTTP production A/B results:

| Path | Prompt | Decode | Wall for 8k+128 |
|---|---:|---:|---:|
| paged turbo4 | 76.5 tok/s | 15.6 tok/s | about 104 s |
| nonpaged tiered turbo4 | 225.9 tok/s | 14.5 tok/s | about 45 s |
| nonpaged tiered F16 | 397.4 tok/s | 11.7 tok/s | about 31.5 s |

Conclusions:

- Paged turbo4 currently imposes a severe prefill penalty.
- Turbo4 is still better for long-context decode than F16 on RX480; decode is
  bandwidth-limited and KV compression is useful.
- Removing paging improves TTFT but does not fix steady decode, and the user
  requires paging/tiering, so production was restored to paged turbo4.

## Paged Vulkan kernel investigation

Paged decode does not use the ordinary Vulkan flash-attention function. It
routes through `GGML_OP_PAGED_ATTN_MT`:

- Host dispatch: `ggml_vk_paged_attn_mt` in
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
- Pass 1: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode.comp`.
- Pass 2: `paged_attn_decode_reduce.comp`.
- Correctness harness: `tests/test-paged-attn-vk.cpp` and
  `build-army/bin/test-paged-attn-vk`.

The decode shader uses a 128-thread shared-memory tree reduction for every KV
token. Each token pays seven reduction barriers plus two score-broadcast
barriers. This is a likely Polaris cost, but direct subgroup-reduction work was
not attempted because GitNexus does not index GLSL functions and repository
instructions prohibit editing an un-analyzed function.

Two constant-only experiments were tested with the full Vulkan-vs-CUDA harness:

- Workgroup 64 / `MAX_VEC=16`: all correctness tests passed, but mean paged-op
  time was 24.7% worse; reverted.
- `CHUNK_KV=256`: all correctness tests passed, but mean paged-op time was
  31.1% worse; reverted.

The existing 128-thread workgroup and 128-token chunk are preferable.

## Important overdispatch diagnosis

Current Vulkan code computes:

```cpp
num_splits = ceil((max_blocks_per_seq * block_size) / 128)
```

and launches decode with grid z=`num_splits`.

`max_blocks_per_seq` reflects configured cache capacity, not the longest active
sequence. With `--ctx-size 524288`, this is approximately 4,096 chunk
workgroups per head/sequence. An 8k sequence only has 64 valid chunks, so about
98.4% of launched workgroups immediately return after reading `context_lens`.
This launch/scheduling waste scales with heads and concurrent sequences and is
a strong candidate for the RX480-specific production collapse.

An initial fix added longest-active-context as a fifth op parameter, computed
from paged-cache host mirrors during graph construction. It passed the static
Vulkan-vs-CUDA harness but failed production: 8k decode fell to 6.1 tok/s. The
host mirrors are installed later in the graph lifecycle, and changing the
bound in op params also interacts badly with graph reuse. That experiment was
fully reverted from source.

The diagnosis remains useful, but the bound must be provided at execution time,
not captured during graph construction.

## Recommended next implementation

Focus on eliminating capacity-sized empty workgroups without synchronizing the
GPU or invalidating graph reuse. Candidate designs, in preferred order:

1. **Indirect dispatch generated on GPU.** Add a tiny kernel that reduces
   `context_lens` for sequences with nonzero `q_lens` and writes Vulkan indirect
   dispatch dimensions for pass 1. Then use `vkCmdDispatchIndirect`. Keep scratch
   stride capacity-sized if needed, but dispatch only active chunks. This is the
   cleanest execution-time solution.
2. **Execution-time host value already available in paged context.** Trace
   `llama_kv_cache_paged_context::apply()` and graph input upload. If the backend
   can receive a small scalar/max tensor updated with the other per-graph inputs,
   use it for indirect dispatch rather than op params.
3. **Bucketed graph variants.** Maintain safe split buckets (128, 512, 2k, 8k,
   32k, etc.) selected by the server/runtime after context preparation. This is
   easier than indirect dispatch but can increase graph-cache churn.

Do not cap split count with a static environment variable unless the shader also
has a correct fallback for contexts exceeding the cap; silently truncating
attention is unacceptable.

After overdispatch is fixed, revisit the barrier-heavy dot-product reduction.
A guarded subgroup-add implementation could reduce nine barriers/token to about
two cross-subgroup barriers, but it must be validated on wave64 and warp32 using
the dedicated harness and production 8k tests.

## Validation requirements

For every candidate:

1. Run `build-army/bin/test-paged-attn-vk`; all Vulkan-vs-CUDA, scatter, turbo4,
   head_dim 64/128/256, and multi-chunk cases must pass.
2. Run the real 524k-capacity server with an 8,192-token prompt and 128 forced
   output tokens (`ignore_eos=true`). Record server metrics separately for
   prompt and predicted tokens.
3. Run 1/2/4/8 concurrent requests with exact completion-token counts.
4. Verify coherent output; do not rely only on tolerance tests.
5. Compare against interleaved baseline runs to catch thermal/clock drift.

## Repository and tooling notes

- GitNexus had to be re-indexed with `--max-file-size 2048` so the large Vulkan
  source was included. It reports `ggml_vk_paged_attn_mt` as LOW impact (two
  upstream dependants) and `ggml_vk_load_shaders` as HIGH impact (108
  dependants). Avoid loader edits unless necessary.
- GitNexus crashes at process exit with allocator errors but still writes a
  usable current index.
- Its refresh changed generated sections in `AGENTS.md` and `CLAUDE.md`. Those
  are the only tracked worktree modifications besides this handoff document at
  the time of writing.

## Current runtime/build state at handoff

- Production service is active and healthy on port 8097.
- Unit is restored to paged turbo4 tiered KV.
- All experimental Vulkan/API source changes have been reverted.
- The final clean `cmake --build build-army --target llama-server -j4`
  completed. It rebuilt the full CUDA template fanout after the public GGML
  signature was restored, restarted the service, and left it active and healthy.
  The build log is `/tmp/llama-vk-rx480-test/build-final-clean.log`.
- Benchmark and profiler artifacts are in `/tmp/llama-vk-rx480-test/`.
