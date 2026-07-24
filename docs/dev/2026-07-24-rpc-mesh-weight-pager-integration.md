# Distributed RPC Mesh + Weight-Pager Integration — findings

**Date:** 2026-07-24
**Machines:** mad-lab-main (R9700 gfx1201 32GB + RX 6900XT gfx1030 16GB, ROCm) and
mad-lab-2026 (GTX 1070 8GB CUDA sm_61 + RX 480 8GB Vulkan/RADV Polaris10)
**Link:** Tailscale, direct wired, measured RTT **0.5–0.6 ms**
**Repo:** kmbandy/llama.cpp, branch `master`, base commit `cd544f5e7`

Goal: determine whether llama.cpp's RPC backend can serve as the transport for a
2-machine inference mesh, and whether the weight pager composes with it — the
prerequisite for running a frontier model (GLM-5.2) that does not fit in
aggregate VRAM.

---

## 1. Summary of results

| # | Question | Answer |
|---|---|---|
| 1 | Does RPC work cross-machine at all? | **Yes** — 23.5 t/s, orpheus-3b fully remote |
| 2 | Multi-device layer split (local + remote)? | **Yes** — 37.7 t/s, 3.6% RPC overhead |
| 3 | MoE (`mul_mat_id`) over RPC? | **Yes** — gpt-oss-20b on 1070+480, 20.4 t/s |
| 4 | 3 backends at once (ROCm + CUDA + Vulkan)? | **Yes** for weight distribution; see §4 |
| 5 | Recurrent / hybrid models over RPC? | **No** — fails at load, see §5 |
| 6 | Does the weight pager compose with RPC? | **Yes, and numerically correct** — see §6 |
| 7 | Should remote GPUs host the attention island? | **No** — costs ~2.9x, see §7 |

---

## 2. Build notes (both boxes)

- RPC is **compiled out by default** (`GGML_RPC=OFF`). Both boxes needed a rebuild
  with `-DGGML_RPC=ON`.
- The rpc-server target is named **`ggml-rpc-server`**, not `rpc-server`.
- The fork trips an upstream tripwire: `ggml/include/ggml-rpc.h` has
  `static_assert(GGML_OP_COUNT == 97)`, which fails because the fork appends custom
  ops (`ML8_*`, `SINKHORN_NORM`, `PAGED_ATTN_MT`, `TURBO_WHT`, ...). Neutralised
  locally to `> 0`.
- **Both ends must be built from the same commit.** The graph is serialized by
  op-enum value; a fork whose `GGML_OP` enum differs from the peer would
  mis-deserialize every node after the divergence point. We rsync'd main's tree to
  2026 (`~/llama-rpc-main`) and built there — protocol 4.0.2 on both ends.
- Do **not** exclude `build*` when rsyncing the tree: it eats
  `common/build-info.cpp.in`, which is a *source* file, and configure fails.

Launch pattern used (one server per device; avoids mixing CUDA+Vulkan in one
process):

```
# on mad-lab-2026
ggml-rpc-server -d CUDA0   -H 0.0.0.0 -p 50052   # GTX 1070
ggml-rpc-server -d Vulkan0 -H 0.0.0.0 -p 50053   # RX 480
# on mad-lab-main
llama-server --rpc 100.102.191.30:50052,100.102.191.30:50053 --device ROCm0,ROCm1,RPC0,RPC1 ...
```

---

## 3. Why RPC is viable as the mesh transport

Read from `ggml/src/ggml-rpc/ggml-rpc.cpp`:

- `RPC_CMD_GRAPH_COMPUTE` serializes the **whole remote subgraph once**, not per-op.
- `RPC_CMD_GRAPH_RECOMPUTE` (newer) re-fires an identical graph by UID with an
  ~8-byte message: `reuse = cgraph->uid == last_graph_uid`. So after the first
  token, each decode token costs **one tiny command + the activation tensor**.
  Observed `graphs reused = 23` on a 24-token run.
- `RPC_CMD_SET_TENSOR_HASH` content-addresses tensors above a 10 MB threshold —
  weights upload once and are referenced by hash.

Consequence: per-token network cost is ~one hidden-state tensor per machine
boundary (~12 KB for GLM's 6144 hidden). At 0.5 ms RTT this is negligible;
measured RPC overhead was **2.3–3.6%** in every working configuration. Bandwidth
is a non-issue; only *round-trip count* matters, and RECOMPUTE already minimises it.

---

## 4. Backend heterogeneity is hidden

One `ggml-rpc-server` calls `ggml_backend_load_all()` and enumerates every non-CPU
device, so the remote's backend (CUDA vs Vulkan) is invisible to the client — it
only speaks the RPC protocol. Confirmed by running ornith-35B distributed across
**all four GPUs / three backends** with weights landing exactly per `--tensor-split
40,30,15,15`:

| GPU | backend | loaded |
|---|---|---|
| R9700 | ROCm0 | 13.0 GB |
| 6900XT | ROCm1 | 8.3 GB |
| 1070 | CUDA (RPC0) | 4.2 GB |
| RX 480 | Vulkan (RPC1) | 4.0 GB |

Weight distribution across 3 backends / 2 machines works. Compute did not — but
for an unrelated reason (§5).

---

## 5. Recurrent / hybrid models fail over RPC (open upstream problem)

**Symptom:** ornith-1.0-35B (`qwen35moe`: gated-delta-net / SSM recurrent + MoE)
loads to all devices, then the client spins at ~90–97% CPU with all GPUs at 0%
and no tokens, indefinitely.

**Isolation performed:**

| model | traits | over RPC |
|---|---|---|
| orpheus-3b | dense, standard | works |
| gpt-oss-20b | MoE, standard attn | works |
| ornith-35B | **recurrent** + MoE | **fails** |

Fails on a **CUDA** remote and a **Vulkan** remote alike, so it is not
backend-specific; and it is not "multi-device" in general, since the other two
models split fine on identical configs. The distinguishing trait is recurrence.

**Where it actually breaks** (via `GGML_RPC_DEBUG=1` on the server): the server
never errors. Its log shows only successful commands and then
`recv returned 0 (peer closed?)` — the **client** hangs up. The sequence dies
after `alloc_buffer` + `buffer_get_base` and **never reaches `set_tensor`
(weight upload) or `graph_compute`**. So the failure is **client-side, at
load-time buffer/tensor setup for the RPC device, before any compute**, and the
client then silently retries forever (hence the CPU spin and the
accept/close churn in the server log).

**Not the cause** (ruled out):
- Op support. `ggml_backend_rpc_device_supports_op()` is a stub that always
  returns `true` (`//TODO: call the remote backend`), and there is an **open**
  upstream PR fixing exactly that ("RPC: query remote backend op support instead
  of assuming all ops are supported", fixes #24177). But that PR addresses an
  *unsupported op crashing the remote at compute time*; qwen35-family recurrent
  ops run fine on both the 1070 and the 480 locally, and our failure is
  pre-compute. Worth cherry-picking for safety, **but it will not fix this**.

**Related open upstream issues** (same op family, no fix):
- **#22927** — dual-GPU **ROCm** segfault at `initializing slots`, immediately
  after `sched_reserve: resolving fused Gated Delta Net support`. Single-GPU
  works; dual-GPU Vulkan works. Stale since 2026-07-22.
- **#20307** — hybrid Mamba-2 MoE (Nemotron-H) asserts in
  `llama_memory_recurrent` recurrent-state buffer init.

**Conclusion:** splitting a recurrent/hybrid model across devices — RPC mesh *or*
even local multi-GPU — is genuinely unsolved in llama.cpp today. **GLM-5.2 is
unaffected** (MLA/DSA attention + standard MoE, no recurrent state). Kimi-K3
(KDA = delta attention, recurrent state) **would** be affected.

---

## 6. Weight pager + RPC compose, and are numerically correct

The pager was already multi-device-aware: `--weight-paging-resident-device`
selects a **resident device** (dense/attention/KV, so FA stays intra-device)
distinct from the **paging device** (the NVMe→VRAM paged expert pool). Pointing
resident at an RPC device required **no code changes**.

Why it works structurally: the eval-callback is registered on the whole scheduler
(`src/llama-context.cpp:1390`), not per-backend, so it still fires when the graph
is RPC-split; the pager keys page-ins off `paging_buft` and naturally ignores
RPC-resident tensors.

Validated with **DeepSeek-V4-Flash** (284B total / 13B active, 151 GB Q8,
`deepseek4` = MLA + MoE — the closest available analogue to GLM):

```
load_tensors: WP_RESIDENT_DENSE router: paging=ROCm0, resident=RPC0(1070), token_embd=CPU
init_weight_pager: resident_dense=ON  paged_pages=33924
wp::WeightPager: 33924 pages, 5500 slots x 4456448 B (23375 MiB)
```

Correctness (greedy, temp 0, prompt "The capital of France is"):

- resident on **local 6900XT**: `" Paris."` — coherent (baseline)
- resident on **remote 1070 over RPC**: `" Paris. The capital of Germany is Berlin."` — **coherent**

A 151 GB model ran with its experts paging from NVMe into the R9700 while its
dense/attention layers lived on a GPU **in another machine**, with correct
numerics.

---

## 7. Throughput: remote GPUs must not host the attention island

DS4-Flash, identical model/prompt/params (`-c 512 -b 128 -ub 64`), varying only
pool size and resident device:

| pool | resident device | decode t/s |
|---|---|---|
| 1.6 GB | 6900XT (local) | 0.53 |
| 1.6 GB | 1070 (RPC) | 0.28 |
| 23.4 GB | 1070 (RPC) | 0.36 |
| **23.4 GB** | **6900XT (local)** | **1.04** |

Two clean, separable effects:

- **Pool size ≈ 2x** (0.53 → 1.04) when resident is local.
- **Resident on a remote GPU ≈ 2.9x slower** (1.04 → 0.36), consistent at both
  pool sizes.

**Why:** attention + KV execute on the resident device **every token,
sequentially**, and every layer boundary crosses RPC. A bigger pool cannot
compensate — which is why 14x more pool bought only +29% in the RPC-resident
configuration.

**Architectural consequence.** Remote GPUs should hold **experts** and perform
**expert compute**, not host the attention island. Expert compute is per-token
*parallel* — ship a ~12 KB activation, compute, return a partial weighted sum:
one round trip — whereas attention is sequential per layer. This is empirical
support for the per-token 2-device expert-dispatch design
(`docs/dev/2026-07-21-tiered-dual-gpu-expert-feeding-design.md` §7.3), which
remains the unbuilt piece.

---

## 8. Practical gotchas

- **8 GB remotes need a small ubatch.** DS4 resident ≈ 6.8 GB on the 1070; the
  default `-ub 512` then wants a ~5 GB compute buffer and fails with
  `failed to allocate RPC0 buffer of size 4985575424`. `-b 128 -ub 64` fits.
- **gpt-oss-20b is incompatible with the weight pager** — emits `????????????`
  when paged. Reproduces with resident local *and* remote, and with
  `WP_SIZE_CLASS_SLOTS` 0 *and* 1, so it is not RPC- and not size-class-related.
  Do not use it as a WP correctness vehicle; use DS4-Flash.
- **`llama-cli` is unusable non-interactively here** — `-no-cnv` is rejected
  ("use llama-completion instead") and it then blocks on an interactive prompt
  with GPU idle, which looks exactly like a hang. Use `llama-completion` or
  `llama-server` + curl.
- `--weight-paging*` flags are registered only for SERVER / CLI / PERPLEXITY —
  `llama-completion` rejects them.
- `common_fit_params: ... abort` in the log is a benign warning on every DS4
  load, not a failure — do not match on "abort" when polling for load errors.
- ptrace is restricted on mad-lab-main (`yama.ptrace_scope`), so gdb cannot
  attach to a spinning process; `GGML_RPC_DEBUG=1` on the server was the
  effective substitute.

---

## 9. Status of GLM-5.2 as the target

Green:
- Arch registered in the fork: `glm4moe` and `glm-dsa`.
- Non-recurrent → clear of §5.
- MoE over RPC proven (§3, §6); pager+RPC proven correct (§6).

Open before a run:
1. **Weights not downloaded.** UD-IQ2 is ~239 GB; main's NVMe had 171 GB free at
   time of writing (`/mnt/hdd` has 825 GB but is a 7200rpm WD Blue — unusable for
   paging, which needs GB/s). kmbandy owns clearing NVMe space.
2. **Mixed-quant size-class pool.** UD-IQ2 experts are mixed precision, which
   fragments the size-class pool (`WP_SIZE_CLASS_SLOTS=0` is the known stopgap;
   a real allocator fix is wanted).
3. **Topology.** Per §7, do not put GLM's attention island on the 1070/480. Keep
   attention/dense on R9700+6900XT. GLM's always-active ~17.3B at IQ2 ≈ 5–6 GB,
   which fits the 6900XT alone — so multi-resident spread is likely *not*
   required for GLM.
4. **Expert dispatch (§7.3)** is the real unlock for using the remote GPUs
   productively, and is still unbuilt.

---

## 10. Uncommitted work

Codex was asked to add **multi-device resident support** (comma-separated
`--weight-paging-resident-device`, layer homes distributed across the listed
devices proportional to free VRAM, dense `.*` catch-all suppressed when
multi-resident so layer-home placement governs). Changes are in the working tree
of `src/llama-model.cpp`, `src/weight-pager/wp-router.{cpp,h}`,
`src/weight-pager/wp-pager.{cpp,h}` — **unreviewed and uncommitted**. Note §9.3:
this is a real capability but likely not on GLM's critical path.

Also uncommitted and pre-existing from other sessions (do not sweep into a
commit): `common/arg.cpp`, `tools/server/server-models.{cpp,h}`,
`docs/examples/router-fleet-main.ini` (kmbandy's live config), and deletions
under `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`
(the DSWS session's scratch area).
