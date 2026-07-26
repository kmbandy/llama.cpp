# 2026-07-26 morning pickup

Written at the end of 2026-07-25 on **mad-lab-2026**. Read this before touching
anything; the Vulkan section in particular records five things that are already
*proven* and must not be re-tested.

---

## 1. TL;DR

Three things shipped and are verified. One is half-built with a precisely
isolated remaining defect. One is queued and unblocked.

| work | state |
|---|---|
| CUDA weight pager on GTX 1070 | **DONE**, perplexity-equivalent to non-paged |
| Vulkan `ggml_sinkhorn_norm` (RX 480) | **DONE**, 60/60 on three backends |
| Vulkan weight paging | **half built**, defect isolated to one link — §4 |
| Instella-MoE-16B | weights downloaded, architecture scoped, spec written — §6 |

**Start at §4.3.** That is the only open engineering question.

---

## 2. Machine and repo state

- **mad-lab-2026**, `~/GitHub/llama.cpp`, master tip `7504856a2`,
  **working tree dirty with substantial uncommitted work** — see §7.
- Build dir is **`build-army`** and it is the only sanctioned one.
  `GGML_CUDA=ON` (sm_61) + `GGML_VULKAN=ON`. Builds clean, no regressions.
- Devices: `CUDA0` = GTX 1070, `Vulkan0` = RX 480 (RADV POLARIS10),
  `Vulkan1` = the 1070 through Vulkan. **`Vulkan1` is a useful oracle** — the
  same shader on silicon whose CUDA path is known good separates a shader bug
  from a RADV quirk.
- `llama-router.service` is **running** (restarted at 20:59, PID 328145, port
  8093). It was down earlier because the reboot stopped it, not because anything
  killed it.
- Instella weights: `/mnt/storage2/models/Instella-MoE-16B-A3B-Think`,
  31.7 GB, 22/22 files verified. `/mnt/storage2` has 1.3 TB free; `/` has ~38 GB
  and cannot hold a conversion.
- No board claims held (released at end of session).

---

## 3. Finished work, with the numbers

### 3.1 CUDA weight pager — validated on Pascal

First execution ever on CUDA hardware, and it is correct rather than merely
running.

```
TRANSPORT: active=HOST (O_DIRECT pthread pool)  host_batches=1204  serial_batches=0
page_ins 23660 · evictions 23408 · host_tier_hits 19651 (83%)
tier_promotion_async_enqueued 7411 · event_pool_exhausted 0
perplexity: paged 33.8775 ± 0.74379 vs non-paged 33.8765 ± 0.74323  (0.003%)
```

All three mechanisms shipped on HIP the day before — HOST O_DIRECT batching,
HostTier borrow/release zero-copy promotion, pipelined async promotion — work
unmodified on sm_61.

Two bugs found doing it:

1. **Pool auto-sizing does not clamp to free VRAM.** It requested 34688 MiB on
   an 8 GB card and hard-failed init, despite `--weight-paging-slots` help text
   claiming `-1 = auto = layer count, capped to free VRAM`. Calibrated for the
   R9700's 32 GB. **Still unfixed** — pin `--weight-paging-slots` explicitly on
   any small-VRAM card. This has now been assigned to Codex twice and not done.
2. **ABI drift segfault.** `llama-cli --version` segfaulted because the
   executable predated a rebuilt `libllama.so.0`. Shared linkage protects
   against stale *code*, not a stale *ABI*. Always build the full target set.

### 3.2 Vulkan `ggml_sinkhorn_norm` — done, plus a real bug caught

60/60 on `Vulkan0`, `Vulkan1` and `CUDA0`. `RMS_NORM` 21/21 and `SOFT_MAX`
212/212 still green.

The shader initially read its token index from `gl_GlobalInvocationID.x` alone,
but `ggml_vk_op_f32` decomposes the dispatch across `y` beyond 512 elements and
`z` beyond 262144. Proven by reverting just that line: `ERR = 1.0078`, 50/60.
Fixed by reconstructing from all three IDs. At n=4 the threshold was **nt > 128**
— an ordinary prefill batch. Added `{4,4,1024,1}` as the regression; every
pre-existing case was too small (nt ≤ 32) to see it.

Also moved the op out of the `ggml_nrows()` dispatch case into its own using
`ne02*ne03`: it is one-invocation-per-token, and `nrows` is `n*nt`, so it was
launching n× more invocations than tokens. Plus `local_size` 32 → 64 to match
Polaris' wave. Combined: 4–8× fewer waves (dispatch arithmetic, **not** a
measured wall-clock win — `test-backend-ops perf` emits nothing for any op on
this backend, verified with `RMS_NORM` as a control).

---

## 4. Vulkan weight paging — the open work

### 4.1 The architecture, so it does not get re-derived

Weight paging needs **two** halves. The transport stages pages into VRAM slots;
the **consumption path** makes the matmul read experts from those slots.

On CUDA/HIP, `wp-eval-cb.cpp:1080` calls `ggml_cuda_set_routed_expert_ptrs`
(`ggml/src/ggml-cuda/mmq.cu`), threading per-expert **device pointers** into the
MMQ kernel. Vulkan had no equivalent — which is why an implementation with a
perfect transport still emitted garbage.

Vulkan is the easier target: every slot lives in **one pool `VkBuffer`**, so the
shader needs per-expert **offsets**, not pointers. No `buffer_device_address`.

The seam is **not** where the first spec said. `ggml_vk_mul_mat_id`
(`ggml-vulkan.cpp:10360`) forks on `ggml_vk_use_mul_mat_vec_id` — `src2->ne[1]
<= 8`, i.e. token count — so **decode and short prefills use
`mul_mat_vec_base.glsl`** and long prefill uses `mul_mm.comp`. Only the vec path
is implemented; `mul_mm.comp:250` still needs the same treatment for prefills
over 8 tokens.

### 4.2 What is built (all of it compiles, no regressions)

- **Pool slot block alignment.** Vulkan's quantized matmul indexes the weight
  buffer as an array of quant blocks (`a_offset = … / QUANT_K`), so an expert's
  base must be an exact multiple of the block byte size. `PoolAllocator::init`
  gained `extra_alignment`; `Config::block_alignment` carries it from
  `llama.cpp`. Two traps here, both live in the code comments:
  - it is the **stride** that must be padded, not just a recorded alignment
    value — `slot_ptr()` is `base + idx*slot_size_`. Getting this wrong makes
    only every 15th slot legal, and the failure is silently wrong weights.
  - it must be the **lcm over all paged types**, not the max. Mixed-quant "UD"
    GGUFs carry Q6_K (210 B) *and* Q8_0 (34 B). Final value 7140 B, +0.02%.
- **Side channel** `ggml_backend_vk_wp_set_expert_offsets` (`ggml-vulkan.h`):
  thread-local, take-and-clear, indexed by **expert id** so one publication
  serves a whole multi-token batch.
- **SSBO at binding 6** (`mul_mat_vec_id_num_bindings` 6 → 7), with a per-node
  slice and a cursor reset per graph. A single shared region would serve every
  dispatch the *last* node's offsets, because dispatches execute after
  recording. CUDA avoids this by uploading through a stream-ordered memcpy.
- **Shader**: `mul_mat_vec_base.glsl` `get_offsets` uses
  `data_wp_expert_off[expert_id]` when `p.paged != 0`; non-paged path untouched.
- **Backend-neutral routing-index readback** in `wp-eval-cb.cpp`. The routed
  block read ids with raw `hipMemcpy`; on a Vulkan tensor that fails and the
  **entire routed block was silently skipped**, so nothing ever published.

### 4.3 START HERE — five links proven, one suspect

Output is still `6666…`. Do **not** re-test these; each was measured:

| link | evidence |
|---|---|
| pool contents | `GGML_VK_WP_VERIFY=1` reads each slot back and memcmps: **3000 pages, 0 mismatched** |
| slot alignment | offsets 0 / 15597960 / 31195920 / 46793880, all exact ×210 |
| offset values | exactness assertion in `wp-eval-cb.cpp`, zero violations |
| publish↔dispatch pairing | `GGML_VK_WP_TRACE=1`: strict 1:1 by tensor name and offset |
| shader / SSBO / flag | `GGML_VK_WP_FORCE=1` on a **non-paged** model → **coherent output** |
| non-regression | `MUL_MAT_ID` 790/790 on `Vulkan0` and `CUDA0` |

**The suspect is execution timing / slot lifetime.** CUDA launches kernels
immediately, so `ensure()`'s residency guarantee still holds when the kernel
runs. ggml-vulkan *records* dispatches and submits in batches
(`max_nodes_per_submit` / `flops_per_submit` in
`ggml_backend_vk_graph_compute`), so a recorded dispatch executes long after
`eval_cb` returned — by which point the pager has recycled that slot for a later
layer's expert. Correct data, correct address, **wrong moment**.

Not yet proven. Two inconclusive probes:
- `--weight-paging-slots 330` (above the ~264 expert pages per token) still
  failed, but `evictions ≈ page_ins`, so the pool was not actually holding a
  graph's working set. Worth re-examining why.
- `GGML_VK_DISABLE_ASYNC=1` unchanged — that knob governs transfers, not node
  execution.

**The next step is a design change, not another probe:** pages used by a graph
must stay pinned until that graph's **command buffer completes**, not until the
next `eval_cb`. `pin_page`/`unpin_page` and `s_range_pins` already exist; the
unpin simply happens far too early for a deferred-execution backend. Needs a
completion hook plus a pool sized for one graph's working set (~264 expert pages
at 15.6 MB per slot ≈ 4.1 GB, which fits the 480's 8 GB alongside resident
dense).

Note the slot size is set by `catalog_.max_page_size()` — 15.6 MB, the
consolidated tensor — while an expert sub-page is only ~2.9 MB. Each expert
therefore wastes ~80% of its slot. Fixing that (size-class slots already exist
behind `WP_SIZE_CLASS_SLOTS`) would cut the working set to ~760 MB and make the
pinning approach comfortable.

**Dense tensors are out of scope** (kmbandy, this session): they will be pinned
on the 6900 XT. Run with `WP_RESIDENT_DENSE=1`, which pages only routed experts.
Verified it engages — page-ins drop 16972 → 11736.

### 4.4 Debug knobs left in, all env-gated and inert by default

| env | effect |
|---|---|
| `GGML_VK_WP_VERIFY=1` | read each staged slot back and memcmp vs source |
| `GGML_VK_WP_TRACE=1` | log publish/dispatch pairing by tensor name |
| `GGML_VK_WP_FORCE=1` | exercise the paged shader path on a non-paged model |
| `GGML_VK_WP_STOCKOFF=1` | publish stock-stride offsets through the paged path |

Reproduce the failure:

```
WP_RESIDENT_DENSE=1 ./bin/llama-cli -m ~/models/LFM2.5-8B-A1B-UD-Q6_K.gguf \
  --device Vulkan0 -ngl 99 --weight-paging --weight-paging-slots 150 \
  -st -n 16 --temp 0 -p "The capital of France is"
```

Correct reference output on this device begins:
`The user wrote: "The capital of France is" and presumably expects the answer "Paris".`

---

## 5. Standing rules learned today

- **Codex/Grok subagents must never run GPU work or inference.** They have no
  board or mneme signals and are blind to GPU/RAM state. Hand them
  implementation and builds only; every GPU run is the interactive session's.
  This is not theoretical: a Codex task ran two concurrent paged jobs with 4 GB
  host tiers each, holding 6.5 GB of GTT, and drove the box to 97.6% RAM with
  swap exhausted and **171 OOM events**. kmbandy had to kill the router.
- Two security warnings fired on those dispatches and **both were correct**. The
  board-as-authorization override was given to the interactive session, not to
  autonomous agents.
- **Three consecutive Codex handoffs produced zero implementation.** Each went
  straight to a verification gate, hit the already-known failure, and stopped.
  The first prompt was self-contradicting (it stated the expected failure *and*
  "stop on failure"), but it repeated after that was fixed. After two failed
  handoffs on a task, do it directly.

---

## 6. Instella — queued, unblocked

`amd/Instella-MoE-16B-A3B-Think`, 27 layers, 64 routed + 2 shared experts,
top-6. At Q6_K roughly 13 GB — a good fit for the 6900 XT, and a far more
convenient pager target than DS4-Flash (31.7 GB of source vs 151 GB).

Spec: `docs/superpowers/specs/2026-07-25-instella-moe-arch-design.md`.

It reports `InstellaMoEForCausalLM`, which is not registered, but `model_type`
is `deepseek_v3` and the tensor inventory is DeepSeek-V3's. **The entire delta is
two features**, both verified against the weights and `modeling_instella_moe.py`:

1. **Gated attention** — one extra tensor per layer,
   `attn_output * sigmoid(gate_proj(hidden_states))` before `o_proj`. Trivial.
2. **farskip** — a dual residual stream, active on every MoE layer:

```
in: (R, A)                      A = shared-experts-only, "combine-free"
attn_out = Attn(LN(A))          attention reads A, never the routed output
R' = R + attn_out
routed, shared = MoE(LN(R'))
out: (R' + routed,  R' + shared)
```

The next layer's attention never waits on the routed-expert combine. That is a
latency-hiding structure aimed at the same problem our split work addresses, so
it is worth reading regardless of whether we convert.

Boundary conditions: layer 0 is dense (`first_k_dense_replace: 1`) and returns a
single tensor; the first farskip layer sees a non-tuple input; the final layer
feeds `model.norm` from the **full** residual `R`.

Testing note in the spec: crossing the two streams yields a model that loads and
produces fluent text while being wrong, so a farskip-specific check is required
— layer-by-layer hidden states against HF, or at minimum proving that
deliberately swapping the streams *changes* the output.

The dense `Instella-3B` is a separate, smaller job: pre-norm plus **OLMo2-style
full-width QK-norm** (`q_norm`/`k_norm` are `[2560]`, not per-head). It sits
between two existing arches and needs its own.

---

## 7. Git state — nothing is committed

`master` is at `7504856a2`. Everything below is **uncommitted**:

```
M ggml/include/ggml-vulkan.h
M ggml/src/ggml-vulkan/ggml-vulkan.cpp
M ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl
M ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iface.glsl
M ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
M src/llama.cpp
M src/weight-pager/wp-eval-cb.cpp
M src/weight-pager/wp-gpu-transport.{h,cpp}
M src/weight-pager/wp-pager.{h,cpp}
M src/weight-pager/wp-pool.{h,cpp}
M tests/test-backend-ops.cpp
?? ggml/src/ggml-vulkan/vulkan-shaders/sinkhorn_norm.comp
```

This spans three separate pieces of work — the finished Sinkhorn op, the
half-finished Vulkan paging, and the pool alignment change. **Committing them
separately in the morning is worth doing before anything else**, because the
Sinkhorn work is complete and verified and should not be hostage to the paging
work.

Specs added today (also uncommitted):
`2026-07-25-vulkan-sinkhorn-norm-design.md`,
`2026-07-25-vulkan-weight-paging-design.md` (revision 2),
`2026-07-25-instella-moe-arch-design.md`.

---

## 8. Other open threads, unchanged

- **HIP regression build on main** — still owed. The CUDA port touched shared
  source; HIP was verified unchanged by construction, never by compiling. Do it
  when main's tree is free (a DSWS spike lives there).
- **The inject path** (split spec §4a) — still the real remaining engineering
  for the two-process split. `llama_get_embeddings_layer_inp` taps a boundary
  residual in production but has no injection counterpart.
- **Four split decisions are kmbandy's**: cut point, which end samples, GGUF
  split, API owner. My cut-point recommendation moved to 20–25% on 2026 (from
  25–33%) because the 1070's page pool is ~6 GB against the R9700's 23 GB, and
  today's finding that auto-sizing does not clamp to free VRAM makes the
  small-VRAM end more fragile than the storage math implied.
- **airllm** was assessed and dismissed: its "compression" is bitsandbytes NF4
  under another name, it streams whole layers with all experts regardless of
  routing, and it uses plain buffered safetensors reads. Strictly weaker. The
  one useful takeaway is corroborative — its 50–200× gap to llama.cpp comes from
  exactly the naive per-token whole-layer I/O we already fixed.

---

## 9. Traps that cost time today

- **`pgrep -f <pattern>` self-matches the shell running it.** Bit me three
  times, including a watcher whose "busy" check could never go false and a
  download that read as running after it had finished. Check `ps -eo comm` or
  the actual artifact instead.
- **A failed build leaves the previous binary in place.** One run produced
  coherent output that briefly looked like a passing control; the build had
  errored. Always check the build exit status before reading a run.
- **llama-cli logs use `\r` from the progress spinner**, so `grep` sees one
  enormous line and silently matches nothing. Pipe through `tr '\r' '\n'` first.
- **`llama-cli` suppresses INFO by default** — pass `-v` or pager init lines are
  invisible.
- **`llama-perplexity`'s `[N]` values are running cumulative averages**, not
  independent samples. A sign test over them is meaningless; I reported a false
  regression from exactly that. Compare final estimates.
- **`--weight-paging` is gated to `LLAMA_EXAMPLE_{SERVER,CLI,PERPLEXITY}`** —
  `llama-completion` rejects it. `llama-cli` rejects `-no-cnv`; use `-st`.
- **amdgpu allocates GTT through shmem**, so a GPU process holds system RAM that
  never appears in per-process RSS. When RAM is exhausted but no process looks
  large, read `/proc/meminfo` `Shmem` and
  `/sys/class/drm/card*/device/mem_info_gtt_used`.
