# Vulkan `ggml_sinkhorn_norm` — design

**Date:** 2026-07-25
**Repo:** `~/GitHub/llama.cpp` on **mad-lab-2026**, master, tip `7504856a2`
**Target:** RX 480 (`Vulkan0`, RADV POLARIS10, GCN gfx803-class)

## 1. Why this exists

`GGML_OP_SINKHORN_NORM` has CPU (`ggml/src/ggml-cpu/ops.cpp`) and CUDA
(`ggml/src/ggml-cuda/sinkhorn.cu`) implementations. It has **no Vulkan
implementation** — `GGML_OP_SINKHORN_NORM` does not appear anywhere in
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`.

Every DeepSeek-V4 layer calls it, via `build_hc_pre` → `build_hc_sinkhorn`
(`src/models/deepseek4.cpp:338`). Without it the RX 480 cannot run a single DS4
layer, which is what produced the garbage output in the earlier 4-GPU run.

**The graph-form fallback is not an acceptable substitute.** `WP_DS4_FUSED_SINKHORN=0`
restores a ~139-node graph form, but the comment at `deepseek4.cpp:314-331`
records that this form is **broken on multi-device**: it emits
`permute`/`cont`/`sum_rows`, which the backend splitter treats as per-row
(`ggml-backend-meta.cpp` `handle_per_row` asserts only `axis != 0`), while
Sinkhorn couples **both** `ne0` and `ne1` within a token. Split across devices
mid-chain, the coupling is severed and the result is **silently wrong** —
measured on ROCm0+ROCm1 as `" 1.1.1.1.1..."` versus coherent prose from the
fused op. Multi-device is precisely the configuration the RX 480 exists to
serve, so the fallback is unsound here. A single node cannot be split
mid-computation; that is the fix, and it is why the fused op is default-ON as a
**correctness** decision rather than a performance one.

## 2. What the operation is

Input and output are `[n, n, nt, 1]` f32, contiguous, same shape. Element
`(dst=i, src=j, token=t)` lives at `t*n*n + j*n + i`. `n == ne0 == ne1`,
`n ∈ {2,4,8}` (DS4 production is `n=4`, from `hc_mult=4`). `nt = ne2*ne3`.

`eps` is a float in `op_params[0]`; `iters` is an int32 at op_params index 1.
`eps >= 0`, `iters >= 1`.

Per token, on the n×n matrix:

1. Softmax along `ne0` (the dst index) for each src column `j`, max-subtracted
   exactly as `ggml_soft_max` does.
2. Add `eps` to **every element of the matrix, once**.
3. `norm_cols`: each dst row `i` divided by (sum over src `j`) **+ eps**.
4. Repeat `iters - 1` times: `norm_rows` (each src column `j` divided by sum
   over dst `i`, **+ eps**), then `norm_cols` again.

**Numerics must match the reference exactly, and the operation order must be
mirrored literally rather than re-derived or algebraically simplified.** DS4
expert routing is downstream of this result, and routing determines which
experts get paged in. A subtly wrong normalization does not crash — it silently
changes expert selection. The existing comment records a measured case where
perturbing only the iteration count moved physical reads from 76.68 GB to
150.37 GB.

Note carefully: `eps` is added **both** once to the matrix **and** to every
running sum. Both are load-bearing. Do not fold them together.

## 3. Approach

Follow the CUDA kernel's strategy, which suits this op exactly: **one
invocation per token**, whole matrix held in registers, no cross-invocation
communication, no shared memory, no subgroup reductions. At `n=4` this is
sixteen floats and roughly 1300 FLOPs per token. The parallel dimension is
`nt`, which is large; the per-token work is tiny and inherently serial.

This is a good fit for Polaris: no subgroup-size assumptions (RADV is 64-wide),
no atomics, no barriers.

**Emit one shader variant per `n`, not a single dynamically-indexed shader.**
The CUDA side templates on `n` and fully unrolls. In GLSL, a private array
indexed by a runtime-varying value is liable to be demoted from registers to
scratch memory, which on Polaris is a large penalty for an op this small. Use
the generator's define mechanism to produce `n = 2, 4, 8` variants with `n`
known at compile time so the loops unroll and the matrix stays in registers.
Dispatch selects the variant, mirroring the `switch (n)` in
`sinkhorn_norm_f32_cuda`. Anything outside `{2,4,8}` should abort with a clear
message, as CUDA does.

Push constants need `n`, `nt`, `eps`, and `iters`. `vk_op_unary_push_constants`
carries only float `param1..param4`, so define a dedicated push-constant struct
for this op rather than smuggling an integer iteration count through a float.
`vk_op_soft_max_push_constants` is the local precedent for an op-specific
struct.

## 4. Integration points

`l2_norm` is the closest structural analogue in the Vulkan backend and should
be used as the template. Its touch points, all in
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` unless noted:

- the shader itself, a new `.comp` under `vulkan-shaders/`
- generator registration in `vulkan-shaders/vulkan-shaders-gen.cpp`
  (`l2_norm` precedent at line 781)
- pipeline member declaration (`l2_norm` at 860)
- `ggml_vk_create_pipeline` call (5193)
- pipeline selection in the op switch (11085)
- the op case at 11705
- the dispatch function (12965) and its call site (15098)
- `supports_op` (17870)
- the CPU-comparison clone path used by the debug harness (18738)

Advertise support only for what is actually implemented: `src0` and `dst` both
f32, both contiguous, same shape, `ne0 == ne1`, `n ∈ {2,4,8}`. Returning true
from `supports_op` for a shape the shader mishandles is worse than returning
false, because the scheduler will route work to it silently.

Per the repo's `CLAUDE.md`, run `gitnexus_impact` on each existing symbol before
modifying it, and `gitnexus_detect_changes()` before committing.

## 5. Verification

Test cases **already exist** and need no authoring —
`tests/test-backend-ops.cpp:8420-8424` covers `{4,4,32,1}` at `iters=20` and
`iters=1`, `{4,4,1,1}`, `{2,2,16,1}`, and `{8,8,16,1}`. `test-backend-ops`
compares each backend against CPU automatically.

Required, in order:

1. `test-backend-ops -o SINKHORN_NORM -b Vulkan0` — the RX 480, the actual
   target. All five cases must pass.
2. `test-backend-ops -o SINKHORN_NORM -b Vulkan1` — the GTX 1070 through
   Vulkan. This is a strong second signal: the same shader on silicon whose
   CUDA path is known-good isolates a shader bug from a Polaris driver quirk.
3. Confirm the CUDA and CPU cases still pass, i.e. nothing shared regressed.

Report the real command output and exit status, not a summary of it. A partial
build exits 0 having done nothing; build the full target set and confirm from
object timestamps that the files you changed actually recompiled.

`iters=1` is a deliberate boundary case in the existing tests: it exercises
step 3 with the loop at step 4 never entered. `n=2` and `n=8` bracket the
production `n=4`.

## 6. Out of scope

- Any change to CPU or CUDA sinkhorn, or to `deepseek4.cpp`. This adds a
  backend implementation of an existing, frozen operation.
- Performance tuning. The op is ~1300 FLOPs per token; correctness and removing
  the dispatch-count problem are the entire point.
- The `WP_DS4_FUSED_SINKHORN=0` graph path — see §1, it stays as-is.
- Running DS4 end-to-end on the 480. That needs a board claim and is a separate
  step after this passes.

## 7. Build and machine notes

Build directory is **`build-army`** — the only sanctioned one on this box
(`GGML_VULKAN=ON` and `GGML_CUDA=ON` are both already configured; `build-cuda`
and `build-rpc` were deleted). Vulkan shaders are compiled by the generator at
build time, so a shader syntax error surfaces during the build, not at runtime.

`/` is at 82% with ~40 GB free — enough, but do not add build directories.

**No GPU workload beyond `test-backend-ops` without a board claim.** That test
is small and short; a full DS4 run is not. A live `llama-router.service` exists
on the fleet — never `pkill`/`pgrep` by pattern.
