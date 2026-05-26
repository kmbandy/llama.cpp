# ml8 Inference Path — Phase Status

**As of 2026-05-26 early morning.** Companion to `ML8_WMMA_KERNEL_DESIGN.md`
(the design spec). This doc tracks **implementation state, file inventory,
and run-commands** so any session can pick up cleanly. Update as phases
land or new bugs surface.

---

## TL;DR

**Phase A + B (all 8 sub-phases) + C.1 + C.2 + C.3 + D.1 + D.2 are shipped
+ verified end-to-end on R9700.** The Python path from `.pt` calibration
blob through `ml8_gemm` kernel call produces bit-exact output. Both C++
wrappers (`mt_ml8_gemm` dense and `mt_ml8_moe_gemm` MoE-full-feature-surface)
pass 2-class deterministic smoke tests with max_err = 0.0 on first run.
`Ml8Linear` now handles rotation + AWQ at forward time (MAD-245), matching
the dequant-path math identity within fp8 quant noise (max_err ≤ 0.023).

**Phase C.2 SIGSEGV closed 2026-05-26 morning.** Root cause: Triton 3.7+
appends two trailing scratch pointers (`&global_scratch`, `&profile_scratch`)
to the launcher's args array (see `triton/tools/compile.py:185`). Our
`kernel_args[]` ended at the user args, so Triton's generated launcher read
two pointers past the array end → wild reads → host-side segfault. Fix is
two lines: declare both as `nullptr` `hipDeviceptr_t` locals and pack them as
the final two slots of `kernel_args[]`. Same pattern as
`mt_aiter_unified_attn.cpp` (which has it documented inline).

**Phase D (end-to-end PPL on Qwen3.5-4B) is unblocked but needs a Cell C
calibration artifact** (`/tmp/ml8-cellB-e4m3/` from prior session is gone;
must be regenerated via `calibrate_ml8.py` — multi-hour GPU run).

---

## Status table

| Phase | Status | Owner artifact |
|---|---|---|
| **A** — Vendor AITER blockscale kernels + LOCAL PATCH #1 (inlined helpers) | ✅ done | `kernels/gemm_ml8.py`, `kernels/moe_op_gemm_ml8.py` |
| **B.0a** — Design doc correction (scale not absorbed, cal vs inf format) | ✅ done | `ML8_WMMA_KERNEL_DESIGN.md` |
| **B.0b** — `.pt` → `.ml8` packed binary converter | ✅ done | `scripts/calibration/ml8_to_packed.py` + `tests/test_ml8_to_packed.py` (5 tests pass) |
| **B.1** — Dense kernel WEIGHT_FORMAT branch (LOCAL PATCH #2) | ✅ done | `kernels/gemm_ml8.py` |
| **B.2** — Triton LUT lookup AMDGCN codegen probe | ✅ done | `tests/test_triton_lut_lookup_probe.py` — verifies `buffer_load_u8` cached path |
| **B.3** — Stage 1 dequant + single-tile GEMM test | ✅ done (max_err=0) | `tests/test_ml8_kernel_stage1_dequant.py` |
| **B.4** — Stage 2/3 multi-tile cross-K-group + asymmetric shapes | ✅ done (max_err=0) | same file |
| **B.5** — MoE kernel WEIGHT_FORMAT branch + 1-expert execution test | ✅ done (max_err=0) | `kernels/moe_op_gemm_ml8.py` + `tests/test_ml8_kernel_moe.py` |
| **B.6** — Dispatch via `Registry::get_or_compile` (NOT AOT) | ✅ done — architecture clarified, design doc updated | `ML8_WMMA_KERNEL_DESIGN.md` §B.6 |
| **C.1** — Python `ml8_runtime` wrapper module | ✅ done (4 tests pass) | `scripts/calibration/ml8_runtime.py` + `tests/test_ml8_runtime.py` |
| **C.2** — C++ wrapper `mt_ml8_gemm.{h,cpp}` | ✅ done (smoke test passes, max_err = 0.0) | `wrappers/mt_ml8_gemm.{h,cpp}` + `wrappers/test_mt_ml8_gemm.cpp` |
| **D.1** — `--use-ml8-kernel` flag + `Ml8Linear` overlay | ✅ done (6 tests pass) | `scripts/calibration/reconstruct_model.py` + `ml8_runtime.Ml8Linear` + `tests/test_ml8_linear_overlay.py` |
| **C.3** — MoE C++ wrapper `mt_ml8_moe_gemm.{h,cpp}` (full feature surface) | ✅ done (smoke test passes, max_err = 0.0) | `wrappers/mt_ml8_moe_gemm.{h,cpp}` + `wrappers/test_mt_ml8_moe_gemm.cpp` + `kernels/moe_op_gemm_ml8.py` Patches #4 + #6 |
| **D.2** — Rotation/AWQ-aware `Ml8Linear` | ✅ done (9 tests pass, max_err ≤ 0.023) | `scripts/calibration/ml8_runtime.py::{Ml8Linear,ml8_linear_from_blob}` + `tests/test_ml8_linear_overlay.py` |
| **D proper** — End-to-end PPL on Qwen3.5-4B with `--use-ml8-kernel` | ⏳ pending — needs fresh Cell C calibration | (gated on calibration regen) |
| **E** — 35B-A3B MoE | ⏳ gated on MAD-238 / Task #27 | (not started) |
| **F** — Bench + AOT shape specialization for hot kernels | ⏳ deferred | (Phase F per design doc) |

---

## File inventory (new and modified this session)

### Kernels (vendored from AITER + ml8 patches)
- `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py` — dense GEMM, Patches #1-5
- `ggml/src/ggml-cuda/aiter-integration/kernels/moe_op_gemm_ml8.py` — MoE GEMM, Patches #1, #2, #4, #6

### Python runtime
- `scripts/calibration/ml8_to_packed.py` — `.pt` → `.ml8` binary converter
- `scripts/calibration/ml8_runtime.py` — `Ml8Layer`, `load_ml8_layer`, `ml8_gemm`, `Ml8Linear`, `ml8_layer_from_blob`, `dequantize_ml8_layer`
- `scripts/calibration/reconstruct_model.py` — added `overlay_ml8_kernels()` + `--use-ml8-kernel` flag

### C++ wrappers
- `ggml/src/ggml-cuda/aiter-integration/wrappers/mt_ml8_gemm.h` — dense public C ABI
- `ggml/src/ggml-cuda/aiter-integration/wrappers/mt_ml8_gemm.cpp` — uses `aiter::Registry::get_or_compile`
- `ggml/src/ggml-cuda/aiter-integration/wrappers/test_mt_ml8_gemm.cpp` — dense smoke test (2-class deterministic input)
- `ggml/src/ggml-cuda/aiter-integration/wrappers/mt_ml8_moe_gemm.h` — MoE public C ABI, full feature surface (5 HAS_* flags + SwiGLU + residual + per-row x-scale + routing)
- `ggml/src/ggml-cuda/aiter-integration/wrappers/mt_ml8_moe_gemm.cpp` — uses `aiter::Registry::get_or_compile`; 38 runtime args + 19 baked constexprs + trailing scratch ptrs
- `ggml/src/ggml-cuda/aiter-integration/wrappers/test_mt_ml8_moe_gemm.cpp` — MoE smoke test (1-expert, identity routing, all HAS_* flags off)

### Build wiring
- `ggml/src/ggml-cuda/aiter-integration/CMakeLists.txt` — added mt_ml8_gemm.cpp to aiter_triton_aot + MT_ML8_KERNEL_SOURCE compile def + test_mt_ml8_gemm in standalone block

### Tests (Python, all green on R9700)
- `tests/test_ml8_vendor_smoke.py`
- `tests/test_ml8_to_packed.py` (5 tests)
- `tests/test_triton_lut_lookup_probe.py`
- `tests/test_ml8_kernel_stage1_dequant.py` (3 tests: single-tile, multi-tile, asymmetric)
- `tests/test_ml8_kernel_moe.py` (2 tests)
- `tests/test_ml8_runtime.py` (4 tests)
- `tests/test_ml8_linear_overlay.py` (6 tests)

### Design docs
- `ggml/src/ggml-cuda/aiter-integration/ML8_WMMA_KERNEL_DESIGN.md` — updated extensively (corrected Appendix A formats; rewrote Phase B sub-phases status; Phase B.6 dispatch via Registry not AOT; Phase F = AOT specialization for hot shapes)
- `ggml/src/ggml-cuda/aiter-integration/ML8_PHASE_STATUS.md` — this file

---

## Triton AOT compatibility patches discovered this session

**Three patches in `kernels/gemm_ml8.py`** that are load-bearing for any
AOT-compile path via `aiter::Registry::get_or_compile` — not specific to ml8.
Documented in detail in KG memory `[[triton-aot-three-patches-2026-05-26]]`.

| Patch | What | Why |
|---|---|---|
| **#3** | Removed `@triton.heuristics(...)` decorator | `triton.tools.compile` calls `.create_binder()` which Heuristics wrapper doesn't expose |
| **#4** | Removed `cache_modifier: tl.constexpr` (and `W_CACHE_MODIFIER` in MoE) arg + bare `tl.load(ptr)` | Triton signature parser rejects string literals (`""` → KeyError). Applied to BOTH `gemm_ml8.py` and `moe_op_gemm_ml8.py`. |
| **#5** | Empty `config_keys` in `make_kernel_repr` | HSACO symbol name was `_kernel_GROUP_K_64_..._cache_modifier_NONE` but Triton's generated .c looks up base `_kernel` → mismatch |
| **#6** | Replace runtime `if X is not None:` checks with explicit `HAS_X: tl.constexpr` flags (5 flags: BIAS, GAMMAS, X/W/QUANT_STATIC_SCALE) | Triton's AOT signature parser (`tools/compile.py:105-116`) only accepts int/float literals as constexprs — there is no way to encode `None`. Without explicit flags, every `is not None` check evaluates True at AOT (pointer dtype ≠ None), forcing optional-feature branches to always execute. Quant_static_scale specifically would silently corrupt output by re-quantizing to fp8. Applied to `moe_op_gemm_ml8.py`. |

All four patches are reasonable for any kernel intended for AOT. The
Python JIT path still works (Phase B.5 MoE test still passes max_err = 0
after Patches #4 and #6 land).

---

## Phase C.2 SIGSEGV — postmortem (closed 2026-05-26)

**Root cause:** Triton 3.7+'s C launcher (generated by `triton/tools/compile.py`)
unconditionally appends `&global_scratch` and `&profile_scratch` as the final
two entries of the args-pointer array — see line 185 of `compile.py`. Our
`kernel_args[]` ended at the 21 user args, so the launcher's `va_arg`-style
unpack read two pointers past the array end. Those reads hit unmapped or
random stack and the host segfaulted before `hipModuleLaunchKernel` ever
issued the dispatch.

**Fix:** Two lines in `wrappers/mt_ml8_gemm.cpp` — declare
`hipDeviceptr_t p_global_scratch = nullptr` and `p_profile_scratch = nullptr`
locals, then add `&p_global_scratch, &p_profile_scratch` as the last two
elements of `kernel_args[]`. Identical to `mt_aiter_unified_attn.cpp`, which
documents the same Triton 3.7+ ABI inline (line 358).

**Standing rule for any new Triton-AOT wrapper in this tree:** every
`kernel_args[]` MUST end with `&p_global_scratch, &p_profile_scratch` (both
nullptr). This is a load-bearing ABI requirement, not optional.

**Build commands to reproduce / re-run:**
```bash
# 1. Configure (one-shot; reuse for incremental builds)
mkdir -p /tmp/ml8_smoke_build && cd /tmp/ml8_smoke_build && \
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \
  cmake -DGPU_TARGETS=gfx1201 -DAITER_AOT_ARCHES=gfx1201 \
        /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration

# 2. Build the test (after code edits in wrappers/mt_ml8_gemm.{h,cpp} or test)
cd /tmp/ml8_smoke_build && \
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \
  cmake --build . --target test_mt_ml8_gemm

# 3. Run (clear cache between attempts if you change the kernel sig)
rm -rf ~/.cache/llama.cpp/aiter/_gemm_a8w8_blockscale_kernel__*
cd /tmp/ml8_smoke_build && \
  PYTHONPATH=/home/kmbandy/GitHub/triton/python ./test_mt_ml8_gemm

# 4. Inspect HSACO symbols if the launch fails again
cache=$(ls -td ~/.cache/llama.cpp/aiter/_gemm_a8w8_blockscale_kernel__* | head -1)
llvm-readelf -s "$cache/kernel.hsaco" | grep FUNC
cat "$cache/meta.json"
```

---

## Pickup order (next sessions)

1. **Cell C calibration regen** (overnight or daytime unattended, multi-hour
   GPU run). Per the GPU safety rule, write VRAM math first. Recipe = Cell
   E (winning Pass 2 config).
2. **Phase D proper**: run reconstruct_model.py --use-ml8-kernel + --eval-ppl
   + --also-eval-baseline on the regenerated calibration. Should produce a
   real Δ_PPL number for the full ml8 inference path (rotation + AWQ + kernel).
3. **Phase E (35B-A3B)**: integrate MoE wrapper with reconstruct_model.py's
   MoE expert path. Gated on MAD-238 / Task #27 calibration scale-out.

---

## Cross-references

- `ML8_WMMA_KERNEL_DESIGN.md` — design spec (mostly stable; updated this session)
- `TURBO_FP8_KERNEL_DESIGN.md` — sibling KV-path design
- `RDNA4_AUDIT_2026-05-20.md` — RDNA4 prior-art audit (Round 2 = ml8-relevant)
- `aiter_runtime_compiler.h` — the dispatch path C.2 hooks into (MAD-188)
- KG: `[[triton-aot-three-patches-2026-05-26]]`, `[[mad-223-pipeline-ml8-4]]`, `[[ml8-ships-two-formats]]`
