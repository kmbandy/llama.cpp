# RDNA4 fp8 GEMM Occupancy Unlock — Design Spec

**Date:** 2026-06-15. **Epic:** kernel-substrate unification (realizes the original MAD-293 intent on measured hardware data).
**Hardware:** AMD Radeon AI PRO R9700 / gfx1201 (RDNA4), 64 CU = 32 WGP, wave32, 32 GB.
**Supersedes:** the disproven "two gathers starve the WMMA" root cause in `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md` (corrected — see §9).

---

## 1. Why this exists — the measured findings

A day of measurement on gfx1201 overturned the prior premise and revealed the real lever:

- **The dequant gathers are NOT the bottleneck.** Removing gather #1 (even/odd K-split) regressed ~25%; removing gather #2 (register LUT ladder) regressed ~40–50%. Oracle stayed green throughout. The dequant micro-ops are cheap.
- **Tiny 16×16 tiles were the wall.** On the *unchanged* kernel, enlarging tiles took it 11 → 70 TF (6.4×) with zero logic change. Best tile/warp/stage tune plateaus ~74–77 TF.
- **The plateau is an occupancy wall.** Measured VGPR usage vs wave occupancy (RDNA4: 1536 VGPR-slots/SIMD, ≤96 regs = full 16-wave occupancy):

  | config | VGPR | occupancy | TFLOPS |
  |---|---|---|---|
  | 16×16 | 74 | 16/16 | 11 (no reuse) |
  | 64×64 | 96 | 16/16 | 34 |
  | 128×64 | 178 | 8/16 | 74 |
  | 128×128 | 228 | 6/16 | 70 |
  | fp8 128×128×128 | 256 | 6/16 + spills | 79 |

  Big tiles (needed for reuse) burn 178–256 VGPRs → only 6–8 of 16 waves → the WMMA units starve for lack of waves to hide memory latency. **That is the 37–50% utilization gap.**
- **The hardware ceiling is real and high.** Raw back-to-back WMMA microbench (rocWMMA, no memory in the hot loop, saturated grid): **fp8 = 307 TFLOPS, f16 = 147 TFLOPS, ratio 2.09×.** The "383 dense fp8 = 2× 191 fp16" architecture is real; 307 = 80% of the 383 marketing peak = the true sustained ceiling. hipBLASLt's 143 ≈ 143×16/6 ≈ 381 at full occupancy — i.e. hipBLASLt is *also* just occupancy-capped.
- **fp8 is the ONLY matrix-core path on this stack.** bf16/fp16 `torch.matmul` = ~10 TF (matrix cores never engage); only fp8 `_scaled_mm`/hipBLASLt (143) and partial int8 (63) are accelerated.
- **Instruction facts (gfx1201):** the only dense fp8 WMMA is `V_WMMA_F32_16X16X16_FP8_FP8` (opcode 70). K=32/64/128 fp8 WMMA are **gfx1250-only**. Sparse `V_SWMMAC_F32_16X16X32_FP8_FP8` exists (the future 766 2:4 path).

**Conclusion:** the entire gap from current kernels (ml8 77, hipBLASLt 143) to the 307 ceiling is **feeding/occupancy**, not instruction selection or dequant. The lever is occupancy — and RDNA4 exposes a feature (`s_alloc_vgpr`, §3.3.3) that no vendor GEMM uses to break the tile-vs-occupancy tradeoff.

## 2. Goal & targets

Build one hand-written gfx1201 fp8 GEMM kernel that closes the occupancy gap toward the measured 307 TF ceiling, and serves **both inference and training** as a unified substrate.

All perf is reported as **% of 307** (the measured real ceiling) and **× hipBLASLt (143)**.

| phase | lever | target | exit gate |
|---|---|---|---|
| **P1** occupancy-aware tiling | CK-grade pipelining (LDS double-buffer, load-transpose feeding) holding reuse without spilling | **≥ 143 (vendor parity), stretch ~180** | correctness oracle green + beats hipBLASLt |
| **P2** dynamic-VGPR moat | `s_alloc_vgpr`: lean waves → grab accumulator only in the compute phase → big tiles **and** 12–16 waves | **245–275 (80–90% of 307)** | preceded by a fail-fast spike; or documented honest stop at the measured wall |

**Quality is non-negotiable:** the kernel never ships a config the correctness oracle hasn't passed (bit-exact within fp8 vs torch).

## 3. Architecture — one core, two front-ends, unified inference + training

```
            ┌─────────────── fp8 front-end ───────────────┐
weights ───▶│ plain fp8 → B-fragments                      │──┐
            └─────────────────────────────────────────────┘  │
            ┌─────────────── ml8 front-end ───────────────┐  ├─▶ WMMA COMPUTE CORE ─▶ epilogue ─▶ C
4-bit ─────▶│ LUT dequant (4-bit idx → fp8) in prologue    │──┘   (LDS double-buffer,        (scale,
            └─────────────────────────────────────────────┘      load-transpose feed,        bf16 store)
                                                                  WMMA accum, hazards)
```

- **WMMA compute core** — the inner tiled fp8 matmul: LDS double-buffered staging, **load-transpose feeding (ISA §11.6.2)**, `V_WMMA_F32_16X16X16_FP8_FP8` accumulate with **hand-managed data hazards (§7.12.1)**, fp32 accumulate, epilogue (per-row/col scale + bf16 store). Templated on the front-end so the inner loop is shared.
- **fp8 front-end** — produces B-fragments directly from plain fp8 weights/activations.
- **ml8 front-end** — dequants 4-bit LUT indices → fp8 fragments in the prologue, then feeds the identical core. (The dequant is cheap — proven — so it rides for free once occupancy is fixed.)
- **Transpose-aware loading** — the same core handles all GEMM orientations by choosing which operand is transposed at load time; RDNA4's load-transpose ops exist precisely for this.

This unifies and **replaces**: the Triton ml8 LUT GEMM (inference), the Triton a8w8 fp8 path, and the MAD-281 Triton fp8 backward. One kernel, one occupancy story.

## 4. The three GEMM orientations (forward specced; backward tracked-deferred)

| GEMM | used by | operands | front-end | contracts | status |
|---|---|---|---|---|---|
| **forward** Y = X·Wᵀ | inference (prefill+decode) **+** training fwd | fp8 act × ml8/fp8 weight | ml8 or fp8 | K | **P0/P1 — fully specced, leads** |
| **dgrad** dX = dY·W | training bwd | fp8 grad × ml8/fp8 weight (transposed orientation) | ml8 or fp8 | N | **Phase B — deferred, tracked (§4.1)** |
| **wgrad** dW = dYᵀ·X | training bwd | fp8 grad × fp8 act | fp8×fp8 | M | **Phase B — deferred, tracked (§4.1)** |

### 4.1 Backward pass (Phase B) — deferred but committed

The backward already works today in Triton (MAD-281's fused fp8 backward). Phase B is a **port + occupancy-upgrade** of that working math into the unified kernel so the trainer's bwd also rides toward 245–275 instead of the Triton ceiling — **not net-new functionality.** Captured here so it is not rediscovered as "another giant kernel" at the next QAT round:

- **dgrad** (dX=dY·W, contracts N): ml8/fp8 weight accessed in the transposed orientation vs forward, via load-transpose; the ml8 front-end must be able to dequant into this orientation.
- **wgrad** (dW=dYᵀ·X, contracts M): fp8×fp8; the most reuse-heavy GEMM (large M contraction); likely wants split-K; fp32 accumulate → bf16/fp32 weight gradient.
- **autograd integration:** a PyTorch `autograd.Function` whose forward calls the fwd kernel and backward calls dgrad+wgrad. The ml8 QAT logic (index reassignment, curvature, STE) stays in Python — the kernel only supplies fast GEMMs.
- **open questions for its own design pass (before next QAT):** gradient-accumulation precision; loss-scaling interaction; how dynamic-VGPR maps to the M-contraction wgrad; whether dgrad/wgrad share the fwd tiling or need their own tune.

Phase B gets its own brief design pass + plan when reached; the unified core + dynamic-VGPR work done in P1/P2 carries directly into it.

## 5. Components

Lives in a dedicated `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/` (standalone asset the consumers link against):

| unit | responsibility | depends on |
|---|---|---|
| **wmma compute core** | shared inner GEMM (tiling, LDS double-buffer, load-transpose, WMMA accum + hazards, epilogue) | ISA §7.12, §11.6 |
| **fp8 front-end** | plain fp8 → B-fragments | compute core |
| **ml8 front-end** | 4-bit LUT dequant → fp8 fragments | compute core, ml8 layer format |
| **ceiling microbench** | promote `/tmp/wmma_peak.hip` into the repo — the 307 TF oracle + regression guard | — |
| **correctness harness** | C API on the kernel → Python harness reuses the existing ml8 dequant oracle (`tests/test_ml8_gemm_optimization.py`) + a new plain-fp8 (`torch._scaled_mm`) oracle | Task-1 oracle |
| **perf bench** | real model shapes → % of 307, × hipBLASLt | ceiling microbench |
| **dyn-VGPR spike** | the Phase-2 fail-fast (§7) | ISA §3.3.3 |
| **build** | RAM-capped build script (15 GB host — never an uncapped LLVM build); a focused hipcc/CMake target | — |

**Integration:** the kernel exposes a stable C API. Inference calls it from llama.cpp (C++); training calls it via ctypes/pybind from PyTorch autograd. Same `.so`, two bindings.

## 6. ISA grounding (RDNA4_ISA.txt / rdna4-instruction-set-architecture.pdf)

The hand-written kernel is written against the ISA, not recollection. Key anchors:
- **§7.12 WMMA** + `V_WMMA_F32_16X16X16_FP8_FP8` (opcode 70): A/B/C fragment→VGPR layout; **NEG must be 0 for fp8 A/B** (encoding).
- **§7.12.1 WMMA data hazards:** the wait-states between dependent WMMA ops we hand-manage (the microbench's NACC/ILP pattern already exploits this).
- **§11.6.2 WMMA Load-Transpose:** the LDS→fragment feeding primitive — most of Phase-1 pipelining.
- **§3.3.3 Dynamic VGPR + `S_ALLOC_VGPR` (opcode 83):** SCC success/fail, alloc block size, 8-block max, 7-block deadlock reservation, **`DYN_VGPR_EN` launch mode, and the whole-WGP-takeover rule (no mixing dynamic/non-dynamic on a WGP).**
- **§3.3.2.1 VGPR Allocation & Alignment.**

## 7. Phase-2 de-risk spike (resolved before any P2 kernel work)

`s_alloc_vgpr` is new enough that two things are unproven; the spike answers both fail-fast:
1. **Can we *launch* in dynamic-VGPR mode from our toolchain?** §3.3.3 requires the dispatch in dynamic-VGPR mode (DYN_VGPR_EN / SQ_DYN_VGPR), taking over the whole WGP. HIP may not expose this → worst case a hand-built kernel descriptor / HSA launch. **This is the campaign's biggest risk and we answer it first.**
2. **Can we *emit* `s_alloc_vgpr`?** No known LLVM/HIP intrinsic → inline asm against opcode-83 encoding. The spike calls it, checks SCC, and confirms occupancy (waves/SIMD) actually rises.

**If the spike fails both:** Phase 2 is blocked; we ship Phase 1 (vendor parity) as the win and document the wall. No months sunk on an unreachable feature.

## 8. Gates & testing

- **Correctness oracle (hard gate, every change):** kernel output ≡ torch reference, bit-exact within fp8. ml8 path reuses the dequant oracle built in the prior Task 1; fp8 path compares to `torch._scaled_mm`. Single-tile, multi-tile, and real-4B shapes.
- **Perf ratchet:** the bench (% of 307, × hipBLASLt) must not regress across a change; phase exits as in §2.
- **rocprof note:** plain `rocprof --stats` does not run on gfx1201; use `rocprof --hip-trace` for kernel-time confirmation. The median-wall-time bench is the primary signal.

## 9. What this supersedes / corrects

- The Phase-0 results doc's "two gathers starve the WMMA" root cause is **wrong** and is corrected to "occupancy wall (VGPR pressure) — measured ceiling 307 TF."
- The dequant-optimization framing of MAD-299 (kill the gathers → 80% of 383) is retired; the real lever is occupancy.
- Replaces three Triton paths (ml8 LUT GEMM, a8w8 fp8, MAD-281 fp8 backward) with one unified hand-written substrate.

## 10. Out of scope (future)

- **Sparse 2:4 path** (`V_SWMMAC_F32_16X16X32_FP8_FP8`, the 766 ceiling) — requires pruning+recalibration, a different model. Noted, deferred.
- **gfx1250 wide-K WMMA** (K=64/128 fp8) — not present on gfx1201; relevant only if/when hardware changes.
- Phase B backward implementation detail (tracked in §4.1, designed when reached).

## 11. Upstream / contribution posture

A deliberate first-impression play: **give the foundational primitive, hold the differentiating kernel.** Structure the work so the upstream-able piece lifts out clean instead of being an afterthought.

**The gift (upstream — clean, foundational, citable):** the dynamic-VGPR *enablement*. The `s_alloc_vgpr` feature is reportedly absent from LLVM/Linux usage; if our Phase-2 spike (§7) determines what's actually missing in the installed toolchain (clang 22, ROCm 7.2 — verify first; the intrinsic may already exist), that gap *is* the contribution:
- an **LLVM AMDGPU intrinsic** `llvm.amdgcn.s.alloc.vgpr` + clang builtin → `llvm/llvm-project`
- **HIP/HSA support for launching in `DYN_VGPR_EN` mode** → `ROCm/clr` (HIP) / `ROCm/ROCR-Runtime`

**The moat (keep, or negotiate into the partnership):** the high-occupancy ml8+fp8 GEMM. Candidate homes if/when offered: `ROCm/aiter` (applied LLM kernels — most likely), `ROCm/composable_kernel` (needs a real RDNA4 WMMA path), `ROCm/rocWMMA` (as a GEMM sample); `ROCm/hipBLASLt` = technique feedback, not a clean PR.

**Consumer-side, natural PR in *this* repo:** a HIP fp8 GEMM op in llama.cpp's `ggml-cuda`/HIP backend (the inference forward path).

**Three structuring rules baked in from the start (cheap now, expensive to retrofit):**
1. **The dynamic-VGPR helper is a standalone, ISA-cited unit** (its own header + test), *not* tangled into the GEMM — so it lifts out to LLVM/HIP untouched.
2. **Permissive license + clean provenance** — stay MIT (already vendored), no GPL contamination, ISA sections cited.
3. **Minimal deps + no local paths** — HIP + rocWMMA only, reproducible bench, real tests → builds anywhere ROCm does.

The Phase-2 spike's first deliverable is therefore a short **"what's missing in the toolchain" report** — which doubles as the scope of the upstream PR(s).
