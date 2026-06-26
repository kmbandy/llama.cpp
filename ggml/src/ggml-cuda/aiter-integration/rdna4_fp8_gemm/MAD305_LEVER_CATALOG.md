# MAD-305 — RDNA4 fp8 GEMM Optimization Lever Catalog

Date started: 2026-06-19. This is the **single consolidated reference** for the gfx1201 (RDNA4 /
R9700, wave32) fp8 (e4m3) WMMA cooperative-GEMM throughput campaign. It folds together:

- our own on-silicon measurements (the part that actually *discriminates* between ideas),
- `NVIDIA_KERNEL_IDEAS_FOR_RDNA4.md` (CUTLASS / FlashAttention / ThunderKittens cross-pollination),
- `RELATED_WORK_SWEEP_2026-06-19.md` (AITER / CK / Triton / HipKittens repo sweep),
- the three AMD RDNA4 WMMA guides (gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/2/3).

**How to use this doc:** read §1 first. Our measurements already kill a big fraction of the idea-bank,
so §1 is the filter. §2 is the ranked live-lever catalog. §3 is hard ISA/HW facts. §4 is the local
asset index (don't re-discover these paths). §5 is the status ledger. §6 is the safety boundary.

---

## 0. Standing numbers (the scoreboard)

| Quantity | Value | Note |
|---|---|---|
| Feed-free WMMA ceiling (rocWMMA peak) | **307 TF** | 100% reference |
| NOFEED ceiling, current 8×2 geometry | **~282 TF (91.8%)** | compute-only, no A/B feed |
| **Current winner** (8×2 KWINBPF + s_setprio, TWN=4) | **165.7 TF (53–54%)** | today's re-run 164; variance |
| HIP hand-written winner | 161 | |
| hipBLASLt | 143 | |
| Resident waves at winner geometry | **64** | occupancy-capped; ~208 VGPR/wave |

**The whole problem in one line:** NOFEED 282 → FED 164 is a **42% loss**, at **64 resident waves**.
Closing that gap is the campaign.

---

## 1. What our measurements have PROVEN (the evidence filter)

These are on-silicon results, not theory. They **eliminate** whole classes of ideas.

1. **Feed is NOT issue-slot-bound.** Two independent tests:
   - **B128** (this session): moved the binding B feed from 4× `global_load_tr_b64` to 2× plain
     `global_load_b128` per K-window → **B-feed instructions 16 → 8 (halved, disasm-confirmed)**,
     oracle PASS. Perf: **flat** (163.4–163.9 vs winner 163.6–164.1).
   - **ALD2** (wide A-read, prior): `ds_load_2addr_stride64_b64` halved A-read slots (16 → 8/slice).
     Perf: **flat**.
   - ⇒ Halving *either* operand's feed instruction count does nothing. **Any lever whose only effect
     is fewer feed instructions is predicted DEAD** (wide-K for both operands, lane-major-to-save-a-load,
     pointer-increment address VALU, etc.).

2. **The FED penalty is feed sitting IN the compute wave's issue stream**, not feed *volume*. NOFEED
   hits 282; the moment real feed instructions interleave with WMMA in the same wave, throughput
   collapses to 164. This is a **latency / scheduling** problem (waitcnt stalls + matrix-unit not
   issuing back-to-back), not a bandwidth or slot-count problem.

3. **Occupancy is capped at 64 resident waves** because the fat 8×2 tile costs ~208 VGPR/wave
   (RDNA4 hard cap is 256 VGPR/wave). Low occupancy means we cannot interleave other waves' WMMA
   across one wave's feed stalls.

4. **The 16-wave cooperative tile is abandoned** — it bricks (see §6).

**Corollary for ranking:** the live hypotheses are exactly two, and both are about *latency*:
(a) feed-data latency in the compute stream → **wave specialization / deeper pipeline / L2 locality**;
(b) too few resident waves to hide latency → **dyn-VGPR / leaner tiles**. Everything that survives
the filter attacks one of these.

---

## 2. The lever catalog

### 2A. LIVE — ranked by (evidence-match ÷ cost)

#### L1. Persistent tile-order / B-panel L2 locality  — ❌ DONE 2026-06-19: NO WIN (default already optimal)
> **Result:** N_STATIONARY (B-stationary) oracle PASS but **108.8 TF vs default A-stationary 164.1 = 33% regression.**
> Cache locality IS real (order swings ~50%), but the **default A-stationary order already exploits the
> correct axis.** Counter to the NVIDIA "B is expensive" premise: for our 256(M)×128(N) tile with **LDS-staged
> A** (heavy) vs light per-wave `global_load_tr` B, **A is the operand to keep L2-hot**, and the default
> (sweep N within a tile-row → A-strip stays hot) already maximizes it. A Morton swizzle would dilute that →
> also no win. ⇒ L1 closed. Move to L2. (Kept behind `.if TILEORD`, default 0 = winner byte-identical.)

- **Source:** NVIDIA #1 (persistent tile scheduler) + #6 (tile swizzle / L2 search); CUTLASS raster
  order + CTA swizzle.
- **Idea:** B is the expensive operand. Reorder the **atomic tile-claim** so adjacent M-tiles reuse a
  resident B panel in L2. Modes: `N_STATIONARY` (hold B panel, sweep M), `M_STATIONARY` (current),
  `BLOCK_SWIZZLE_2/4/8` (Morton-like grouping of output tiles).
- **RDNA4 translation:** the PM4 atomic tile-claim IS a persistent scheduler — treat it as one. No
  kernel-math change; only the claim order changes.
- **Evidence verdict:** LIVE. Could attack hypothesis (a) — part of the feed *latency* may be L2 miss.
- **Cost / risk:** hours / none (harness + claim order only, no WMMA change).
- **Experiment:** add scheduler modes; run **FEEDONLY vs FED** at each. If only FED moves → it was
  issue/residency, not cache. If FEEDONLY moves too → cache locality is real, make it shape-dependent.
- **Gate:** existing 512³ oracle (correctness is layout-invariant to claim order).

#### L2. Wave specialization (producer/consumer)  — *best evidence-match, the real lever*
- **Source:** NVIDIA #2 (warp→wave specialization); ThunderKittens "Load-Store-Compute-Finish";
  FlashAttention-3 producer/consumer; **HipKittens** 8-wave ping-pong + 4-wave interleave
  (arXiv 2511.08083, CDNA-targeted but the *scheduling discipline* transfers).
- **Idea:** dedicate **loader wave(s)** (pull next B / publish A to LDS) and **compute wave(s)** (issue
  dense WMMA only). The point is **temporal separation** so compute waves stop interleaving address/feed
  ops every few instructions — i.e. directly attacks measured fact #2 (the 282→164 gap).
- **RDNA4 translation:** no TMA/WGMMA, but a workgroup can still split roles across waves with an LDS
  handoff ring. The **BLDS** (B-in-LDS) plumbing already exists as a starting point.
- **Evidence verdict:** LIVE, **highest evidence-match** — it targets the exact penalty we measured.
- **Cost / risk:** days / real surgery; **risk = it adds barriers instead of overlap.** The whole point
  can be lost to barrier stalls.
- **Experiment:** prototype a 4-wave WG = 1 loader + 3 compute, tiny K window + LDS handoff, *even if
  not faster first*. **Gate with FEEDONLY vs FED**: if FED closes toward FEEDONLY, specialization buys
  real overlap; if both sag, it just added barriers.
- **Mine for schedule vocabulary (issue order, not syntax):** AITER FlyDSL gfx1250 `fp8_quadrant`,
  `fp8_deep_pipeline`, `b_streaming` (see §4).

#### L3. dyn-VGPR occupancy  — *the moat; stacks on L1/L2*
- **Source:** our MAD-304 (RSRC2 bit 6 armable via raw PM4) + NVIDIA #8 asymmetric-scaling mindset.
- **Idea:** raise resident-wave count (attacks fact #3) by launching leaner and growing VGPR
  dynamically, instead of statically reserving the fat tile for life.
- **The moat:** AMD's HIP/ROCr path **cannot emit dyn-VGPR**; only our raw-PM4 path can. This is the
  one lever that structurally goes past what AMD/HIP kernels can reach.
- **RDNA4 translation:** RSRC2 bit 6 (gfx1201). Needs `s_setprio` phase-stagger so the dynamic VGPR
  peak < Σ statics. Per-wave dyn cap ≈ 128 VGPR at occ 16 (blocks 160+ tiles, not ≤128 ones).
- **Evidence verdict:** LIVE. Likely **stacks on** L2 rather than competing — best layered *after* we
  see how much wave-spec alone recovers.
- **Cost / risk:** days / PM4 + stagger complexity; per-wave cap; non-deterministic exact-fill edge
  behavior (characterize, don't trust blindly).
- **Note:** "dyn-VGPR is the ONLY lever to 300+" was an over-narrowing (see strategic-recalibration KG);
  it is one of several stackable levers, not the sole path.

#### L4. Lean single-wave register-blocked tile  — *AMD's actual structure; dyn-VGPR substrate*
- **Source:** AMD guide part 1/2 (single-wave, register-blocked M0_M×M0_N tile, `dim3(32,1,1)`).
- **Idea:** instead of a fat 8-wave 208-VGPR cooperative LDS tile, use a **lean register-blocked
  single-wave tile** (≈48–80 VGPR) launched as **many small WGs** → high occupancy + in-wave ILP from
  independent accumulator chains.
- **Caveat from our data:** our 8×2 already has **16 independent accumulator chains** (more ILP than
  AMD's 2×3=6), so the *register-blocking* itself isn't the missing piece — the **occupancy** the lean
  tile enables is. So L4 is mostly valuable as the **lean substrate for L3 (dyn-VGPR)**.
- **Cost / risk:** days / full kernel rebuild.
- **Relationship:** L3 ⊕ L4 together = "AMD lean structure + our dyn-VGPR moat".

### 2B. REFUTED / DEAD (do not re-attempt without new evidence)

| Lever | Source | Why dead |
|---|---|---|
| 128-bit transpose load for fp8 | original premise | **No fp8 b128 transpose exists** on RDNA4 (§3). |
| K=32 fp8 WMMA op | guessed | **Does not exist**; double-K = two fused 16×16×16 (AMD part 2). |
| Plain b128 B feed (issue-slot win) | AMD part 2 / sweep #1 | Built + oracle PASS, perf **flat** (fact #1). Kept as occupancy-neutral substrate only. |
| Wide A-read (ALD2) | issue-slot axis | perf **flat** (fact #1). |
| "Wide-K for both operands to cut loads" | sweep #1/#2 | predicted dead by fact #1 (it only cuts feed *count*). Don't build unless paired with a latency lever. |

### 2C. CONDITIONAL / SECONDARY (real, but not the throughput headline)

- **Blockscale in the mainloop** (NVIDIA #4, CUTLASS ex.67/81): training wants per-block fp8 scale as a
  *core data path*, not a scalar side channel. Scale layout should follow the **MMA K step**. **Risk:**
  if loaded late/through a conflicting path, scale becomes the next bottleneck after feed. *Experiment:*
  scale-feed microbench (SGPR vs VGPR-alongside-B vs LDS-staged) measuring WMMA issue density with mocked
  math, **before** building full blockscale GEMM. → relevant when we move from raw TF to real training GEMM.
- **Overlapping accumulators / epilogue latency hiding** (NVIDIA #5; CK "leak last WMMA into epilogue"):
  split accumulators into 2 groups; while group 1 stores/scales/casts, group 2 finishes last K steps.
  Only after mainloop feed is less binding (else it masks the wrong issue). → training epilogue (store +
  scale + cast + bias/act for fused paths).
- **Split-K / Stream-K as load balancing** (NVIDIA #7; CK streamk): for skinny/imbalanced *training*
  shapes, split-K may let a **smaller, cleaner inner tile** avoid barriers/register cliffs — value is the
  cleaner tile, not the parallelism. Judge on end-to-end step time incl. reduction, not C-write duplication.
- **Ragged/grouped without blanket padding** (NVIDIA #10): residual handlers for M/N/K mod 16, K mod 32 —
  matters for MoE/training batches where padding-to-128 hides a fast kernel behind wasted math. Keep out
  of the peak square-GEMM path.
- **Producer-side fragment-major layout / in-register transpose** (AMD part 3; sweep): the identity-WMMA
  transpose lets a kernel transpose in registers with no memory traffic. For *fused* training ops, if a
  prior op can emit a register/lane-major fragment, the global repack disappears entirely. Minor for
  standalone GEMM; real for fused chains.

### 2D. PROCESS / INFRA (cheap, compounding — adopt alongside whatever lever)

- **Layout IDs as a first-class contract** (NVIDIA #3; CuTe layout-algebra): name the B/A layouts
  (`B_TR`, `B_LANE64`, `B_LANE128_WIDEK`, `A_LDS`, `A_LANE128_WIDEK`) and have the host repacker + oracle
  **print the layout ID**. Only compare kernels with known layout contracts. (We informally do this; B128
  is effectively `B_LANE128`.)
- **Variant manifest** (NVIDIA #9; Triton-as-discipline): record per variant — VGPR bases/pads, waitcnt
  distances, `s_setprio` polarity, B layout ID, tile-scheduler mode, and disasm + perf. Turns tuning into
  a reproducible search instead of ad-hoc asm diffs.
- **Descriptor audit** (sweep #6): pre-perf check of PM4/HSA metadata — wave32, dyn-VGPR bit, LDS size,
  workgroup size — before any perf run.
- **Rank-before-coding metric** (NVIDIA #8): score each proposed kernel by `B global bytes per WMMA` and
  `B issue instructions per WMMA` before writing it. Prefer asymmetric tiles that make **B stationary**.

---

## 3. Hard ISA / hardware facts (don't re-derive)

- **No 128-bit transpose load for 8-bit data on RDNA4.** Transpose-load family is element-width-locked:
  `global_load_tr_b64` = 8-bit (gfx1250 name `tr8_b64`, builtin `..._tr_b64_v2i32`);
  `global_load_tr_b128` = **16-bit only** (gfx1250 `tr16_b128`, builtin `..._tr_b128_v8f16`). Confirmed by
  ROCm CK `amd_transpose_load.hpp` (sizeof==1 → tr_b64, ==2 → tr_b128), the LLVM symbol table, and the
  gfx1201 assembler rejecting `tr8/tr16` (gfx1250-only spellings). ⇒ the widest fp8 transpose is `tr_b64`,
  which the kernel already uses. The only 128-bit fp8 path is **plain `global_load_b128` over a CPU
  frag-ready (lane-linear) preshuffle** (= B128, built & tested flat).
- **No K=32 fp8 WMMA op.** `v_wmma_f32_16x16x32_fp8_fp8` is invalid on gfx1201. "Double-K" = two fused
  `v_wmma_f32_16x16x16_fp8_fp8` (AMD part 2). gfx1201 `opus.hpp` exposes only the 16x16x16 fp8 builtin.
- **VGPR cap:** 256 VGPR/wave hard (RDNA4). 8×2 tile ≈ 208 VGPR → occupancy-capped at 64 resident waves.
  Each 16×16 f32 accumulator = 8 VGPR/lane; FM×FN frags × 8.
- **dyn-VGPR arming:** RSRC2 **bit 6** (gfx1201); gfx1250 moves related control to RSRC3 bit 17. Source of
  truth = LLVM `AMDHSAKernelDescriptor.h`, not the public HSA C header. Armable via raw PM4 (MAD-304);
  **HIP/ROCr cannot emit it.** Per-wave dyn cap ≈128 VGPR at occ 16.
- **Fragment layout (AMD-validated, matches our reverse-engineering):** A & B K-major, 8 elements/lane;
  `lane = lIdx%16`, `laneGroup = lIdx/16`; lanes 0-15 hold K0-7, lanes 16-31 hold K8-15 (NOT duplicated).
  A row-distributed, B/C column-distributed. AMD calls `wmma(B, A, C)` for TN.
- **WMMA runs on the vector ALU** sharing the single ISSUE port with feed instructions (unlike CDNA's
  separate Matrix Core pipe). LDS/VMEM are separate back-ends that overlap WMMA → feed costs ISSUE SLOTS
  *and* serializes via waitcnt. (Our data: the slot count isn't the binding part — the in-stream
  latency/scheduling is.)

---

## 4. Local asset index (oracle sources & reference kernels)

**Correctness oracles / fragment maps:**
- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201.cu` — gfx12 fragment map (A
  row-distributed, B/C column-distributed); matches our hand layout — **good oracle source**.
- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201_w64.cu` — wave64 mapping trap
  (C/D row groups interleaved via LUT); guard against accidental wave64 drift.
- `/home/kmbandy/GitHub/aiter/op_tests/opus/device/test_wmma_gfx1201_tiled.cu` — gfx12 tiled adaptor;
  if repaired, its partition layouts become another oracle.
- `/home/kmbandy/GitHub/aiter/csrc/include/opus/opus.hpp` — gfx1201/1200 wave32 fp8 WMMA builtin
  (`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`); confirms no native wide-K fp8.
- ROCm CK fp8 WMMA tests: `.../composablekernel/test/gemm_universal/test_gemm_universal_wmma_fp8.cpp`.

**Layout / preshuffle / pipeline references:**
- `/opt/rocm/include/ck/utility/amd_transpose_load.hpp` — the transpose-load dtype binding (§3).
- `.../composablekernel/include/ck/utility/amd_wmma.hpp` — gfx12 WMMA wrappers (f8/f8, f8/bf8, …).
- `.../composablekernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`
  — B-preshuffle gridwise GEMM.
- `/opt/rocm/include/ck` installed headers: a **"Skip LDS" direct-VMEM WMMA** mode + a pipeline that
  intentionally overlaps the final WMMA block with epilogue latency — both worth translating to asm.
- `/home/kmbandy/GitHub/aiter/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py` — preshuffle pipeline.
- `/home/kmbandy/GitHub/aiter/aiter/ops/flydsl/kernels/gemm_fp8fp4_gfx1250.py` — schedule names to mine:
  `fp8_quadrant`, `fp8_deep_pipeline`, `b_streaming` (issue order, not ISA).
- AITER gfx1201 Triton A8W8 blockscale configs (`BK=128`, `kpack=2`, `waves_per_eu` varies by shape;
  preshuffled often `num_stages=1`, lower `waves_per_eu`) — **search priors, not proof** (different
  compiler/runtime path): `.../aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE*.json`.

**Triton AMD backend passes (map to hand-asm knobs):**
- `/home/kmbandy/GitHub/triton/third_party/amd/lib/TritonAMDGPUTransforms/MoveUpPrologueLoads.cpp`
  — early-load-issue vs register-pressure tradeoff (= our FED<FEEDONLY problem).
- `.../OptimizeBufferOpPtr.cpp` — loop address-add (VALU) pressure → pointer-increment rewrite.
- `.../WmmaGroup.cpp` — gfx12 fp8/bf8 → 16x16x16; wider K preferred only where legal.
- `.../MoveUpPrologueLoads.cpp`, block ping-pong, waitcnt update, in-thread transpose passes.

**CUTLASS / FA references (concepts):**
- `.../cutlass/examples/54_hopper_fp8_warp_specialized_gemm`, `67_..._blockwise_scaling`,
  `74_blackwell_gemm_streamk`, `81_blackwell_gemm_blockwise`, `92_blackwell_moe_gemm`, `94_ada_fp8_blockwise`.
- `.../composable_kernel/include/ck_tile/.../persistent_async_input_scheduler.hpp`,
  `.../streamk_common.hpp`, `.../03_gemm/gemm_splitk_two_stage.cpp`.
- Papers: CuTe layout algebra (arXiv 2603.02298), Stream-K (2301.03598), FA-3 (2407.08608),
  FA-4 (2603.05451), HipKittens (2511.08083). Repos: ThunderKittens, HazyResearch/HipKittens, flash-attention.

**dyn-VGPR / launch metadata:**
- `/opt/rocm/lib/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h` — RSRC2 bit6 (gfx120) / RSRC3 bit17 (gfx125).
- `/opt/rocm/include/hsa/amd_hsa_kernel_code.h`, `/home/kmbandy/GitHub/aiter/csrc/include/aiter_hip_common.h`.

**Negative findings (don't go hunting):** no public gfx1201 fp8 GEMM that is both hand-PM4 and built
around `S_ALLOC_VGPR`; no public `global_load_tr_b128`-for-fp8 use; public AITER/CK/Triton don't solve
the same PM4 wave-group / dyn-VGPR problem.

---

## 5. Status ledger

- **DONE (this session): B128** — frag-ready CPU preshuffle (`mbg_preshuffle_B128`, CPU-cross-checked
  byte-identical to the tr_b64 feed) + plain `global_load_b128` device feed, all behind `.if B128`
  (default 0 → winner byte-identical). Oracle PASS; perf **flat** (~164). Kept as occupancy-neutral,
  AMD-canonical substrate. Binaries: `occ_wggemm2_82_tw4_kwin4_bpf_sp_b128{,_st1}.bin`; mode `--b128`.
- **Uncommitted real fixes worth landing:** `FMFN_LOG2` (C-store buffer-size, was the page-fault root
  cause) + `TROW_SH` (per-FM hardcode) — both byte-identical for the winner.
- **Working-tree cruft to keep/revert at commit:** the 16-wave exploration (tw8 / tw4x4 / controls,
  WGC_DBG, guarded `TW4LEAN_KWIN0`).
- **Abandoned:** 16-wave cooperative tile (bricks); the exact-½ 16-wave correctness bug (moot).

---

## 6. Safety boundary (non-negotiable)

- **Never dispatch WAVES=16.** A 16-wave barrier WG that can't co-reside on one WGP parks half its waves
  at `s_barrier_wait` forever; the driver can't preempt → **hard GPU brick** (reboot). Non-deterministic.
  An R9700 brick can wedge the desktop-driving 6900XT via the shared amdgpu driver.
- **Warn before any GPU run that can hang/brick.** 8-wave KWINBPF geometry is the safe regime.
- **dyn-VGPR IS armable on gfx1201** (RSRC2 bit 6, raw PM4) — never record it as locked/disabled.
- Harness targets KFD **Node 1 = gfx1201 R9700** (PCI 0000:42:00.0). **Node 2 = gfx1030 6900XT** drives
  the user's Hyprland desktop — leave it alone. sudo needs the user's password (`! ` prefix or ask).
