# Wave Specialization Research — NVIDIA `setmaxnreg` → AMD `s_alloc_vgpr`

Primary-source verified 2026-06-21. Feeds MAD-305 task #323 (wave-specialization
loader/compute prototype). All claims below are from CUTLASS source + Colfax
tutorial (NOT the Gemma 12B draft, which got the unit, the constraints, and the
timing wrong — see "Corrections" at bottom).

## The mechanism (NVIDIA Hopper sm_90)

Instructions: `setmaxnreg.inc.sync.aligned.u32` (acquire) /
`setmaxnreg.dec.sync.aligned.u32` (release).

Exact source — `cutlass/include/cutlass/arch/reg_reconfig.h`:

```cpp
template<uint32_t RegCount> CUTLASS_DEVICE
void warpgroup_reg_alloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount));
#endif
}
template<uint32_t RegCount> CUTLASS_DEVICE
void warpgroup_reg_dealloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount));
#endif
}
```

- **Granularity: per-warpgroup** (`.sync.aligned` = collective across the 4-warp /
  128-thread warpgroup). Changes the *per-thread* max register count.
- **Operand constraints: range [24, 256] inclusive, multiple of 8.**
- Arch guard `CUDA_CTA_RECONFIG_ACTIVATED` → sm_90 (Hopper) and later
  (SM100/101/103/120/121).
- It is a **hint/request**. Kernel must launch with a valid per-thread max via
  `__launch_bounds__` or the realloc can't apply.
- Contents: `inc` → acquired registers are **uninitialized**; `dec` → released high
  registers must be treated as **invalid** (not read).

## How GEMM uses it (warp-specialized schedule)

- **Producers (TMA load warpgroups) `dealloc`** → drop to a low count.
  **Consumers (WGMMA math warpgroups) `alloc`** → raise to a high count.
- Real splits (Colfax, verbatim):
  - `24/240/240` — 1 producer warpgroup + 2 consumer warpgroups.
  - `32/160/160/160` — 1 producer + 3 consumers.
  - Producer collapses to 24–32 regs/thread; donates the rest to consumers'
    accumulators.
- **Timing: one-time prologue action.** Each warpgroup picks its role and sets its
  register budget ONCE at kernel entry; then the steady-state mainloop runs. The
  register split is NOT per-iteration.
- Synchronization: **mbarrier objects in SMEM**, via CUTLASS `Pipeline`
  (`PipelineTmaAsync`, `producer_acquire` / `consumer_wait`). The mbarrier
  handshake is per-iteration for DATA; the register split is not.

## Relevance to the AMD port (task #323)

| Aspect            | NVIDIA (Hopper)                          | AMD RDNA4 / gfx1201                            | Transfers? |
|-------------------|------------------------------------------|-----------------------------------------------|------------|
| Instruction       | `setmaxnreg.inc/.dec`                     | `s_alloc_vgpr`                                | YES — direct analog |
| Granularity       | per-warpgroup (4 warps collective)        | per-wave (wave32, independent)                | FINER on AMD — each wave sizes itself |
| Reg unit          | per-thread, [24,256], x8                  | per-wave VGPR, rounded to BLOCK_SIZE (256 max)| different unit, same idea |
| Grow/shrink contents | grow=uninit, shrink=invalid            | grow=uninit, shrink=forfeit                   | IDENTICAL semantics |
| Failure mode      | hint; can't fail (capped by launch_bounds)| **can FAIL (SCC=0)** if SIMD pool exhausted   | NO — AMD must handle failure |
| Sync primitive    | hardware mbarrier in SMEM (TMA-aware)     | `s_barrier` + LDS ring/flags (no mbarrier/TMA)| NO — hand-build the handshake |
| When              | one-shot prologue, per role               | same: lean LOADER small, fat COMPUTE large    | YES — direct |

### Transferable blueprint for #323

Launch lean. **LOADER waves `s_alloc_vgpr(small)`** (the 24/32 analog) and stream B
from HBM. **COMPUTE waves `s_alloc_vgpr(large)`** (the 240/160 analog) grow into the
freed pool to hold fatter accumulator tiles. The register split is **one-time at
entry, keyed on wave role** — which resolves the earlier "can't reshape a live
accumulator mid-k-loop" question: NVIDIA doesn't either; budget is set by role, once.

### The two things that do NOT port (real design work)

1. **No mbarrier / TMA.** Build the loader→compute handoff from `s_barrier` + LDS
   flags (we already have the LDS-ring substrate from the KWIN work).
2. **`s_alloc_vgpr` can fail.** The compute wave's grow must be provably satisfiable
   from the pool (loaders must release enough first), or handle SCC=0. NVIDIA's
   `__launch_bounds__` guarantees headroom statically; we must guarantee it by
   construction.

## Corrections to the Gemma 12B draft (for the record)

- Said warpgroup = "32 warps in a 128-thread block". WRONG — 4 warps = 128 threads.
- Said operand "typically 32 or 64". WRONG — [24,256], multiple of 8.
- Said realloc happens "during the handoff phase of the mainloop" (per-iteration).
  WRONG — one-time prologue, keyed on role.
- Cited non-existent CUTLASS paths ("cutlass_kernel_types", "cutlass_gemm"
  folders). Real path: `include/cutlass/arch/reg_reconfig.h`.

## Sources

- CUTLASS source: `include/cutlass/arch/reg_reconfig.h` (github.com/NVIDIA/cutlass)
- Colfax: https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
- PTX ISA setmaxnreg ref (operand constraints corroborated via CUTLASS docs +
  NVIDIA dev forums): docs.nvidia.com/cuda/parallel-thread-execution §9.7.20.5
