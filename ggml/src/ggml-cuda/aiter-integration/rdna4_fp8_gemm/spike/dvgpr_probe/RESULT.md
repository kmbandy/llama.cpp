# Dynamic-VGPR arm-probe on gfx1201 — RESULT: bit ignored (clean negative, no hang)

**Date:** 2026-06-15. **Hardware:** AMD R9700 / gfx1201 (RDNA4), HIP device 0.

## Question
Does gfx1201/RDNA4 arm "dynamic-VGPR mode" (wave `STATUS.DYN_VGPR_EN`, bit 30) when the
gfx1250-only `COMPUTE_PGM_RSRC3.ENABLE_DYNAMIC_VGPR` bit (bit 17) — *reserved* on gfx1201 —
is set in the kernel descriptor and the kernel is dispatched via the normal HIP module path?

## Method
- `probe.hip` — a 1-wave kernel that reads its own `STATUS[30]` via
  `s_getreg_b32 ..., hwreg(HW_REG_WAVE_STATUS, 30, 1)` and writes it to global memory.
  Compiled to `probe.hsaco` (`hipcc --genco --offload-arch=gfx1201`, RAM-capped).
- `patch_kd.py` — locates the `dvgpr_probe.kd` kernel descriptor, sets **bit 17** of
  `compute_pgm_rsrc3` (KD byte offset **44**, verified vs `AMDHSAKernelDescriptor.h`).
  `0x00000010 → 0x00020010`, exactly one bit changed (`cmp -l` confirmed).
- `harness.cpp` — `hipModuleLoad` + `hipModuleLaunchKernel` (1 wave), reads back the value.

## Results
| dispatch | `DYN_VGPR_EN` | notes |
|---|---|---|
| control (`probe.hsaco`, bit 17 = 0) | **0** | proves the harness + STATUS read (normal wave) |
| **patched (`probe_patched.hsaco`, bit 17 = 1)** | **0** | **gfx1201 ignores the gfx1250 enable bit** |

- `hipModuleLoad` of the patched binary returns `hipSuccess` — ROCr does **not** validate/reject
  the reserved bit, so the CP *does* see bit 17 at dispatch; it is the **hardware** that ignores it.
- The patched dispatch completed cleanly (exit 0) and the GPU stayed responsive — the
  reserved-bit gamble did **not** hang the WGP/CP. The clean-negative, not the hang.

## Conclusion
**On the R9700 / gfx1201 with ROCm 7.2, dynamic-VGPR mode cannot be armed from userspace.**
The descriptor enable bit is silently ignored by the gfx1201 command processor (consistent with
either the CP not reading bit 17 on this arch, or the chip-wide `SQ_DYN_VGPR` config being off —
both privileged / firmware-level, not userspace-reachable). Combined with the toolchain analysis
(`../FINDINGS.md`): the launch-enable is a **gfx1250 feature**, and RDNA4 wires dynamic-VGPR only
through the `amdgpu_cs_chain` path, which the compute AQL-dispatch path does not use.

**Implication for the 300+ goal:** unlocking the dynamic-VGPR occupancy lever on the R9700 is a
genuine **AMD-side change** — backport the gfx1250 `COMPUTE_PGM_RSRC3.ENABLE_DYNAMIC_VGPR` mechanism
(+ `.dynamic_vgpr_en` metadata for `amdgpu_kernel` + ROCr dispatch plumbing) to gfx1201, or expose
the cs_chain compute-dispatch path. It is not reachable by any kernel-, descriptor-, or
runtime-level hack on the current stack. The oracle-green fp8/ml8 GEMM that would consume it is
already written (`../../gemm_wmma.hip`, `../gemm_wmma_raw_intrinsic_verified.hip`).

Files: `probe.hip`, `patch_kd.py`, `harness.cpp`, `loadonly.cpp` (sources committed; the
`.hsaco` binaries + compiled harnesses are reproducible build artifacts, not committed).
