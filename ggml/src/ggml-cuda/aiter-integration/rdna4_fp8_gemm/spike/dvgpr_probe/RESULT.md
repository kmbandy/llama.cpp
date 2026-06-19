# Dynamic-VGPR arm-probe on gfx1201 — RESULT: bit ignored (clean negative, no hang)

> ⚠️ **VERDICT OVERTURNED (2026-06-16).** This doc's conclusion — "dynamic-VGPR cannot be
> armed from userspace on gfx1201 / AMD-side wall / not reachable by any hack" — is **WRONG**
> and applies only to the **HIP-module / kernel-descriptor path tested here** (which sets the
> *gfx1250-only* `COMPUTE_PGM_RSRC3` bit 17, genuinely reserved on gfx1201). The **correct**
> gfx1201 enable is `COMPUTE_PGM_RSRC2` **bit 6** (DYNAMIC_VGPR), and it **IS armable from
> userspace via raw PM4** on a KFD compute queue — **PROVEN in MAD-304** (`../dvgpr_pm4/RESULT.md`,
> commit `133f9d151`): deterministic `DYN_VGPR_EN` 0→1→0→1, no hang, MES bypassed. The chip-wide
> `SQ_DYN_VGPR` gate is already open (0xff). **Do NOT cite this doc as evidence dyn-VGPR is
> locked/dead on gfx1201.** Kept below for the historical probe record of the *HIP path only*.

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

## Conclusion (CORRECTED — the original "AMD-side wall" verdict is WRONG; see banner)
What this probe actually showed: **the gfx1250-only `COMPUTE_PGM_RSRC3` bit 17, set via the HIP
module / kernel-descriptor path, is ignored on gfx1201** — a true negative for *that bit on that
path*, and the *expected* no-op (bit 17 is genuinely reserved on GFX120; see UPDATE 2).

It does **NOT** mean dynamic-VGPR is unreachable on gfx1201. **dyn-VGPR IS armable on gfx1201
COMPUTE from userspace** — the correct enable is `COMPUTE_PGM_RSRC2` **bit 6** (DYNAMIC_VGPR),
written via **raw PM4** (`SET_SH_REG` into a KFD compute-queue IB, MES bypassed). **PROVEN in
MAD-304** (`../dvgpr_pm4/RESULT.md`, commit `133f9d151`): deterministic `DYN_VGPR_EN` 0→1→0→1, no
hang. The chip-wide `SQ_DYN_VGPR` gate is already open (live umr read: `0xff`). The HIP/ROCr path
fails only because ROCr/LLVM never emit bit 6 — **not** because the silicon refuses it.

**Implication for the 300+ goal:** the dynamic-VGPR occupancy lever **is reachable today** on the
R9700 via the raw-PM4 vehicle (`../dvgpr_occ/`) — which is the whole reason that vehicle exists.
The remaining work is engineering on that vehicle (big-reuse tiles past the 128/256-VGPR cap),
**not** an AMD-side change. The oracle-green fp8/ml8 GEMM that consumes it is already written
(`../../gemm_wmma.hip`, `../gemm_wmma_raw_intrinsic_verified.hip`).

Files: `probe.hip`, `patch_kd.py`, `harness.cpp`, `loadonly.cpp` (sources committed; the
`.hsaco` binaries + compiled harnesses are reproducible build artifacts, not committed).

---

## UPDATE 2 — we had patched the WRONG bit. Corrected probe (RSRC2 bit 6).

**The bit above (`COMPUTE_PGM_RSRC3` bit 17) is the gfx1250 (GFX125) enable.** On gfx1201
(GFX120) that bit is genuinely `RESERVED` — so its no-op is *expected*, not informative.

The **RDNA4 (gfx1200/gfx1201 = GFX120)** dynamic-VGPR enable is a different bit in a different
register, and it has been in our installed toolchain header all along — ROCm 7.2.3
`AMDHSAKernelDescriptor.h`:
```
157:  COMPUTE_PGM_RSRC2_GFX6_GFX11(ENABLE_TRAP_HANDLER, 6, 1),
158:  COMPUTE_PGM_RSRC2_GFX120(ENABLE_DYNAMIC_VGPR,    6, 1),   <- the RDNA4 enable
230:  COMPUTE_PGM_RSRC3_GFX125(ENABLE_DYNAMIC_VGPR,   17, 1),   <- what UPDATE 1 patched (gfx1250)
```
Documented for GFX120* in the ROCm 7.13.0-preview LLVM `AMDGPUUsage` doc:
> "Enables dynamic VGPR mode, where each wave allocates one VGPR chunk at launch and can request
> for additional space to use during execution in SQ. **Used by CP to set up
> COMPUTE_PGM_RSRC2.DYNAMIC_VGPR.**"

`COMPUTE_PGM_RSRC2` is at KD byte offset **52** (static_assert-verified in the header).
`patch_kd_rsrc2.py` flips bit 6 there (`0x84 -> 0xc4`, one bit).

### Corrected result
| dispatch | `DYN_VGPR_EN` | notes |
|---|---|---|
| control (`probe.hsaco`, RSRC2 bit 6 = 0) | **0** | normal wave |
| **patched (`probe_rsrc2_patched.hsaco`, RSRC2 bit 6 = 1)** | **0** | **right bit, still not armed. No hang, GPU healthy.** |

Probe **validated**: RDNA4 ISA line 1587 confirms `STATUS[30] = DYN_VGPR_EN`
("Indicates that the wave is running using Dynamic VGPRs"). So `0` is a true negative.

### Refined verdict — the gate is *below* the kernel descriptor
The per-dispatch enable bit is **necessary but not sufficient.** RDNA4 ISA §3.3.3 (line 1350):
> "A single-state (**chip-wide**) config register defines the maximum number of waves per SIMD
> that can be present when using dynamic VGPRs: **SQ_DYN_VGPR**."

If `SQ_DYN_VGPR.max_waves == 0` (never programmed), zero wave-slots can hold a dynamic-VGPR
wave, so the KD bit is inert → `DYN_VGPR_EN=0`. That is the most likely cause. So the open
question is no longer "can we set the bit" (yes — proven harmless) but **who programs the
chip-wide `SQ_DYN_VGPR`**:
- **open amdgpu KMD via MMIO** → portable, a kernel-driver patch (best case); or
- **closed MES firmware** (gfx1201 = `gc_12_0_1_mes.bin` / `gc_12_0_1_uni_mes.bin`, both present
  on this host) → AMD-side wall.

### RESOLVED (2026-06-16, MAD-304) — the "who programs SQ_DYN_VGPR / AMD-side wall" fork is closed
Both halves answered, and neither is an AMD-side wall:
1. **`SQ_DYN_VGPR` is already open.** Live umr read on this R9700: `regSQ_DYN_VGPR = 0xff`
   (WAVE_LIMIT=15, FWD_PROGRESS=1, MAX_BLOCK_ALLOC=7). The chip-wide gate is wide open — not off.
2. **The per-dispatch bit DOES arm it — when set via raw PM4, not via the HIP/ROCr path.** ROCr/LLVM
   rebuild `compute_pgm_rsrc2` at dispatch and never emit bit 6; raw PM4 (`SET_SH_REG` into a KFD
   compute-queue IB) sets `COMPUTE_PGM_RSRC2` bit 6 ourselves and the CP consumes it verbatim (MES
   bypassed). Result: deterministic `DYN_VGPR_EN` 0→1→0→1, no hang — **MAD-304**, `../dvgpr_pm4/`.

So the HIP-path `DYN_VGPR_EN=0` above was a *toolchain* gap (ROCr drops the bit), not silicon. The
only live cap is the **128-VGPR dyn block** (raise to 256 via `SQ_DYN_VGPR.BLOCK_SIZE=1`, a `sudo
umr` write). The raw-PM4 vehicle (`../dvgpr_occ/`) is the productionizing path.
