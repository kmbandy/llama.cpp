# Arm RDNA4 dynamic VGPR on COMPUTE via raw PM4 (lift Vulkan's method)

**Goal:** launch a gfx1201 compute wave in dynamic-VGPR mode and read `STATUS.DYN_VGPR_EN == 1`
— the first known time dynamic VGPR is armed on the RDNA4 **compute** path.

## Why this can work when HIP/AQL could not

The card is hit the *same way* either path: the dispatch uses whatever is in
`mmCOMPUTE_PGM_RSRC2` (bit 6 = `DYNAMIC_VGPR` on GFX12). The difference is **who writes it**:

| Path | Who writes COMPUTE_PGM_RSRC2 | Result |
|---|---|---|
| HIP / ROCm compute (AQL) | **MES firmware** reads the kernel descriptor and programs the dispatch | drops bit 6 → static |
| **Vulkan/PAL & kfdtest (raw PM4)** | **userspace** writes `SET_SH_REG mmCOMPUTE_PGM_RSRC2` into the IB; the **CP** consumes it verbatim | bit 6 honored |

We proved every other layer is ready:
- Silicon: `regSQ_DYN_VGPR = 0xff` (WAVE_LIMIT=15) — chip-wide dyn-VGPR **enabled**.
- `rsrc2.6` is the documented GFX120 enable; setting it is harmless on silicon.
- The block was the **MES** dropping the bit on the AQL path. Raw PM4 bypasses the MES.

## The lift (one line)

ROCm's own `libhsakmt/tests/kfdtest/src/Dispatch.cpp` builds `pgmRsrc2` in userspace and writes
it via `PM4SetShaderRegPacket(mmCOMPUTE_PGM_RSRC1, {pgmRsrc1, pgmRsrc2}, 2)`. Lines 149-152:

```cpp
if (m_FamilyId < FAMILY_GFX12) {                 // <-- on GFX12 they SKIP bit 6,
    pgmRsrc2 |= (1 << COMPUTE_PGM_RSRC2__TRAP_PRESENT__SHIFT) & ...;  // because bit 6 became DYNAMIC_VGPR
}
```

Our change drops into that exact gap:
```cpp
if (m_FamilyId >= FAMILY_GFX12 && dynVgprEnable)
    pgmRsrc2 |= (1u << 6);   // COMPUTE_PGM_RSRC2.DYNAMIC_VGPR
```
`pgmRsrc1.VGPRS = 0x4` (32 VGPRs) is a fine initial block allocation for a tiny probe.

## Architecture (self-contained, vendors kfdtest's method, not its gtest scaffolding)

- `probe.s` — gfx1201 shader: read `STATUS[30]` via `s_getreg_b32 hwreg(HW_REG_STATUS,30,1)`,
  store to the output pointer passed in `COMPUTE_USER_DATA`, `s_endpgm`. Hand-assembled to raw
  bytes (NOT a .hsaco — raw ISA loaded at a GPU address for PM4 dispatch).
- `pm4_dispatch.cpp` — links `/opt/rocm/lib/libhsakmt.a` (+ `hsakmt.h`). Does:
  `hsaKmtOpenKFD` → `AcquireSystemProperties`/`GetNodeProperties` (find gfx1201 node)
  → `hsaKmtAllocMemory`+`MapMemoryToGPU` for ISA / output / IB / (scratch)
  → `hsaKmtCreateQueue(node, HSA_QUEUE_COMPUTE, ...)`
  → build PM4 IB (ACQUIRE_MEM, SET_SH_REG for PGM_LO/HI + RSRC1/2/3 + RESOURCE_LIMITS + TMPRING
    + USER_DATA, DISPATCH_DIRECT, RELEASE_MEM fence) → submit (write ring + doorbell)
  → wait on fence → read output `DYN_VGPR_EN`.
- `pm4_defs.h` — vendored: PM4 type-3 header + the ~6 packet encodings + the gfx12 `mm*` register
  offsets + `PERSISTENT_SPACE_START`. (From kfdtest `PM4Packet.cpp` and its gfx12 asic_reg header,
  pinned ROCR-Runtime commit `ba56a24c`.)
- `build.sh` — RAM-capped compile (link libhsakmt.a + libhsa-runtime64). Author-run.

## Run protocol (discipline: baseline before lift)

1. **Baseline:** dispatch with bit 6 **clear** → MUST read `DYN_VGPR_EN = 0` (matches our AQL
   result). Proves the PM4 vehicle is correct before changing anything.
2. **Lift:** flip bit 6 → re-dispatch → read `DYN_VGPR_EN`.
   - **1** → dynamic VGPR armed on RDNA4 compute. 🎯
   - **0, no hang** → CP/SQ took the bit but the wave didn't enter dyn mode → need rsrc1 block
     encoding / scratch consistency; iterate on the KD/register fields.

## Risk

Raw PM4 on the **headless** R9700 (GPU[1], PCI 0000:42:00.0 — NOT the display GPU). A malformed
IB or bad dispatch can hang the compute queue / trigger a GPU MODE1 reset (recoverable; worst case
a card reset/reboot). kfdtest does exactly this dispatch safely on gfx12, so the path is sound.

## Status of the layer map (all proven this session)

silicon ✅ enabled · `rsrc2.6` ✅ documented+settable · LLVM ❌ won't emit (compute path) ·
ROCr ❌ no concept, passes KD through · MES ❌ drops bit on AQL → **bypassed by raw PM4 (this spike)**.
No prior art found (GitHub/forums/web) for arming dyn-VGPR on RDNA4 compute.
