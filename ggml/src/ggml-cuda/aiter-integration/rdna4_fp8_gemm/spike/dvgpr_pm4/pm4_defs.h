// pm4_defs.h  (MAD-304 T1)
//
// gfx12 / gfx1201 (RDNA4, R9700) compute-dispatch register definitions and
// COMPUTE_PGM_RSRC builders for the raw-PM4 dynamic-VGPR unlock harness.
//
// THE WHOLE POINT of this spike lives in BuildPgmRsrc2(): on GFX12 bit 6 of
// COMPUTE_PGM_RSRC2 is DYNAMIC_VGPR (it was TRAP_PRESENT pre-GFX12). kfdtest's
// Dispatch.cpp deliberately skips bit 6 for FAMILY_GFX12; HIP/AQL dispatch lets
// the MES firmware reprogram the dispatch and likewise drops it. By writing
// COMPUTE_PGM_RSRC2 ourselves via SET_SH_REG in a raw PM4 IB (the kfdtest /
// PAL / Vulkan method) the CP consumes our value verbatim, MES out of the loop.
// Setting bit 6 here is the entire "lift".
//
// Every numeric constant below is transcribed from pinned upstream sources
// (ROCm/ROCR-Runtime @ ba56a24c) -- NO values are guessed, because a wrong
// SET_SH_REG offset writes a garbage register and can hang the compute queue.
//
//   register dword offsets : asic_reg/gfx_7_2_d.h  (the COMPUTE_* SH-register
//                            block is stable CIK->GFX12; kfdtest itself reuses
//                            this CIK header for gfx12 dispatch, which is the
//                            behavioural proof the offsets are correct there)
//   mmCOMPUTE_PGM_RSRC3    : Dispatch.cpp:33  (0x2e2d; absent from CIK header)
//   RSRC1/RSRC2 bit shifts : asic_reg/gfx_7_2_sh_mask.h
//   PERSISTENT_SPACE_START : gfx_7_2_enum.h:517 (0x2c00, applied inside
//                            PM4SetShaderRegPacket; we pass absolute offsets)
//   FAMILY_* enum          : KFDTestFlags.hpp (via compat shim)
#ifndef RDNA4_DVGPR_PM4_DEFS_H
#define RDNA4_DVGPR_PM4_DEFS_H

#include <cstdint>

#include "KFDBaseComponentTest.hpp"  // compat shim: KfdFamilyId, hsakmt_is_dgpu()
#include "PM4Packet.hpp"             // vendored kfdtest packet encoders

// ---- COMPUTE_* register dword offsets (absolute; gfx_7_2_d.h) --------------
// SET_SH_REG inside PM4SetShaderRegPacket subtracts PERSISTENT_SPACE_START
// (0x2c00) from these, so e.g. RSRC2 -> packet reg_offset 0x213.
static const unsigned int mmCOMPUTE_START_X         = 0x2e04;  // block of 8: START_X/Y/Z, NUM_THREAD_X/Y/Z, PIPESTAT, PERFCOUNT
static const unsigned int mmCOMPUTE_PGM_LO          = 0x2e0c;  // gfx9+ block of 6 (pgm_lo/hi + scratch)
static const unsigned int mmCOMPUTE_PGM_RSRC1       = 0x2e12;  // block of 2: RSRC1, RSRC2
static const unsigned int mmCOMPUTE_PGM_RSRC2       = 0x2e13;
static const unsigned int mmCOMPUTE_PGM_RSRC3       = 0x2e2d;  // Dispatch.cpp:33
static const unsigned int mmCOMPUTE_RESOURCE_LIMITS = 0x2e15;
static const unsigned int mmCOMPUTE_TMPRING_SIZE    = 0x2e18;
static const unsigned int mmCOMPUTE_RESTART_X       = 0x2e1b;  // block of 4
static const unsigned int mmCOMPUTE_USER_DATA_0     = 0x2e40;  // block of 16: s0..s15 preload source

// ---- COMPUTE_PGM_RSRC1 bit shifts (gfx_7_2_sh_mask.h) ----------------------
static const unsigned int RSRC1_VGPRS_SHIFT      = 0x0;
static const unsigned int RSRC1_SGPRS_SHIFT      = 0x6;
static const unsigned int RSRC1_PRIORITY_SHIFT   = 0xa;
static const unsigned int RSRC1_FLOAT_MODE_SHIFT = 0xc;
static const unsigned int RSRC1_PRIV_SHIFT       = 0x14;

// ---- COMPUTE_PGM_RSRC2 bit shifts (gfx_7_2_sh_mask.h) ----------------------
static const unsigned int RSRC2_SCRATCH_EN_SHIFT     = 0x0;
static const unsigned int RSRC2_USER_SGPR_SHIFT      = 0x1;   // mask 0x3e
static const unsigned int RSRC2_TRAP_PRESENT_SHIFT   = 0x6;   // mask 0x40  <-- GFX12 DYNAMIC_VGPR
static const unsigned int RSRC2_TGID_X_EN_SHIFT      = 0x7;   // mask 0x80
static const unsigned int RSRC2_TIDIG_COMP_CNT_SHIFT = 0xb;   // mask 0x1800
static const unsigned int RSRC2_EXCP_EN_MSB_SHIFT    = 0xd;   // mask 0x6000

// On GFX12, COMPUTE_PGM_RSRC2 bit 6 (formerly TRAP_PRESENT) == DYNAMIC_VGPR.
static const unsigned int RSRC2_DYNAMIC_VGPR_MASK = (1u << RSRC2_TRAP_PRESENT_SHIFT);  // 0x40

// gfx1201 raw probe uses 32 VGPRs (VGPRS field = 4 -> 4*8 = 32), matching
// Dispatch.cpp:141. Plenty for the tiny STATUS-read probe.
static const unsigned int PROBE_VGPRS_FIELD = 0x4;

// ---- COMPUTE_PGM_RSRC1 for the gfx12 probe wave ---------------------------
// Mirrors Dispatch.cpp:136-141 for FAMILY_GFX12 (SGPRS term is 0 on >=GFX12),
// SpiPriority 0. priv lets the wave run privileged to avoid CWSR/trap WAs on
// some gfx11/12 asics (Dispatch.cpp m_NeedCwsrWA); default false here.
static inline uint32_t BuildPgmRsrc1(bool priv) {
    uint32_t r = 0;
    r |= (0xc0u << RSRC1_FLOAT_MODE_SHIFT);            // FLOAT_MODE = 0xc0
    r |= (PROBE_VGPRS_FIELD << RSRC1_VGPRS_SHIFT);     // 32 VGPRs
    if (priv)
        r |= (1u << RSRC1_PRIV_SHIFT);
    return r;                                          // priv=0 -> 0x000C0004
}

// ---- COMPUTE_PGM_RSRC2 for the gfx12 probe wave ---------------------------
// Mirrors Dispatch.cpp:143-161 for FAMILY_GFX12, no scratch:
//   USER_SGPR=4 (s0..s3 preloaded from USER_DATA_0..3 -> our out ptr in s0:s1),
//   TGID_X_EN=1, TIDIG_COMP_CNT=1, EXCP_EN_MSB=1, SCRATCH_EN=0.
//   Bit 6 (TRAP_PRESENT) is SKIPPED on GFX12 upstream -> baseline static VGPR.
// dynVgpr=true sets bit 6 == DYNAMIC_VGPR: the unlock.
static inline uint32_t BuildPgmRsrc2(bool dynVgpr) {
    uint32_t r = 0;
    r |= (4u << RSRC2_USER_SGPR_SHIFT)      & 0x3e;    // 4 user SGPRs
    r |= (1u << RSRC2_TGID_X_EN_SHIFT)      & 0x80;
    r |= (1u << RSRC2_TIDIG_COMP_CNT_SHIFT) & 0x1800;
    r |= (1u << RSRC2_EXCP_EN_MSB_SHIFT)    & 0x6000;
    if (dynVgpr)
        r |= RSRC2_DYNAMIC_VGPR_MASK;                 // <-- THE LIFT (bit 6)
    return r;                                          // baseline 0x2888, lift 0x28C8
}

// ---- COMPUTE_DISPATCH_INITIATOR for a 1x1x1 wave32 gfx12 dGPU dispatch -----
// Dispatch.cpp:225-226 : 0x21 | (dgpu?0:0x1000) | (>=NV ? 0x8000 : 0).
//   bit0  COMPUTE_SHADER_EN
//   bit5  USE_THREAD_DIMENSIONS
//   bit15 CS_W32_EN (wave32) -- set for FAMILY>=NV
static inline uint32_t BuildDispatchInitiator() {
    uint32_t v = 0x00000021u;
    if (!hsakmt_is_dgpu())
        v |= 0x1000u;
    v |= 0x8000u;  // FAMILY_GFX12 >= FAMILY_NV
    return v;      // dGPU -> 0x00008021
}

#endif  // RDNA4_DVGPR_PM4_DEFS_H
