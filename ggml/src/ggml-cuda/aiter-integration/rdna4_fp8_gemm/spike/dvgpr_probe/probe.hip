// dvgpr_probe: reads wave STATUS register bit 30 (DYN_VGPR_EN) and writes it to
// global memory. On gfx1201 a normal wave is NOT launched in dynamic-VGPR mode,
// so STATUS[30] == 0. If a kernel-descriptor with COMPUTE_PGM_RSRC3 bit 17
// (ENABLE_DYNAMIC_VGPR, gfx1250-only / reserved on gfx1201) set ever produced a
// wave with STATUS[30] == 1, gfx1201 silicon honored the reserved launch-enable.
//
// STATUS layout (gfx12 ISA 3.4.x): WAVE64 @ bit 29, DYN_VGPR_EN @ bit 30.
// s_getreg_b32 with hwreg(HW_REG_STATUS, 30, 1) extracts 1 bit at offset 30,
// right-justified into the destination SGPR.
#include <hip/hip_runtime.h>

extern "C" __global__ void dvgpr_probe(unsigned* out) {
    unsigned s;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_STATUS, 30, 1)" : "=s"(s));
    if (threadIdx.x == 0) {
        out[0] = s; // DYN_VGPR_EN: 1 = armed, 0 = not
    }
}
