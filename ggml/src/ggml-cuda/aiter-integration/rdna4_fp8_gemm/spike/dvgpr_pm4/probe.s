// probe.s - gfx1201 raw-ISA compute shader for the PM4 dyn-VGPR probe.
//
// Reads its own wave STATUS bit 30 (DYN_VGPR_EN) and stores it to the output
// pointer. This is NOT a .hsaco / amdhsa kernel - it is raw ISA meant to be
// loaded at a GPU address and dispatched via a raw PM4 IB (the kfdtest method),
// so there is no kernel descriptor and no MES in the loop. COMPUTE_PGM_RSRC2
// (incl. bit 6 = DYNAMIC_VGPR on GFX12) is written by the harness via SET_SH_REG.
//
// Calling convention (matches the harness pm4_dispatch.cpp):
//   COMPUTE_USER_DATA_0/1 -> s0:s1 = output pointer (uint32*) at wave entry.
//   wave32, single workgroup, single wave.
//
// STATUS[30] = DYN_VGPR_EN (RDNA4 ISA: "the wave is running using Dynamic VGPRs").
// Assemble:
//   /opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
//       -c probe.s -o probe.o
//   /opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=.text probe.o probe.bin

	.text
	.globl	dvgpr_probe
	.p2align 8
	.type	dvgpr_probe,@function
dvgpr_probe:
	; s0:s1 = output pointer (from COMPUTE_USER_DATA_0/1)
	s_getreg_b32 s2, hwreg(HW_REG_STATUS, 30, 1)   ; s2 = STATUS.DYN_VGPR_EN (1 bit @ offset 30)
	v_mov_b32_e32 v0, s2                            ; data = DYN_VGPR_EN
	v_mov_b32_e32 v1, 0                             ; byte offset 0 into the buffer
	global_store_b32 v1, v0, s[0:1]                 ; *(uint32*)out = DYN_VGPR_EN
	s_waitcnt vmcnt(0)
	s_endpgm
	.size	dvgpr_probe, .-dvgpr_probe
