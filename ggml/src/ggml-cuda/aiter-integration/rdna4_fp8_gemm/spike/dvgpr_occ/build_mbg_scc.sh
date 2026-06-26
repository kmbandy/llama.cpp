#!/usr/bin/env bash
# Rebuild ONLY the bins the plain `--mbgemm` (BATCH=32, no --fat) run consumes, now with the
# SCC-retry grow guard baked into occ_kernel_mbgemm.s. Safe arms only (no umr 4x4/5x4 dyn).
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
asm() { # dv fm fn extra outbase
  local dv=$1 fm=$2 fn=$3 extra=$4 out=$5
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=32 $extra \
     -c occ_kernel_mbgemm.s -o "$out.o"
  "$L/llvm-objcopy" -O binary --only-section=.text "$out.o" "$out.bin"
  echo "  built $out.bin ($(wc -c < "$out.bin")B)"
}
# dyn arms (DYNVGPR=1 -> SCC-retry path), within 128 cap
asm 1 1 1 "" occ_mbgemm_1x1_b32_d1
asm 1 2 2 "" occ_mbgemm_2x2_b32_d1
asm 1 2 4 "" occ_mbgemm_2x4_b32_d1
# static arms (DYNVGPR=0 -> no grow, 256-VGPR reservation, no umr)
asm 0 4 4 "" occ_mbgemm_4x4_b32_d0
asm 0 5 4 "" occ_mbgemm_5x4_b32_d0
# nofeed framework-ceiling probe (dyn 2x4)
asm 1 2 4 "-Wa,-defsym,NOFEED=1" occ_mbgemm_2x4_b32_nf
echo "ALL SAFE MBGEMM BINS REBUILT WITH SCC-RETRY GUARD"
