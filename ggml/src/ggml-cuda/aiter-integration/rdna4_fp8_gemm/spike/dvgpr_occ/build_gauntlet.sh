#!/usr/bin/env bash
# All gauntlet tile bins: GENDIV, prefetch, BATCH=1. d0=static (all tiles), d1=dyn (<=128 VGPR tiles only).
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
mk() { local dv=$1 fm=$2 fn=$3 out=$4
  if "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
       -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=1 -Wa,-defsym,GENDIV=1 \
       -c occ_kernel_mbgemm.s -o "$out.o" 2>/tmp/asm.err; then
     "$L/llvm-objcopy" -O binary --only-section=.text "$out.o" "$out.bin"
     echo "  OK  $out.bin ($(wc -c < "$out.bin")B)"
  else echo "  FAIL $out  ($(grep -m1 -iE 'error' /tmp/asm.err | head -c 80))"; fi; }
# dyn-able tiles: d0 + d1
for t in "1 1 1x1" "2 2 2x2" "1 4 1x4" "4 1 4x1" "2 4 2x4" "4 2 4x2"; do
  set -- $t; mk 0 $1 $2 occ_mbgemm_${3}_b1_d0_gd; mk 1 $1 $2 occ_mbgemm_${3}_b1_d1_gd
done
# static-only fat tiles: d0
for t in "4 4 4x4" "8 2 8x2" "2 8 2x8" "8 1 8x1"; do
  set -- $t; mk 0 $1 $2 occ_mbgemm_${3}_b1_d0_gd
done
