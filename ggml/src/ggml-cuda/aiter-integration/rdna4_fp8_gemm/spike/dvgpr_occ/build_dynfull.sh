#!/usr/bin/env bash
# COMPLETE dyn field: ALL dyn-able pow2 tiles <=128 VGPR, each x {pf(dg=0),dg(dg=1)} x BATCH{1,8,32}.
# 8 tiles x 2 feeds x 3 batches = 48 dyn bins. All DYNVGPR=1, GENDIV=1. Naming matches the harness:
#   occ_mbgemm_{FM}x{FN}_b{BATCH}_d1_{gd|dg}.bin   (gd = prefetch, dg = DEFERGROW grow-first)
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mk() { local fm=$1 fn=$2 batch=$3 dg=$4 tag=$5
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DYNVGPR=1 -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=$batch \
     -Wa,-defsym,GENDIV=1 -Wa,-defsym,DEFERGROW=$dg \
     -c occ_kernel_mbgemm.s -o "$tag.o" 2>/tmp/dp.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag : $(grep -m1 -iE error /tmp/dp.err|head -c 80)"; fail=1; }; }
# all dyn-able pow2 tiles <=128 VGPR
for t in "1 1 1x1" "1 2 1x2" "2 1 2x1" "2 2 2x2" "1 4 1x4" "4 1 4x1" "2 4 2x4" "4 2 4x2"; do
  set -- $t; fm=$1; fn=$2; nm=$3
  for b in 1 8 32; do
    mk $fm $fn $b 0 occ_mbgemm_${nm}_b${b}_d1_gd   # prefetch
    mk $fm $fn $b 1 occ_mbgemm_${nm}_b${b}_d1_dg   # DEFERGROW grow-first
  done
done
echo "dyn-full field built (target 48 bins). fail=$fail"
exit $fail
