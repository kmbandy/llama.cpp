#!/usr/bin/env bash
# dyn-PROPER bins for the fair race: top dyn tiles (2x4, 4x2) across BATCH x DEFERGROW variants.
# All DYNVGPR=1, GENDIV=1. defergrow=accumulators-only footprint + lean-block single-buffer feed.
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
mk() { local fm=$1 fn=$2 batch=$3 dg=$4 tag=$5
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DYNVGPR=1 -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=$batch \
     -Wa,-defsym,GENDIV=1 -Wa,-defsym,DEFERGROW=$dg \
     -c occ_kernel_mbgemm.s -o "$tag.o" 2>/tmp/dp.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK  $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag : $(grep -m1 -iE error /tmp/dp.err|head -c 70)"; }; }
for t in "2 4 2x4" "4 2 4x2"; do
  set -- $t; fm=$1; fn=$2; nm=$3
  # prefetch BATCH amortization track (defergrow=0): B8, B32  (B1 already built as _gd)
  mk $fm $fn 8  0 occ_mbgemm_${nm}_b8_d1_gd
  mk $fm $fn 32 0 occ_mbgemm_${nm}_b32_d1_gd
  # DEFERGROW track: B1, B8, B32
  mk $fm $fn 1  1 occ_mbgemm_${nm}_b1_d1_dg
  mk $fm $fn 8  1 occ_mbgemm_${nm}_b8_d1_dg
  mk $fm $fn 32 1 occ_mbgemm_${nm}_b32_d1_dg
done
echo "dyn-proper bins built."
