#!/usr/bin/env bash
# NO-FEED framework-ceiling probe for the funnel: each shape's TOP-4 STATIC + TOP-4 DYN, run on its
# OWN kernel with the per-K feed REMOVED (operands loaded once, reused for all KT WMMAs). Isolates each
# config's compute ceiling -- "current potential" with DRAM bandwidth off the table. The no-feed TF
# delta between configs reveals real headroom; that ranking picks the top-2 of each group per shape.
#
# Respective kernels (NO cross-pollination -- dyn stays dyn, static stays static):
#   static  -> DYNVGPR=0, DEFERGROW=0  (no grow; static framework ceiling)
#   dyn pf  -> DYNVGPR=1, DEFERGROW=0  (grow + prefetch path; grow-tax intact)
#   dyn dg  -> DYNVGPR=1, DEFERGROW=1  (grow + DEFERGROW single-buffer; needs the new dg NOFEED guard)
# All NOFEED=1, GENDIV=1 (real ml8 non-pow2 N). Result is GARBAGE by design (perf probe; oracle BAD).
# Naming: occ_mbgemm_{FM}x{FN}_b{BATCH}_d{0|1}_{gd|dg}_nf.bin
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mk() { local fm=$1 fn=$2 batch=$3 dyn=$4 dg=$5 tag=$6
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DYNVGPR=$dyn -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=$batch \
     -Wa,-defsym,GENDIV=1 -Wa,-defsym,DEFERGROW=$dg -Wa,-defsym,NOFEED=1 \
     -c occ_kernel_mbgemm.s -o "$tag.o" 2>/tmp/nf.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag : $(grep -m1 -iE error /tmp/nf.err|head -c 90)"; fail=1; }; }

echo "== STATIC top-4 tiles (d0, NOFEED, b1) =="
for t in "2 8" "2 4" "4 4" "1 4" "8 2" "4 2"; do
  set -- $t; fm=$1; fn=$2
  mk $fm $fn 1 0 0 occ_mbgemm_${fm}x${fn}_b1_d0_gd_nf
done

echo "== DYN top-4 configs (d1, NOFEED), prefetch (gd) =="
#         FM FN BATCH
for c in "2 4 8" "2 4 32" "2 4 1" "1 4 8" "1 4 32" "4 2 8" "4 2 1" "4 1 8" "2 2 8" "2 2 32"; do
  set -- $c; fm=$1; fn=$2; b=$3
  mk $fm $fn $b 1 0 occ_mbgemm_${fm}x${fn}_b${b}_d1_gd_nf
done

echo "== DYN top-4 configs (d1, NOFEED), DEFERGROW (dg) -- exercises the new dg NOFEED guard =="
for c in "2 4 8" "2 4 32" "1 4 8" "2 2 1"; do
  set -- $c; fm=$1; fn=$2; b=$3
  mk $fm $fn $b 1 1 occ_mbgemm_${fm}x${fn}_b${b}_d1_dg_nf
done

echo "no-feed field built (target 20 bins). fail=$fail"
exit $fail
