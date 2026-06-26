#!/usr/bin/env bash
# MAD-305 HYBRID COOPERATIVE kernel (occ_kernel_coop.s). Builds the PM4 .bin per (FM,FN,P,RINGD,BATCH,dyn)
# cell + an RGADESC .o for livereg. Bin name matches run_mbcoop: occ_coop_<FM>x<FN>_p<P>_r<RINGD>_b<B>_d<dyn>_gd.bin
# All GENDIV (the ml8 N are non-pow2; the harness drives useGenDiv=true). OFFLINE/CPU -- safe, no GPU.
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mk() { local fm=$1 fn=$2 p=$3 ringd=$4 batch=$5 dyn=$6
  local tag="occ_coop_${fm}x${fn}_p${p}_r${ringd}_b${batch}_d${dyn}_gd"
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,P=$p -Wa,-defsym,RINGD=$ringd \
     -Wa,-defsym,BATCH=$batch -Wa,-defsym,DYNVGPR=$dyn \
     -c occ_kernel_coop.s -o "$tag.o" 2>/tmp/coop.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag"; sed -n '1,12p' /tmp/coop.err; fail=1; }
  # RGADESC .o (livereg only; never dispatched)
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,P=$p -Wa,-defsym,RINGD=$ringd \
     -Wa,-defsym,BATCH=$batch -Wa,-defsym,DYNVGPR=$dyn -Wa,-defsym,RGADESC=1 \
     -c occ_kernel_coop.s -o "${tag}_rga.o" 2>/dev/null || true
}

echo "== HYBRID COOPERATIVE bins =="
#  FM FN P RINGD BATCH DYN
mk 2 4 1 2 1 1          # B2a: P=1 protocol bring-up (dyn)
mk 2 4 1 2 1 0          # static fallback (uniform alloc) for A/B isolation
echo "coop build done. fail=$fail"
exit $fail
