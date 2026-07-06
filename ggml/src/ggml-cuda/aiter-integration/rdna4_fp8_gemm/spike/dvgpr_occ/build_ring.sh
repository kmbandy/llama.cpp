#!/usr/bin/env bash
# build_ring.sh — FIX 1a D=2 double-buffered ring-of-slots bin (occ_kernel_dsws_ring.s).
#   Bin name MUST match occ_dispatch.cpp DSWS2_RING path: occ_dsws2_<c>c<a>a<b>b_ring_gd.bin
#   OFFLINE/CPU only — assemble + objcopy, NO GPU. Usage: ./build_ring.sh [NCOMP NAFEED NBFEED]
#   Env: PHASEPROBE={0|1} NOCFLUSH={0|1} CSTORE={0|1} SLEEPN=N DIAG=0
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mkring() { # $1=NCOMP $2=NAFEED $3=NBFEED
  local tag="occ_dsws2_${1}c${2}a${3}b_ring_gd"
  nice -19 ionice -c3 "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
     -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-0} \
     -Wa,-defsym,NCOMP=$1 -Wa,-defsym,NAFEED=$2 -Wa,-defsym,NBFEED=$3 \
     -Wa,-defsym,PHASEPROBE=${PHASEPROBE:-0} -Wa,-defsym,NOCFLUSH=${NOCFLUSH:-0} -Wa,-defsym,CSTORE=${CSTORE:-0} \
     -Wa,-defsym,SLEEPN=${SLEEPN:-2} -Wa,-defsym,TFPROBE=${TFPROBE:-0} \
     -c occ_kernel_dsws_ring.s -o "$tag.o" 2>/tmp/ring_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; \
        echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B .text)  [PHASEPROBE=${PHASEPROBE:-0} NOCFLUSH=${NOCFLUSH:-0}]"; } \
   || { echo "  FAIL $tag"; sed -n '1,20p' /tmp/ring_build.err; fail=1; }
}
c=${1:-4}; a=${2:-2}; b=${3:-2}
echo "== FIX 1a ring bin (occ_kernel_dsws_ring.s; G=6 SEGK=64 FM=2 FN=4 D=2) =="
mkring "$c" "$a" "$b"
echo "ring build done. fail=$fail"
exit $fail
