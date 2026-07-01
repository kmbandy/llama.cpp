#!/usr/bin/env bash
# build_dsws.sh — MAD-305 DSWS static 3-role substrate bins (occ_kernel_coop.s, DSWS=1).
#   FM=2 FN=4 POOLTERM=1 fixed (the v1 coop tile; role counts are the swept defsyms).
#   Bin name MUST match occ_dispatch.cpp --dsws: occ_dsws_<c>c<a>a<b>b_r<RINGD>[_dyn]_gd.bin
#   OFFLINE/CPU only — assemble + RGA, no GPU. Usage: ./build_dsws.sh [static] [rga]
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mk() { # $1=NCOMP $2=NAFEED $3=NBFEED $4=RINGD $5=DYN
  local dtag=""; [ "$5" = "1" ] && dtag="_dyn"
  local tag="occ_dsws_${1}c${2}a${3}b_r${4}${dtag}_gd"
  nice -19 ionice -c3 "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DSWS=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,POOLTERM=1 -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-1} \
     -Wa,-defsym,NCOMP=$1 -Wa,-defsym,NAFEED=$2 -Wa,-defsym,NBFEED=$3 \
     -Wa,-defsym,RINGD=$4 -Wa,-defsym,DYNVGPR=$5 \
     -c occ_kernel_coop.s -o "$tag.o" 2>/tmp/dsws_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag"; sed -n '1,15p' /tmp/dsws_build.err; fail=1; }
}

mk2() { # $1=NCOMP $2=NAFEED $3=NBFEED  (DSWS2 v2 substrate, occ_kernel_dsws.s; G=6 SEGK=64 FM=2 FN=4)
  local tag="occ_dsws2_${1}c${2}a${3}b_gd"
  nice -19 ionice -c3 "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
     -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-1} \
     -Wa,-defsym,NCOMP=$1 -Wa,-defsym,NAFEED=$2 -Wa,-defsym,NBFEED=$3 \
     -c occ_kernel_dsws.s -o "$tag.o" 2>/tmp/dsws2_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B)"; } \
   || { echo "  FAIL $tag"; sed -n '1,15p' /tmp/dsws2_build.err; fail=1; }
}

echo "== DSWS static 3-role bins (FM=2 FN=4 POOLTERM=1) =="
#  NCOMP NAFEED NBFEED RINGD DYN
mk 4 2 2 2 1 ; mk 4 2 2 2 0     # 4c2a2b
mk 6 1 1 2 1 ; mk 6 1 1 2 0     # 6c1a1b
mk 2 3 3 2 1 ; mk 2 3 3 2 0     # 2c3a3b
echo "dsws build done. fail=$fail"

echo "== DSWS2 v2 substrate scaffold bin (occ_kernel_dsws.s; G=6 SEGK=64) =="
mk2 4 2 2                        # 4c2a2b
echo "dsws2 build done. fail=$fail"

# RGA gate (offline static analysis; 0-spill is the bar). Runs by default unless 'norga' passed.
if [ "${1:-}" != "norga" ] && [ "${2:-}" != "norga" ]; then
  echo "== RGA gate (4c2a2b dyn — compute peak-live is tile-fixed, representative) =="
  KSRC=occ_kernel_coop.s ./rga_check.sh dsws_build_4c2a2b \
     DSWS=1 FM=2 FN=4 NCOMP=4 NAFEED=2 NBFEED=2 RINGD=2 POOLTERM=1 SAFEPROBE=1 DYNVGPR=1 2>&1 \
     | grep -E "gfx1201,|livereg" || true
fi
exit $fail
