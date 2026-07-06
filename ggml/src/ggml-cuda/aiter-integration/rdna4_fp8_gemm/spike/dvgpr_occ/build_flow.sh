#!/usr/bin/env bash
# build_flow.sh — FIX 1 (flow economy) bin (occ_kernel_dsws_flow.s). N-deep pool + ROLE mailbox +
#   coordinator. Bin name matches occ_dispatch.cpp DSWS2_FLOW path: occ_dsws2_<c>c<a>a<b>b_flow_gd.bin
#   OFFLINE/CPU only. Usage: ./build_flow.sh [NCOMP NAFEED NBFEED]
#   Env: POOL_N=3 PHASEPROBE={0|1} NOCFLUSH={0|1} CSTORE={0|1} SLEEPN=N COORD_PERIOD=N DIAG=0
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0
mkflow() { # EMERGENT economy: no mix args. Env: WAVES VBUDGET G SEGK POOL_N ACC_N ...
  local tag="occ_dsws2_w${WAVES:-16}_flow_gd"
  nice -19 ionice -c3 "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=${G:-6} -Wa,-defsym,SEGK=${SEGK:-64} \
     -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-0} -Wa,-defsym,POOL_N=${POOL_N:-3} -Wa,-defsym,ACC_N=${ACC_N:-1} -Wa,-defsym,WOFLUSH=${WOFLUSH:-0} \
     -Wa,-defsym,WAVES=${WAVES:-16} -Wa,-defsym,VBUDGET=${VBUDGET:-1536} \
     -Wa,-defsym,PHASEPROBE=${PHASEPROBE:-0} -Wa,-defsym,NOCFLUSH=${NOCFLUSH:-0} -Wa,-defsym,CSTORE=${CSTORE:-0} \
     -Wa,-defsym,SLEEPN=${SLEEPN:-2} -Wa,-defsym,COORD_PERIOD=${COORD_PERIOD:-64} -Wa,-defsym,TFPROBE=${TFPROBE:-0} -Wa,-defsym,DEADMAN=${DEADMAN:-1} -Wa,-defsym,DEADMAN_TICKS=${DEADMAN_TICKS:-50000000} -Wa,-defsym,STAGINSTR=${STAGINSTR:-0} -Wa,-defsym,TRACE=${TRACE:-0} \
     -c occ_kernel_dsws_flow.s -o "$tag.o" 2>/tmp/flow_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; \
        echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B .text)  [POOL_N=${POOL_N:-3} PHASEPROBE=${PHASEPROBE:-0}]"; } \
   || { echo "  FAIL $tag"; sed -n '1,25p' /tmp/flow_build.err; fail=1; }
}
echo "== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=${WAVES:-16} G=${G:-6} SEGK=${SEGK:-64} POOL_N=${POOL_N:-3} VBUDGET=${VBUDGET:-1536}) =="
mkflow
echo "flow build done. fail=$fail"
exit $fail
