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
     -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=${FM:-2} -Wa,-defsym,FN=4 -Wa,-defsym,G=${G:-6} -Wa,-defsym,SEGK=${SEGK:-64} \
     -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-0} -Wa,-defsym,POOL_N=${POOL_N:-3} -Wa,-defsym,ACC_N=${ACC_N:-1} -Wa,-defsym,WOFLUSH=${WOFLUSH:-0} \
     -Wa,-defsym,WAVES=${WAVES:-16} -Wa,-defsym,VBUDGET=${VBUDGET:-1536} \
     -Wa,-defsym,PHASEPROBE=${PHASEPROBE:-0} -Wa,-defsym,NOCFLUSH=${NOCFLUSH:-0} -Wa,-defsym,KMAJOR=${KMAJOR:-0} -Wa,-defsym,JDEPTH=${JDEPTH:-1} -Wa,-defsym,STAGGER=${STAGGER:-0} ${RELSTART:+-Wa,-defsym,RELSTART=${RELSTART}} ${BATONGATE:+-Wa,-defsym,BATONGATE=${BATONGATE}} ${GRELAX:+-Wa,-defsym,GRELAX=${GRELAX}} ${BATON_SEED:+-Wa,-defsym,BATON_SEED=${BATON_SEED}} -Wa,-defsym,MAXFAT=${MAXFAT:-0} -Wa,-defsym,STAGERS=${STAGERS:-4} -Wa,-defsym,DUTYPROBE=${DUTYPROBE:-0} -Wa,-defsym,NTLOAD=${NTLOAD:-0} -Wa,-defsym,RBU=${RBU:-1} -Wa,-defsym,NOFEED=${NOFEED:-0} -Wa,-defsym,MULTISLOT=${MULTISLOT:-0} -Wa,-defsym,MSCOMP=${MSCOMP:-${MULTISLOT:-0}} -Wa,-defsym,MSSCAN=${MSSCAN:-${MSCOMP:-${MULTISLOT:-0}}} -Wa,-defsym,MSDRAIN=${MSDRAIN:-${MSCOMP:-${MULTISLOT:-0}}} -Wa,-defsym,MSFEED=${MSFEED:-${MULTISLOT:-0}} -Wa,-defsym,BATCHASN=${BATCHASN:-0} -Wa,-defsym,DECENTASN=${DECENTASN:-0} -Wa,-defsym,SELFSERVE=${SELFSERVE:-0} -Wa,-defsym,SSWIN=${SSWIN:-8} -Wa,-defsym,PHIST=${PHIST:-0} -Wa,-defsym,NOBLOAD=${NOBLOAD:-0} -Wa,-defsym,NODSADD=${NODSADD:-0} -Wa,-defsym,NOWMMA=${NOWMMA:-0} -Wa,-defsym,BNDPROBE=${BNDPROBE:-0} -Wa,-defsym,RESVPROBE=${RESVPROBE:-0} -Wa,-defsym,BATCH=${BATCH:-1} -Wa,-defsym,INITBAR=${INITBAR:-1} -Wa,-defsym,TERMFIX=${TERMFIX:-1} -Wa,-defsym,DUTY_EVERY=${DUTY_EVERY:-64} -Wa,-defsym,CSTORE=${CSTORE:-0} \
     -Wa,-defsym,SLEEPN=${SLEEPN:-2} -Wa,-defsym,COORD_PERIOD=${COORD_PERIOD:-64} -Wa,-defsym,TFPROBE=${TFPROBE:-0} -Wa,-defsym,DEADMAN=${DEADMAN:-1} -Wa,-defsym,DEADMAN_TICKS=${DEADMAN_TICKS:-50000000} -Wa,-defsym,STAGINSTR=${STAGINSTR:-0} -Wa,-defsym,TRACE=${TRACE:-0} \
     -Wa,-defsym,FORENSICS=${FORENSICS:-0} -Wa,-defsym,BANKZERO=${BANKZERO:-1} -Wa,-defsym,FATGAUGE=${FATGAUGE:-0} \
     -c occ_kernel_dsws_flow.s -o "$tag.o" 2>/tmp/flow_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; \
        "$L/llvm-objcopy" -O binary --only-section=.lds_total "$tag.o" "$tag.lds" 2>/dev/null || rm -f "$tag.lds"; \
        echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B .text)  [POOL_N=${POOL_N:-3} SSWIN=${SSWIN:-8} PHASEPROBE=${PHASEPROBE:-0}] LDS=$(od -An -tu4 -N4 "$tag.lds" 2>/dev/null | tr -d ' ')B"; } \
   || { echo "  FAIL $tag"; rm -f "$tag.bin" "$tag.o"; sed -n '1,25p' /tmp/flow_build.err; fail=1; }   # DELETE the stale bin -- a failed build must never leave a runnable artifact behind
}
echo "== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=${WAVES:-16} G=${G:-6} FM=${FM:-2} SEGK=${SEGK:-64} POOL_N=${POOL_N:-3} VBUDGET=${VBUDGET:-1536}) =="
mkflow
echo "flow build done. fail=$fail"
exit $fail
