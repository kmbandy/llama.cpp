#!/bin/bash
# gate_duty.sh -- THE DUTY-CYCLE INVARIANT GATE (2026-07-20)
#
# DSWS beats a static GEMM only because split-K keeps each wave's VGPR peak BRIEF, so peaks can be
# phase-offset and many waves time-multiplex one budget. TIME AT PEAK ~ JDEPTH*SEGK. Raise it and
# the trapezoid becomes a full-K square wave and the dyn-VGPR moat is gone (dyn==static, measured).
#
# The kernel's own flush arithmetic (flush/WMMA = 128/SEGK) argues FOR raising SEGK and says nothing
# about duty cycle -- that asymmetry is what walked me into proposing SEGK 256->1024. This gate
# makes the invariant executable instead of advisory.
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
CC=/opt/rocm/llvm/bin/clang
BASE=(-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=1 -Wa,-defsym,FN=4 -Wa,-defsym,G=6
  -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,POOL_N=1 -Wa,-defsym,ACC_N=3
  -Wa,-defsym,WOFLUSH=0 -Wa,-defsym,WAVES=30 -Wa,-defsym,VBUDGET=1536 -Wa,-defsym,PHASEPROBE=0
  -Wa,-defsym,NOCFLUSH=0 -Wa,-defsym,STAGGER=1 -Wa,-defsym,KMAJOR=0
  -Wa,-defsym,MAXFAT=0 -Wa,-defsym,STAGERS=4 -Wa,-defsym,NTLOAD=0 -Wa,-defsym,DUTYPROBE=0
  -Wa,-defsym,RBU=1 -Wa,-defsym,NOFEED=0 -Wa,-defsym,MULTISLOT=0 -Wa,-defsym,MSCOMP=0
  -Wa,-defsym,MSSCAN=0 -Wa,-defsym,MSDRAIN=0 -Wa,-defsym,MSFEED=0 -Wa,-defsym,BATCHASN=0
  -Wa,-defsym,DECENTASN=1 -Wa,-defsym,DUTY_EVERY=64 -Wa,-defsym,CSTORE=0 -Wa,-defsym,SLEEPN=2
  -Wa,-defsym,COORD_PERIOD=64 -Wa,-defsym,DEADMAN=1 -Wa,-defsym,DEADMAN_TICKS=50000000
  -Wa,-defsym,TRACE=0 -Wa,-defsym,BANKZERO=1 -Wa,-defsym,FATGAUGE=0 -Wa,-defsym,SELFSERVE=1
  -Wa,-defsym,FORENSICS=0 -Wa,-defsym,STAGINSTR=1 -Wa,-defsym,TFPROBE=1 -Wa,-defsym,PHIST=0)
rc=0
try() {  # $1=SEGK $2=JDEPTH $3=expect(pass|refuse) $4=extra
  local out
  out=$($CC "${BASE[@]}" -Wa,-defsym,SEGK=$1 -Wa,-defsym,JDEPTH=$2 $4 \
        -c occ_kernel_dsws_flow.s -o /dev/null 2>&1 | grep -c "DUTY-CYCLE INVARIANT VIOLATED")
  local got; [ "$out" = "0" ] && got=pass || got=refuse
  if [ "$got" = "$3" ]; then printf "   OK   SEGK=%-5s J=%-2s -> %-6s (J*SEGK=%s)\n" "$1" "$2" "$got" "$(($1*$2))"
  else printf "   *** FAIL SEGK=%-5s J=%-2s -> %s, expected %s\n" "$1" "$2" "$got" "$3"; rc=1; fi
}

echo "== POSITIVE CONTROL: every sanctioned/historical geometry must still assemble =="
try 32  1 pass; try 64  1 pass; try 128 1 pass
try 256 1 pass          # <- the config of record (15.5 TF)
try 128 2 pass          # <- J*SEGK == 256, the amortization depth the testing log already groups by

echo "== NEGATIVE CONTROL: the gate MUST be able to fail (a gate that can't fail is worthless) =="
try 512  1 refuse       # <- exactly what I proposed on 2026-07-20
try 1024 1 refuse
try 256  2 refuse       # <- the SAME violation via JDEPTH, the trap labelled "FREE" in the source
try 256  4 refuse

echo "== OVERRIDE is deliberate, not accidental =="
try 1024 1 pass "-Wa,-defsym,DUTY_OVERRIDE=1"

echo "gate_duty rc=$rc"; exit $rc
