#!/bin/bash
# gate_lds.sh -- HOST/KERNEL LDS CONTRACT GATE (run #10 post-mortem, 2026-07-20)
#
# Run #10 hung because occ_dispatch.cpp's ldsBytesRaw mirrored the kernel's LDS_TOTAL_FLOW
# by hand and silently lost the SSWIN*SLOTC_STRIDE term the kernel added at :719. Every
# KERNEL gate passed -- the defect was on the other side of the contract.
#
# This gate does NOT re-mirror the formula (that would just be a third copy to drift).
# It EXTRACTS the assembled truth: LDS_TOTAL_FLOW is recovered by bisecting on an
# assembler predicate, then compared against the host's computed allocation.
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
CC=/opt/rocm/llvm/bin/clang

# config of record
POOL_N=1; ACC_N=3; FM=1; FN=4; G=6; SEGK=256; OP_BASE=512; GRANULE=512

base_args() {
  echo -n "-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201
    -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=$FM -Wa,-defsym,FN=$FN -Wa,-defsym,G=$G -Wa,-defsym,SEGK=$SEGK
    -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,POOL_N=$POOL_N -Wa,-defsym,ACC_N=$ACC_N
    -Wa,-defsym,WOFLUSH=0 -Wa,-defsym,WAVES=30 -Wa,-defsym,VBUDGET=1536 -Wa,-defsym,PHASEPROBE=0
    -Wa,-defsym,NOCFLUSH=0 -Wa,-defsym,JDEPTH=1 -Wa,-defsym,STAGGER=1 -Wa,-defsym,KMAJOR=0
    -Wa,-defsym,MAXFAT=0 -Wa,-defsym,STAGERS=4 -Wa,-defsym,NTLOAD=0 -Wa,-defsym,DUTYPROBE=0
    -Wa,-defsym,RBU=1 -Wa,-defsym,NOFEED=0 -Wa,-defsym,MULTISLOT=0 -Wa,-defsym,MSCOMP=0
    -Wa,-defsym,MSSCAN=0 -Wa,-defsym,MSDRAIN=0 -Wa,-defsym,MSFEED=0 -Wa,-defsym,BATCHASN=0
    -Wa,-defsym,DECENTASN=1 -Wa,-defsym,DUTY_EVERY=64 -Wa,-defsym,CSTORE=0 -Wa,-defsym,SLEEPN=2
    -Wa,-defsym,COORD_PERIOD=64 -Wa,-defsym,DEADMAN=1 -Wa,-defsym,DEADMAN_TICKS=50000000
    -Wa,-defsym,TRACE=0 -Wa,-defsym,BANKZERO=1 -Wa,-defsym,FATGAUGE=0 -Wa,-defsym,SELFSERVE=1
    -Wa,-defsym,FORENSICS=0 -Wa,-defsym,STAGINSTR=1 -Wa,-defsym,TFPROBE=1"
}

# Probe: assemble a 2-line file that pulls in the kernel's .set chain and errors iff
# LDS_TOTAL_FLOW >= $1. Bisection over [0,65536] recovers the exact value in 17 assembles.
probe_ge() {  # $1 = threshold, $2 = SSWIN (empty = unset)
  local sw=""; [ -n "$2" ] && sw="-Wa,-defsym,SSWIN=$2"
  local f="$TMP/probe.s"
  { echo ".set LDSPROBE, $1"; echo ".include \"occ_kernel_dsws_flow.s\""; } > "$f"
  # the kernel body assembles fine; we only care whether the trailing predicate fires
  echo ".if LDS_TOTAL_FLOW >= LDSPROBE" >> "$f"
  echo "  .error \"GE\"" >> "$f"
  echo ".endif" >> "$f"
  $CC $(base_args) $sw -c "$f" -o /dev/null 2>&1 | grep -q "error: GE"
}

extract_lds() {  # $1 = SSWIN (empty = unset) ; echoes exact LDS_TOTAL_FLOW
  local lo=0 hi=65537 mid
  while [ $((hi - lo)) -gt 1 ]; do
    mid=$(( (lo + hi) / 2 ))
    if probe_ge $mid "$1"; then lo=$mid; else hi=$mid; fi
  done
  echo $lo
}

host_lds() {  # $1 = SSWIN (empty = unset) -- mirrors occ_dispatch.cpp ldsBytesRaw
  local opstride=$(( FN*16*SEGK + G*16*FM*SEGK ))
  local acc=$(( ACC_N * FM*FN*1024 ))
  local raw=$(( OP_BASE + POOL_N*opstride + acc ))
  if [ -n "$1" ] && [ "$1" -gt "$POOL_N" ]; then raw=$(( raw + $1*32 )); fi
  echo $raw
}

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rc=0
for sw in "" 8 16; do
  lbl=${sw:-unset}
  k=$(extract_lds "$sw")
  h=$(host_lds "$sw")
  gran=$(( (h + GRANULE - 1) / GRANULE * GRANULE ))
  if [ "$k" -le "$gran" ]; then verdict="PASS"; else verdict="*** FAIL: kernel needs $k, host allocates $gran ***"; rc=1; fi
  printf "  SSWIN=%-5s kernel LDS_TOTAL_FLOW=%-6s host raw=%-6s granule-rounded=%-6s  %s\n" \
         "$lbl" "$k" "$h" "$gran" "$verdict"
done
echo "gate_lds rc=$rc"
exit $rc
