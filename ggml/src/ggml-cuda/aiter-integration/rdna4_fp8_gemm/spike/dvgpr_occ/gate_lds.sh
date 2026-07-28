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

# config of record -- ENV-OVERRIDABLE (fixed 2026-07-26; see below)
#
# *** THIS GATE LIED ON 2026-07-26 AND REPORTED PASS WHILE MEASURING A DIFFERENT CONFIG. ***
# These were plain assignments, so `FM=2 ACC_N=2 ./gate_lds.sh` silently tested FM=1 ACC_N=3
# anyway -- the assignment clobbered the env. I read its PASS as evidence the FM=2 LDS number
# was sound; it was evidence of nothing. Now `: ${VAR:=default}` so an explicit env wins,
# matching build_flow.sh's idiom. A gate whose inputs you cannot set is a gate for one config.
: ${POOL_N:=1}; : ${ACC_N:=3}; : ${FM:=1}; : ${FN:=4}; : ${G:=6}; : ${SEGK:=256}
: ${SELFSERVE:=1}
OP_BASE=512; GRANULE=512
echo "  gate_lds config: FM=$FM FN=$FN G=$G ACC_N=$ACC_N POOL_N=$POOL_N SEGK=$SEGK SELFSERVE=$SELFSERVE"

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
  # OPERAND POOL RECLAIM (fixed 2026-07-26, mirrors the occ_dispatch.cpp fix of the same day).
  #   Kernel: `.if SELFSERVE || DSWS2_OVERLAP` -> ACC_BASE = OP_BASE, i.e. POOL_N*opstride is NOT reserved.
  #   The host tested only DSWS2_OVERLAP and this mirror reproduced that same half-condition, so the gate
  #   faithfully re-implemented the bug and therefore could never catch it. A mirror of a formula only
  #   validates the copy, never the original -- which is exactly why this gate is written to EXTRACT the
  #   assembled truth by bisection and compare, rather than trust either copy.
  local raw
  if [ "$SELFSERVE" = "1" ] || [ "${DSWS2_OVERLAP:-1}" = "1" ]; then
    raw=$(( OP_BASE + acc ))
  else
    raw=$(( OP_BASE + POOL_N*opstride + acc ))
  fi
  # Under SELFSERVE the kernel adds SSWIN*SLOTC_STRIDE UNCONDITIONALLY (no `> POOL_N` guard on that arm),
  # so the host must too. The old `> POOL_N` condition under-allocated by SSWIN*32 for SSWIN <= POOL_N.
  local sw=${1:-$POOL_N}
  if [ "$SELFSERVE" = "1" ] || [ "$sw" -gt "$POOL_N" ]; then raw=$(( raw + sw*32 )); fi
  echo $raw
}

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rc=0
for sw in "" 8 16 32; do            # 32 ADDED 2026-07-26: SSWIN=32 IS the config of record and
  lbl=${sw:-unset}                 #   this gate had never once tested it.
  k=$(extract_lds "$sw")
  h=$(host_lds "$sw")
  gran=$(( (h + GRANULE - 1) / GRANULE * GRANULE ))
  if [ "$k" -gt "$gran" ]; then
    verdict="*** FAIL: kernel needs $k, host allocates $gran (UNDER-ALLOCATION -> OOB LDS -> GPU RESET) ***"; rc=1
  elif [ "$SELFSERVE" = "1" ] && [ "$gran" -ge $(( k * 2 )) ]; then
    # OVER-allocation was INVISIBLE to this gate until 2026-07-26: the check was `k <= gran`,
    # which a 4x over-allocation passes happily. The host's ldsBytesRaw includes POOL_N*opstride,
    # but under SELFSERVE the kernel RECLAIMS the operand pool (ACC_BASE = OP_BASE, see
    # occ_kernel_dsws_flow.s "operand pool reclaimed (SELFSERVE owns this)"). So the host
    # over-counts by exactly one operand pool and its number is garbage.
    # WHY IT MATTERS EVEN THOUGH THE HOST "TRUSTS THE BIN": this bogus number is what prints
    # "host reconstruction says N but the BIN PUBLISHES M" on EVERY SINGLE DISPATCH. A warning
    # that always fires is a warning nobody reads -- and on 2026-07-26 I read past the one time
    # it was telling the truth, which is how a stale .lds sidecar reset the GPU.
    verdict="*** HOST OVER-COUNTS ${gran}B vs kernel ${k}B (~$(( gran / (k?k:1) ))x) -- ldsBytesRaw counts POOL_N*opstride that SELFSERVE reclaims ***"; rc=1
  else
    verdict="PASS"
  fi
  printf "  SSWIN=%-5s kernel LDS_TOTAL_FLOW=%-6s host raw=%-6s granule-rounded=%-6s  %s\n" \
         "$lbl" "$k" "$h" "$gran" "$verdict"
done
echo "gate_lds rc=$rc"
exit $rc
