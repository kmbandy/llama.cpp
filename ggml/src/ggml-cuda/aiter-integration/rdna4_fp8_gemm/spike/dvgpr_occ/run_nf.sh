#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
ROCM=/opt/rocm; PM4=../dvgpr_pm4
clang++ -std=c++17 -O2 -Wall -Wno-unused -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
    occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
    "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch 2>cc.err || { echo "COMPILE FAIL"; grep -iE 'error' cc.err | head; exit 1; }
echo "compiled OK"
for spec in "2048 9216 2560 down-train" "2048 2560 9216 gateup-train"; do
  set -- $spec
  echo "==== $4 : M=$1 K=$2 N=$3 (champion 42_tw4 best ~112/105) ===="
  WG_M=$1 WG_K=$2 WG_N=$3 timeout 150 ./occ_dispatch --sp82 2>&1 \
    | grep -E "(42_tw4|42_NOFEED|82_NOFEED|42_FEEDONLY) .* 64 " | grep -v "wggemm2] 512"
done
