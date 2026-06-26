#!/usr/bin/env bash
# NAIVE-FEED (exposed) + GENDIV mbgemm bins for the real ml8 shapes -> _ndgd suffix. Same-tile pairs only.
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
mk() { local dv=$1 fm=$2 fn=$3 out=$4
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=1 \
     -Wa,-defsym,GENDIV=1 -Wa,-defsym,NAIVEFEED=1 \
     -c occ_kernel_mbgemm.s -o "$out.o"
  "$L/llvm-objcopy" -O binary --only-section=.text "$out.o" "$out.bin"
  echo "  $out.bin ($(wc -c < "$out.bin")B)"; }
mk 0 2 2 occ_mbgemm_2x2_b1_d0_ndgd
mk 1 2 2 occ_mbgemm_2x2_b1_d1_ndgd
mk 0 2 4 occ_mbgemm_2x4_b1_d0_ndgd
mk 1 2 4 occ_mbgemm_2x4_b1_d1_ndgd
echo "--- recompile harness ---"
ROCM=/opt/rocm; PM4=../dvgpr_pm4
clang++ -std=c++17 -O2 -Wall -Wno-unused -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
    occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
    "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch 2>cc.err \
    && echo "harness OK ($(wc -c < occ_dispatch)B)" || { echo "COMPILE FAIL"; grep -iE 'error' cc.err | head; exit 1; }
