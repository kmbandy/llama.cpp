#!/usr/bin/env bash
# build.sh  (dvgpr_occ T5) -- RAM-capped build of the dyn-VGPR occupancy A/B harness.
#
# Produces:
#   occ_dyn.bin / occ_static.bin  - the two gfx1201 kernel variants (via -defsym DYNVGPR)
#   test_oracle                   - CPU fp8 e4m3 oracle self-test (built + run here)
#   occ_dispatch                  - the KFD PM4 A/B dispatch harness (links libhsakmt.a)
#
# RAM safety: the harness compile is wrapped in a systemd-run --user scope capped
# at 4G (host has ~15G; never run an uncapped build here).
set -euo pipefail
cd "$(dirname "$0")"

ROCM=/opt/rocm
L="$ROCM/llvm/bin"
PM4=../dvgpr_pm4
MEMMAX="${MEMMAX:-4G}"

run_capped() {
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 "$@"
    else
        echo "WARN: systemd-run unavailable; running uncapped" >&2
        "$@"
    fi
}

echo "[1/3] assembling occ_kernel.s -> occ_dyn.bin + occ_static.bin (gfx1201)"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DYNVGPR=1 -c occ_kernel.s -o occ_dyn.o
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DYNVGPR=0 -c occ_kernel.s -o occ_static.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_dyn.o    occ_dyn.bin
"$L/llvm-objcopy" -O binary --only-section=.text occ_static.o occ_static.bin
echo "      occ_dyn.bin: $(wc -c < occ_dyn.bin) bytes   occ_static.bin: $(wc -c < occ_static.bin) bytes"

echo "[2/3] oracle self-test"
clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle
./test_oracle

echo "[3/3] building occ_dispatch (MemoryMax=$MEMMAX)"
run_capped clang++ -std=c++17 -O2 -Wall -Wno-unused \
    -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
    occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
    "$ROCM/lib/libhsakmt.a" \
    -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt \
    -o occ_dispatch

echo "OK -> ./occ_dispatch [nWG]   (SUPERVISED: raw PM4 on the gfx12 node)"
echo "      smoke : timeout 30 ./occ_dispatch 64"
echo "      A/B   : timeout 30 ./occ_dispatch 2048"
