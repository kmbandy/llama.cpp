#!/usr/bin/env bash
# build.sh  (dvgpr_occ P3) -- RAM-capped build of the dyn-VGPR GEMM-occupancy de-risk harness.
#
# Produces:
#   occ_n{8,16}_d{0,1}.bin  - the gfx1201 throughput-kernel matrix (NACC x DYNVGPR via -defsym)
#   test_oracle             - CPU fp8 e4m3 oracle self-test (built + run here)
#   occ_dispatch            - the KFD PM4 throughput/de-risk harness (links libhsakmt.a)
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

echo "[1/3] assembling occ_kernel.s -> occ_n{8,16}_d{0,1}.bin (gfx1201)"
for nacc in 8 16; do for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=$nacc -c occ_kernel.s -o occ_n${nacc}_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_n${nacc}_d${dv}.o occ_n${nacc}_d${dv}.bin
    echo "      occ_n${nacc}_d${dv}.bin: $(wc -c < occ_n${nacc}_d${dv}.bin) bytes"
done; done

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

echo "OK -> ./occ_dispatch [--prong1|--prong2]   (SUPERVISED: raw PM4 on the gfx12 node)"
echo "      prong1 (occupancy->throughput curve)     : timeout 40 ./occ_dispatch --prong1"
echo "      prong2 (dyn vs static heavy, KDEPTH sweep): timeout 60 ./occ_dispatch --prong2"
echo "      correctness A/B (KDEPTH=1)               : timeout 30 ./occ_dispatch"
