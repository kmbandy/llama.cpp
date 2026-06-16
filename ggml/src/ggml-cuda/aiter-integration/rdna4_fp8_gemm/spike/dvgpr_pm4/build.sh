#!/usr/bin/env bash
# build.sh  (MAD-304 T3) -- RAM-capped build of the raw-PM4 dyn-VGPR harness.
#
# Produces:
#   probe.bin       - 32-byte gfx1201 raw ISA (reassembled from probe.s)
#   pm4_dispatch    - the KFD PM4 dispatch harness (links static libhsakmt.a)
#
# RAM safety: the compile is wrapped in a systemd-run --user scope capped at 4G
# (host has ~15G; never run an uncapped build here).
set -euo pipefail
cd "$(dirname "$0")"

ROCM=/opt/rocm
LLVM="$ROCM/llvm/bin"
MEMMAX="${MEMMAX:-4G}"

run_capped() {
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 "$@"
    else
        echo "WARN: systemd-run unavailable; running uncapped" >&2
        "$@"
    fi
}

echo "[1/2] assembling probe.s -> probe.bin (gfx1201)"
"$LLVM/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c probe.s -o probe.o
"$LLVM/llvm-objcopy" -O binary --only-section=.text probe.o probe.bin
echo "      probe.bin: $(wc -c < probe.bin) bytes"

echo "[2/2] compiling pm4_dispatch (MemoryMax=$MEMMAX)"
run_capped clang++ -std=c++17 -O2 -Wall -Wno-unused \
    -I vendor/compat -I vendor -I "$ROCM/include" \
    pm4_dispatch.cpp vendor/PM4Packet.cpp vendor/BasePacket.cpp \
    "$ROCM/lib/libhsakmt.a" \
    -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt \
    -o pm4_dispatch

echo "OK -> ./pm4_dispatch   (run: sudo ./pm4_dispatch            # baseline, must read 0)"
echo "                       (then sudo ./pm4_dispatch --dynvgpr  # the lift)"
