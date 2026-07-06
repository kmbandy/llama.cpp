#!/usr/bin/env bash
# Capped Vulkan / optional CUDA build for SP1+SP2 (turbo4_0 Vulkan FA / paged-attn oracle).
#
# WHY: this host has 15GB RAM / 8 cores. An uncapped `cmake --build -j` of the
# CUDA backend OOM-killed the box on 2026-06-28. This wrapper makes that
# impossible:
#   * CUDA is OFF by default -> nvcc/cudafe++ (the RAM hogs) never run.
#   * WITH_CUDA=1 enables CUDA (sm_61 / GTX1070) with a conservative -j2.
#   * the build runs inside a transient systemd --user scope with a hard
#     MemoryMax -> any OOM is contained to the build cgroup, never the system.
#   * CPUQuota=700% -> leaves cores/RAM for the live services.
#
# Usage:  build-vk.sh [target]             (default target: test-backend-ops)
#         WITH_CUDA=1 build-vk.sh [target] (dual-backend build, slow but capped)
set -euo pipefail

REPO=/home/kmbandy/GitHub/llama.cpp
BUILD="$REPO/build-vk"
TARGET="${1:-test-backend-ops}"

# Adaptive cgroup limits, sized from actual free RAM (2026-06-29 rewrite).
# LESSON: the old -j1 + MemoryHigh=5G was the *cause* of "painfully slow", not a
# safety net — the big ggml-vulkan.cpp TU peaks ~5G and a 5G MemoryHigh throttled it
# into a full swap mid-compile (thrash). So: floor the ceiling at 6G so that TU runs
# un-throttled, give the cgroup most of free RAM (keep a ~2.5G host buffer), and scale
# -j to the ceiling so the ~300 shader-wrapper TUs build in parallel. MemoryMax stays
# a HARD host-protective cap — overflow kills the build cgroup, never the host.
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)   # MiB
load=$(cut -d' ' -f1 /proc/loadavg)
if [ "$avail" -lt 3000 ]; then
  echo "[build-vk] ABORT: <3000MiB available — free memory before building." >&2
  exit 1
fi
cap_mib=$(( avail - 2560 ))                                       # keep host buffer
[ "$cap_mib" -lt 6144 ]  && cap_mib=6144                          # big TU needs ~5G
[ "$cap_mib" -gt 11264 ] && cap_mib=11264                         # cap at 11G
high_mib=$(( cap_mib - 1536 ))
jobs=$(( cap_mib / 2048 ))                                        # ~1 job / 2G ceiling
[ "$jobs" -lt 1 ] && jobs=1
maxj=$(( $(nproc) - 1 )); [ "$jobs" -gt "$maxj" ] && jobs=$maxj

CUDA_FLAGS="-DGGML_CUDA=OFF"
if [ "${WITH_CUDA:-0}" = "1" ]; then
  CUDA_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61"   # GTX1070 = sm_61
fi
# CUDA TUs are RAM-heavy: force conservative parallelism regardless of cap_mib
[ "${WITH_CUDA:-0}" = "1" ] && jobs=2

CAP=(systemd-run --user --scope --quiet
     -p MemoryMax=${cap_mib}M -p MemoryHigh=${high_mib}M -p CPUQuota=700%)
echo "[build-vk] pre-flight: avail=${avail}MiB load=${load} -> MemoryMax=${cap_mib}M MemoryHigh=${high_mib}M -j${jobs} target=${TARGET}"

# Wipe a stale CMakeCache when switching from CUDA OFF → ON (reconfigure needed).
if [ "${WITH_CUDA:-0}" = "1" ] && [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "GGML_CUDA:BOOL=ON" "$BUILD/CMakeCache.txt"; then
  echo "[build-vk] reconfiguring for CUDA (wiping stale CMakeCache)…"; rm -f "$BUILD/CMakeCache.txt"
fi

# Configure once: Vulkan ON, CUDA per WITH_CUDA flag.
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  echo "[build-vk] configuring (${CUDA_FLAGS}, Vulkan ON, Ninja)…"
  "${CAP[@]}" cmake -S "$REPO" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON ${CUDA_FLAGS} -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
fi

echo "[build-vk] building target '${TARGET}' (capped: MemoryMax=${cap_mib}M, -j${jobs})…"
"${CAP[@]}" cmake --build "$BUILD" -j${jobs} --target "$TARGET"
echo "[build-vk] done: $BUILD/bin/${TARGET}"
