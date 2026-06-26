#!/usr/bin/env bash
# Stage-1 DRAM-saturation gauge: sample amdgpu memory-controller busy% while the
# proven-safe cooperative GEMM (--decomp) loops. Self-bounded; kills the GEMM loop.
# Output: /tmp/membw_sample.log  (epoch  mem_busy_percent  sclk_mhz)
set -u
cd "$(dirname "$0")"
MODE="${1:---decomp}"                 # which harness mode to loop (default --decomp)
LOG="${2:-/tmp/membw_sample.log}"
: > "$LOG"
MB=/sys/class/drm/card1/device/mem_busy_percent
GB=/sys/class/drm/card1/device/gpu_busy_percent

# Kernel loop, hard-capped at 28s so the GPU stays continuously pressured for sampling.
timeout 28 bash -c "while true; do ./occ_dispatch $MODE >/dev/null 2>&1; done" &
GEMM=$!

# Sample at ~20 Hz for ~28s.
for i in $(seq 1 560); do
  m=$(cat "$MB" 2>/dev/null)
  g=$(cat "$GB" 2>/dev/null)
  printf '%s %s %s\n' "$(date +%s.%N)" "${m:-NA}" "${g:-NA}" >> "$LOG"
  sleep 0.05
done

kill "$GEMM" 2>/dev/null
wait "$GEMM" 2>/dev/null
echo "DONE" >> "$LOG"
