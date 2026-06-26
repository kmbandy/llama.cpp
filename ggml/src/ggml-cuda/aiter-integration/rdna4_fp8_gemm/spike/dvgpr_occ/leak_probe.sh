#!/usr/bin/env bash
# Settle: does a SILENT R9700 PM4 dispatch leak compute to the 6900XT?
# Samples both cards (gpu/mem busy) across idle -> silent-compute, and snapshots
# which KFD node gets a compute context mid-run.
set -u
cd "$(dirname "$0")"
LOG=/tmp/leak_probe.log; : > "$LOG"
C0=/sys/class/drm/card0/device   # 6900XT (PCI 0b)
C1=/sys/class/drm/card1/device   # R9700  (PCI 42)
rd(){ printf '%s/%s' "$(cat $1/gpu_busy_percent 2>/dev/null)" "$(cat $1/mem_busy_percent 2>/dev/null)"; }

{
echo "=== connector -> card (which GPU drives which monitor) ==="
for cc in /sys/class/drm/card*-*; do
  st=$(cat "$cc/status" 2>/dev/null)
  [ "$st" = "connected" ] && echo "  $(basename "$cc"): $st"
done
echo "=== legend: c0=6900XT gpu%/mem%  c1=R9700 gpu%/mem% ==="

echo "-- baseline idle (1s) --"
for i in $(seq 1 25); do echo "$(date +%s.%N) idle c0=$(rd $C0) c1=$(rd $C1)"; sleep 0.04; done

echo "-- launch SILENT occ_dispatch --decomp (output -> /dev/null) --"
./occ_dispatch --decomp >/dev/null 2>&1 &
DPID=$!
i=0
while kill -0 $DPID 2>/dev/null; do
  echo "$(date +%s.%N) run  c0=$(rd $C0) c1=$(rd $C1)"
  i=$((i+1))
  if [ $((i % 25)) -eq 8 ]; then
    echo "   --- KFD compute contexts @ sample $i ---"
    kp=$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null | tr '\n' ' '); echo "   KFD proc pids: ${kp:-none}"
    rocm-smi --showpids 2>/dev/null | grep -viE "===|ROCm System|End of|^$" | sed 's/^/   /'
  fi
  sleep 0.03
done
echo "-- done --"
} >> "$LOG" 2>&1
echo "WROTE $LOG"
