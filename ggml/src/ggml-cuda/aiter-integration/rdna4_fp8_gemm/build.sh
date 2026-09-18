#!/usr/bin/env bash
# RAM-capped HIP build for the RDNA4 fp8 GEMM (15 GB host — never uncapped).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-gfx1201}"
MEM_MAX="${MEM_MAX:-6G}"
ROCM_INC="${ROCM_INC:-/opt/rocm/include}"
cap() { systemd-run --user --scope -p MemoryMax="$MEM_MAX" -p MemoryHigh=5G "$@"; }

echo "== free RAM =="; free -h | awk 'NR<=2'
avail="$(free -m | awk '/^Mem:/{print $7}')"
[ "${avail:-0}" -ge 4000 ] || { echo "ABORT: <4GB available"; exit 1; }

mkdir -p "$HERE/out"
echo "== build ceiling microbench =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" "$HERE/bench/wmma_peak.hip" -o "$HERE/out/wmma_peak"

echo "== build global_load_tr_probe =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" \
  "$HERE/bench/global_load_tr_probe.hip" -o "$HERE/out/global_load_tr_probe"

echo "== build gemm_trfeed_bench =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" \
  "$HERE/bench/gemm_trfeed_bench.hip" -o "$HERE/out/gemm_trfeed_bench"

if [ -f "$HERE/gemm_wmma.hip" ]; then
  echo "== build librdna4_gemm.so =="
  cap hipcc --offload-arch="$ARCH" -O3 -fPIC --shared -I"$ROCM_INC" \
    "$HERE/gemm_wmma.hip" -o "$HERE/out/librdna4_gemm.so"
fi

if [ -f "$HERE/gemm_blockscale.hip" ]; then
  echo "== build gemm_ml8fp8_blockscale_bench (MAD-305 Phase 5, abandoned kernels — kept for A/B only) =="
  cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" \
    "$HERE/gemm_blockscale.hip" "$HERE/bench/gemm_ml8fp8_blockscale_bench.hip" \
    -o "$HERE/out/gemm_ml8fp8_blockscale_bench"
fi

if [ -f "$HERE/gemm_trfeed_prod.hip" ]; then
  echo "== build gemm_trfeed_prod_bench (MAD-305 Phase 5 round 3 — production entry point) =="
  # Links gemm_blockscale.hip only for its rdna4_[un]preshuffle_b_ml8fp8 byte-
  # shuffle helpers (reused unchanged by the frozen trfeed kernel's weight
  # packing); its two abandoned GEMM kernels are unused dead code in this bin.
  cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" -I"$HERE" \
    "$HERE/gemm_blockscale.hip" "$HERE/gemm_trfeed_prod.hip" \
    "$HERE/bench/gemm_trfeed_prod_bench.hip" \
    -o "$HERE/out/gemm_trfeed_prod_bench"
fi
echo "== DONE =="
