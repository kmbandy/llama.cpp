#!/usr/bin/env bash
# Build R4D into r4d.so, an importable pybind11 extension.
#
# Requires hipcc and the pybind11 headers; run it wherever those live -- normally inside the ROCm
# container the library will be loaded from, since the extension must be compiled with the same
# ROCm/hipcc it links against at runtime.
#
#     ./build.sh                      # r4d.so for gfx1201
#     GFX_ARCH=gfx1200 ./build.sh     # a different RDNA4 part
#     OUT=/tmp/r4d.so ./build.sh
#
# Every translation unit is compiled with -ffp-contract=off. The quantized all-reduce REQUIRES it:
# it sums two dequantized products per element, and otherwise the compiler contracts `a*sa + b*sb`
# into an FMA that absorbs one product with a single rounding. Because the "self" and "peer"
# products are swapped between the two tensor-parallel ranks, each rank fuses a different multiply
# and the ranks disagree by ~1 ULP on a few elements, breaking the replicated-state invariant. For
# the rest it is reproducibility: it is how they were built when they were measured and validated.
# Neither the skinny GEMM (an explicit v_dot2 intrinsic) nor the exact all-reduce (a plain sum) has
# a contraction site at all.
#
# The library is compiled to objects and linked rather than built as one big source include so that
# per-kernel target features stay scoped: the GDN kernel is built in CU mode (-mcumode), because it
# asks for 60 KB of LDS per workgroup and was tuned and validated with one workgroup per CU, and
# -mcumode is a whole-translation-unit flag.
set -euo pipefail
cd "$(dirname "$0")"

GFX_ARCH=${GFX_ARCH:-gfx1201}
PYTHON=${PYTHON:-python3}
HIPCC=${HIPCC:-hipcc}
OUT=${OUT:-r4d.so}
JOBS=${JOBS:-$(nproc)}

INC=$($PYTHON -m pybind11 --includes)
BASE="-O3 -std=c++17 -fPIC --offload-arch=${GFX_ARCH} -Wno-unused-result -ffp-contract=off"

# translation unit : extra flags
UNITS=(
  "r4d_attn_paged_h256_gqa6:"
  "r4d_attn_vit_h72_bf16:"
  "r4d_gdn_chunk_scan_k128_v128_c64_bf16:-mcumode"
  "r4d_gdn_conv_w4_h128_bf16:"
  "r4d_gdn_kkt_solve_k128_c64_bf16:"
  "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state:"
  "r4d_gdn_fused_update_w4k128v128:"
  "r4d_gdn_gated_rmsnorm_h128_bf16:"
  "r4d_ar_oneshot_2rank_exact:"
  "r4d_ar_oneshot_2rank_wht6:"
  "r4d_ar_oneshot_3rank_exact:"
  "r4d_gemm_bf16_nt_m16:"
  "r4d_registry:"
  "r4d_module:"
)

pids=()
for u in "${UNITS[@]}"; do
  name=${u%%:*}
  extra=${u#*:}
  echo "[hipcc] ${name}.o (${GFX_ARCH})${extra:+ ${extra}}"
  # shellcheck disable=SC2086
  $HIPCC $BASE $extra $INC -c "${name}.hip" -o "${name}.o" &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
for p in "${pids[@]}"; do wait "$p"; done

echo "[hipcc] ${OUT} (link)"
# hipcc emits a fatbin per object; no -fgpu-rdc is needed, because no device function is called
# across translation units.
$HIPCC -shared --offload-arch="${GFX_ARCH}" r4d_attn_paged_h256_gqa6.o r4d_attn_vit_h72_bf16.o r4d_gdn_chunk_scan_k128_v128_c64_bf16.o r4d_gdn_conv_w4_h128_bf16.o \
  r4d_gdn_kkt_solve_k128_c64_bf16.o r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.o r4d_gdn_fused_update_w4k128v128.o r4d_gdn_gated_rmsnorm_h128_bf16.o \
  r4d_ar_oneshot_2rank_exact.o r4d_ar_oneshot_2rank_wht6.o r4d_ar_oneshot_3rank_exact.o \
  r4d_gemm_bf16_nt_m16.o r4d_registry.o r4d_module.o -o "${OUT}"
echo "[build.sh] $(ls -la "${OUT}")"
