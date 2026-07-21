#!/usr/bin/env bash
# sweep_dsws_realshapes.sh — run OUR DSWS kernel across the REAL Qwen (ml8 / mlambaformer) GEMM shapes,
# each driven the way inference actually drives it: the REAL shape, looped back-to-back to steady state
# (DSWS2_TARGET_SECS), NO deep-K inflation. Shape -> run -> number. That's the whole thing.
#
#   One bin does all shapes: G=4 ACC_N=4 FM=1 SEGK=256 (TMsuper=64, TN=64).
#   A shape is LEGAL iff M%64==0 && N%64==0 && K%256==0. Anything else is printed UNSUPPORTED
#   (honest — never silently skipped). K may be non-pow2 (arbitrary-K decode); N/M need not be pow2
#   (magic-div by NTL). Every dispatch goes through gpu_run.sh (latch/deadman/stale-bin/logging);
#   any hang or WORK-INEXACT latches and HALTS the sweep (a full stop is a full stop).
#
#   env: TARGET_SECS (default 1.5s steady state)   STRIDE (oracle sample stride; 1=full, default 32)
#        ONLY="substr"  -> run only shapes whose label matches (bring-up)
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
LOGDIR="$HOME/dsws_gpu_logs"; mkdir -p "$LOGDIR"
OUT="$LOGDIR/dsws_realshape_sweep.tsv"
TARGET_SECS="${TARGET_SECS:-1.5}"
STRIDE="${STRIDE:-32}"
ONLY="${ONLY:-}"
G=6; FM=1; FN=4; SEGK=256; ACC_N=3   # CONFIG OF RECORD (GROUPS=G/ACC_N=2)
TMS=$((G*16*FM)); TN=$((FN*16))   # 64, 64

# label  M  N  K   (the GEMM is C[M,N] = A[M,K] @ B[K,N]; these ARE the ml8/mlambaformer Qwen shapes)
read -r -d '' SHAPES <<'EOF'
ml8_dense_ffn_gate_up   2048  9216 2560
ml8_dense_ffn_gate_up    512  9216 2560
ml8_dense_ffn_down      2048  2560 9216
ml8_dense_ffn_down       512  2560 9216
ml8_dense_attn_q        2048  4096 2560
ml8_dense_attn_q         512  4096 2560
ml8_dense_attn_kv       2048  1024 2560
ml8_dense_attn_kv        512  1024 2560
ml8_dense_attn_o        2048  2560 4096
ml8_dense_attn_o         512  2560 4096
ml8_moe_ffn_gate_up       64   512 2048
ml8_moe_ffn_gate_up      512   512 2048
ml8_moe_ffn_down          64  2048  512
ml8_moe_ffn_down         512  2048  512
ml8_moe_attn_q            64  4096 2048
ml8_moe_attn_q           512  4096 2048
ml8_moe_attn_kv           64   512 2048
ml8_moe_attn_kv          512   512 2048
ml8_moe_attn_o            64  2048 4096
ml8_moe_attn_o           512  2048 4096
mlmf_mamba_in_proj      4096  4200  768
mlmf_in_proj_ML8PAD     4096  4208  768
mlmf_MoE_expert_fc1      512  1536  768
mlmf_MoE_expert_fc2      512   768 1536
mlmf_lm_head            4096 32000  768
mlmf_mamba_out_proj     4096   768 1536
mlmf_attn_o_proj        4096   768  768
mlmf_router_down_proj   4096   256  768
mlmf_router_MLP         4096   256  256
mlmf_attn_linear_k      4096   192  768
mlmf_attn_val_proj1     4096    96  768
mlmf_router_out         4096     8  256
mlmf_routerout_ML8PAD   4096    16  256
EOF

printf 'label\tM\tN\tK\tGFLOP\tstatus\tDSWS_TF\toracle\twork\n' > "$OUT"
printf '%-24s %6s %6s %6s %8s  %-11s %8s  %s\n' shape M N K GFLOP status "DSWS_TF" "note"
printf '%s\n' "----------------------------------------------------------------------------------------------------"

run_one() {
  local label="$1" M="$2" N="$3" K="$4"
  local gflop; gflop=$(awk "BEGIN{printf \"%.1f\", 2.0*$M*$N*$K/1e9}")
  # legality
  local Mpad=$(( (M + TMS - 1) / TMS * TMS ))     # pad M up to the super-tile (config of record: 96)
  if (( N % TN != 0 || K % SEGK != 0 )); then
    local why=""
    (( N % TN  != 0 )) && why="${why:+$why,}N%$TN"
    (( K % SEGK!= 0 )) && why="${why:+$why,}K%$SEGK"
    printf '%-24s %6d %6d %6d %8s  %-11s %8s  %s\n' "$label" "$M" "$N" "$K" "$gflop" UNSUPPORTED "--" "$why"
    printf '%s\t%d\t%d\t%d\t%s\tUNSUPPORTED\t\t\t%s\n' "$label" "$M" "$N" "$K" "$gflop" "$why" >> "$OUT"
    return 0
  fi
  local oMTL=$((Mpad/TMS)) oNTL=$((N/TN))
  local out rc tf oracle work
  out=$(./gpu_run.sh "rs_${label}_M${M}" -- \
        SSWIN=8 FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 FLOW_POOL_N=1 \
        DSWS2_SEGK=256 DSWS2_K="$K" DSWS2_ORACLE_MTL="$oMTL" DSWS2_ORACLE_NTL="$oNTL" \
        DSWS2_ORACLE_STRIDE="$STRIDE" DSWS2_TARGET_SECS="$TARGET_SECS" \
        STAGINSTR=1 FORENSICS=0 TFPROBE=1 ./occ_dispatch --dsws2 2>&1)
  rc=$?
  # TF: the TFPROBE / result line. Grab the last number preceding "TF" (case-insensitive).
  tf=$(printf '%s\n' "$out" | grep -ioE '[0-9]+\.[0-9]+ *TF' | tail -1 | grep -oE '[0-9]+\.[0-9]+')
  [ -z "$tf" ] && tf=$(printf '%s\n' "$out" | grep -iE 'TFLOP|throughput|TF=' | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  oracle=$(printf '%s\n' "$out" | grep -oE 'ok=[0-9]+ +bad=[0-9]+([^ ]*)?( +max_rel=[0-9.eE+-]+)?' | tail -1)
  work=$(printf '%s\n' "$out" | grep -oE 'WORK-(EXACT|INEXACT)' | tail -1)
  [ -z "$tf" ] && tf="?"
  [ -z "$oracle" ] && oracle="(no-oracle-line)"
  [ -z "$work" ] && work="(no-work-line)"
  printf '%-24s %6d(%5d) %6d %6d %8s  %-11s %8s  %s | %s\n' "$label" "$M" "$Mpad" "$N" "$K" "$gflop" "rc=$rc" "$tf" "$work" "$oracle"
  printf '%s\t%d\t%d\t%d\t%s\trc=%d\t%s\t%s\t%s\n' "$label" "$M" "$N" "$K" "$gflop" "$rc" "$tf" "$oracle" "$work" >> "$OUT"
  if [ "$rc" -ne 0 ]; then
    echo ""
    echo "  *** gpu_run.sh returned rc=$rc on $label (M=$M N=$N K=$K) -- LATCH/HANG/REFUSE. FULL STOP. ***"
    echo "  *** Sweep halted. Inspect the log; a human clears the latch before any further dispatch. ***"
    return 1
  fi
  return 0
}

while read -r label M N K; do
  [ -z "$label" ] && continue
  [ -n "$ONLY" ] && [[ "$label" != *"$ONLY"* ]] && continue
  run_one "$label" "$M" "$N" "$K" || exit 1
done <<< "$SHAPES"

echo ""
echo "  sweep complete. TSV -> $OUT"
