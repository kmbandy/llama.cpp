#!/bin/bash
# 256k mix head-to-head: static + block-seq across percdamp, queued AFTER the
# current blockseq_ppl.sh sweep (which already produced block-seq 256k pd0.05).
# Fills: block-seq {0.01,0.10} + static {0.01,0.05,0.10} → full 2x3 256k mix table.
set -u
cd /home/kmbandy/GitHub/llama.cpp
export PYTHONPATH=gguf-py
export ML8_DETERMINISTIC=1
export ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8"

EXP=/home/kmbandy/models/phase2/blockseq
RES=$EXP/results_256k.tsv
MODEL=/home/kmbandy/models/Qwen3.5-0.8B-hf
BASE_GGUF=/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf
HELDOUT=/home/kmbandy/models/hessian-sweep/quant_so_eval.txt
BUDGET=256000

# --- wait for the current sweep to finish (avoid GPU contention) ---
echo "[queue $(date +%H:%M:%S)] waiting for blockseq_ppl.sh to finish..."
while pgrep -f "blockseq_ppl.sh" >/dev/null 2>&1; do sleep 30; done
echo "[queue $(date +%H:%M:%S)] current sweep done — starting 256k mix matrix"

printf "config\tmode\twiki_ppl\theldout_ppl\tsize_mb\tcalib_s\n" > "$RES"
ppl_of(){ grep -oE "Final estimate: PPL = [0-9.]+" "$1" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1; }

run_one(){
  local mode=$1 pd=$2
  local tag="${mode}_pd${pd}_b${BUDGET}"
  local cdir=$EXP/$tag gguf=$EXP/$tag.gguf clog=$EXP/$tag.calib.log
  echo "########## [$(date +%H:%M:%S)] START $tag (mode=$mode pd=$pd budget=$BUDGET) ##########"
  local t0=$(date +%s)
  python3 scripts/calibration/calibrate_ml8_paged.py \
    --model "$MODEL" --gguf "$BASE_GGUF" --arch qwen35 --device cuda:0 --strategy dense \
    --output-dir "$cdir" --rotation kronecker --group-size 64 --n-centroids 16 \
    --percdamp "$pd" --fit-loss mse --dense-coverage full --faithful-acts --faithful-weights \
    --awq none --corpus mix --seq-len 2048 --corpus-seed 0 --token-budget "$BUDGET" --no-resume \
    --hessian-mode "$mode" --phase-timing 2>&1 | tee "$clog"
  local calib_s=$(( $(date +%s) - t0 ))
  python3 scripts/calibration/ml8_to_gguf.py --base-gguf "$BASE_GGUF" --calib-dir "$cdir" \
    --out-gguf "$gguf" --allow-partial > "$EXP/$tag.convert.log" 2>&1
  local sz=$(stat -c%s "$gguf" 2>/dev/null); sz=$(( ${sz:-0} / 1048576 ))
  build-hip/bin/llama-perplexity --no-mmap -m "$gguf" -ngl 99 --device ROCm0 \
    -f wikitext-2-raw/wiki.test.raw -c 512 > "$EXP/$tag.wiki.log" 2>&1
  build-hip/bin/llama-perplexity --no-mmap -m "$gguf" -ngl 99 --device ROCm0 \
    -f "$HELDOUT" -c 512 > "$EXP/$tag.heldout.log" 2>&1
  local w=$(ppl_of "$EXP/$tag.wiki.log"); local h=$(ppl_of "$EXP/$tag.heldout.log")
  printf "pd%s\t%s\t%s\t%s\t%s\t%s\n" "$pd" "$mode" "${w:-NA}" "${h:-NA}" "$sz" "$calib_s" >> "$RES"
  echo "########## [$(date +%H:%M:%S)] DONE $tag  wiki=${w:-NA} heldout=${h:-NA} size=${sz}MB calib=${calib_s}s ##########"
  rm -f "$gguf"
}

# block-seq: fill pd0.01 (operating point / headline) + pd0.10 (pd0.05 already done in main sweep)
run_one block-sequential 0.01
run_one block-sequential 0.10
# static: full percdamp curve at 256k mix
run_one single 0.01
run_one single 0.05
run_one single 0.10

echo "########## [$(date +%H:%M:%S)] 256K MIX MATRIX COMPLETE ##########"
echo "=== block-seq 256k pd0.05 (from main sweep) ==="; grep "block-seq-256000" "$EXP/results.tsv" 2>/dev/null
echo "=== this matrix ==="; column -t "$RES"
