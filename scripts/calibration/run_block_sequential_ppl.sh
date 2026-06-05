#!/bin/bash
# Task 8 (MAD-264): block-sequential GPTQ PPL acceptance on Qwen3.5-0.8B.
# Apples-to-apples with the offset experiment (same corpus/seq/seed/device), so
# results compare directly to offset_exp/results.tsv (static + per-target).
# Matrix: 80k x {pd0.01,0.05,0.10} (percdamp characterization) + 256k x pd0.05 (clean).
# CRITICAL: export ML8_TIER_OVERRIDE so the CONVERTER quantizes token_embd too (498MB).
set -u
cd /home/kmbandy/GitHub/llama.cpp
export PYTHONPATH=gguf-py
export ML8_DETERMINISTIC=1
export ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8"

EXP=/home/kmbandy/models/phase2/blockseq
mkdir -p "$EXP"
RES=$EXP/results.tsv
MODEL=/home/kmbandy/models/Qwen3.5-0.8B-hf
BASE_GGUF=/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf
HELDOUT=/home/kmbandy/models/hessian-sweep/quant_so_eval.txt

printf "config\tmode\twiki_ppl\theldout_ppl\tsize_mb\tcalib_s\n" > "$RES"
ppl_of(){ grep -oE "Final estimate: PPL = [0-9.]+" "$1" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1; }

run_one(){
  local pd=$1 budget=$2
  local tag="bs_pd${pd}_b${budget}"
  local cdir=$EXP/$tag gguf=$EXP/$tag.gguf clog=$EXP/$tag.calib.log
  echo "########## [$(date +%H:%M:%S)] START $tag (block-sequential pd=$pd budget=$budget) ##########"
  local t0=$(date +%s)
  python3 scripts/calibration/calibrate_ml8_paged.py \
    --model "$MODEL" --gguf "$BASE_GGUF" --arch qwen35 --device cuda:0 --strategy dense \
    --output-dir "$cdir" --rotation kronecker --group-size 64 --n-centroids 16 \
    --percdamp "$pd" --fit-loss mse --dense-coverage full --faithful-acts --faithful-weights \
    --awq none --corpus mix --seq-len 2048 --corpus-seed 0 --token-budget "$budget" --no-resume \
    --hessian-mode block-sequential --phase-timing 2>&1 | tee "$clog"
  local calib_s=$(( $(date +%s) - t0 ))
  python3 scripts/calibration/ml8_to_gguf.py --base-gguf "$BASE_GGUF" --calib-dir "$cdir" \
    --out-gguf "$gguf" --allow-partial > "$EXP/$tag.convert.log" 2>&1
  local sz=$(stat -c%s "$gguf" 2>/dev/null); sz=$(( ${sz:-0} / 1048576 ))
  build-hip/bin/llama-perplexity --no-mmap -m "$gguf" -ngl 99 --device ROCm0 \
    -f wikitext-2-raw/wiki.test.raw -c 512 > "$EXP/$tag.wiki.log" 2>&1
  build-hip/bin/llama-perplexity --no-mmap -m "$gguf" -ngl 99 --device ROCm0 \
    -f "$HELDOUT" -c 512 > "$EXP/$tag.heldout.log" 2>&1
  local w=$(ppl_of "$EXP/$tag.wiki.log"); local h=$(ppl_of "$EXP/$tag.heldout.log")
  printf "pd%s\tblock-seq-%s\t%s\t%s\t%s\t%s\n" "$pd" "$budget" "${w:-NA}" "${h:-NA}" "$sz" "$calib_s" >> "$RES"
  echo "########## [$(date +%H:%M:%S)] DONE $tag  wiki=${w:-NA} heldout=${h:-NA} size=${sz}MB calib=${calib_s}s ##########"
  rm -f "$gguf"   # PPL extracted; free 498MB
}

# 80k percdamp characterization (compare to offset_exp static+per-target at 80k)
for pd in 0.01 0.05 0.10; do run_one "$pd" 80000; done
# 256k clean number at the middle damping
run_one 0.05 256000

echo "########## [$(date +%H:%M:%S)] BLOCK-SEQ ACCEPTANCE COMPLETE ##########"
column -t "$RES"
