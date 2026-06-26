#!/bin/bash
# MAD-281 4B Phase E pipeline chain: wait for Stage 1 (calibration) → Stage 2
# (convert to ml8 GGUF) → Stage 3 (frozen/gptq/gptq-interleave smoke). Each stage
# is success-gated; the chain aborts (and says why) if a stage didn't produce its
# output. Stage 3 runs under the RAM-safe scope. Ignore the calibrator's --eval-ppl
# numbers (untied-lm_head artifact); the real signal is the smoke's holdout KL.
set -uo pipefail
cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration

CALIB_DIR=/home/kmbandy/models/4b-ml8-phaseE-calib
BASE=/home/kmbandy/models/Qwen3.5-4B-bf16.gguf
MODEL=/home/kmbandy/models/Qwen3.5-4B-hf
OUT_GGUF=/home/kmbandy/models/Qwen3.5-4B-ml8.gguf

echo "[chain] $(date +%H:%M) waiting for Stage 1 calibration (calibrate_ml8_paged) to finish..."
while pgrep -f "[c]alibrate_ml8_paged" >/dev/null; do sleep 30; done
echo "[chain] $(date +%H:%M) Stage 1 process exited."

NBLOB=$(ls "$CALIB_DIR"/*.pt 2>/dev/null | wc -l)
echo "[chain] calib blobs found: $NBLOB (expect ~249 = 200 ml8 + 49 fp8)"
if [ "$NBLOB" -lt 200 ]; then
  echo "[chain] ABORT: only $NBLOB blobs — Stage 1 did not complete (target was ~249). Not converting."
  exit 1
fi

echo "[chain] $(date +%H:%M) === Stage 2: ml8_to_gguf → $OUT_GGUF ==="
python ml8_to_gguf.py --base-gguf "$BASE" --calib-dir "$CALIB_DIR" \
  --out-gguf "$OUT_GGUF" --allow-partial
if [ ! -s "$OUT_GGUF" ]; then
  echo "[chain] ABORT: $OUT_GGUF was not created. Stopping before smoke."
  exit 1
fi
echo "[chain] $(date +%H:%M) Stage 2 done: $(ls -lh "$OUT_GGUF" | awk '{print $5}')"

echo "[chain] $(date +%H:%M) === Stage 3: Phase E smoke (frozen, gptq, gptq-interleave) ==="
systemd-run --user --scope -p MemoryHigh=9G -p MemoryMax=11G --unit mad281-4b-smoke \
  python smoke_fp8_qat.py --model "$MODEL" --gguf "$OUT_GGUF" \
    --arms frozen,gptq,gptq-interleave --steps 30 --eval-interval 5
echo "[chain] $(date +%H:%M) === 4B PHASE E PIPELINE COMPLETE ==="
