#!/bin/bash
# After the 4B chain's smoke finishes: if Axis B (gptq-interleave) beat the frozen
# Axis-A floor by > 0.005 KL (a real move), run a proper-length converged QAT of the
# winning config. This is a MEASUREMENT run (no re-emit -> not a deployable artifact;
# the deployable QAT'd GGUF needs the re-emit step, which is a supervised dev task).
# Detached, RAM-safe scope. Fail-safe: if the smoke numbers can't be parsed, run it
# anyway (overnight GPU is free and the converged number is informative either way).
set -uo pipefail
cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration
CHAIN_LOG=/home/kmbandy/models/act_replay/MAD281_4B_chain.log
REAL_LOG=/home/kmbandy/models/act_replay/MAD281_4B_realrun.log
MODEL=/home/kmbandy/models/Qwen3.5-4B-hf
GGUF=/home/kmbandy/models/Qwen3.5-4B-ml8.gguf
exec > "$REAL_LOG" 2>&1

echo "[real-run] $(date +%H:%M) waiting for the 4B chain (smoke) to COMPLETE/ABORT..."
while ! grep -qE "PIPELINE COMPLETE|ABORT" "$CHAIN_LOG" 2>/dev/null; do sleep 60; done
if grep -q "ABORT" "$CHAIN_LOG"; then echo "[real-run] chain ABORTed — nothing to run."; exit 1; fi

FRO=$(grep -E "\[arm frozen\].*final" "$CHAIN_LOG" | tail -1 | grep -oE "final [0-9.]+" | awk '{print $2}')
GPI=$(grep -E "\[arm gptqi\].*final" "$CHAIN_LOG" | tail -1 | grep -oE "final [0-9.]+" | awk '{print $2}')
echo "[real-run] smoke finals: frozen=${FRO:-?}  gptq-interleave=${GPI:-?}"

RUN=1
if [ -n "${FRO:-}" ] && [ -n "${GPI:-}" ]; then
  if awk -v a="$GPI" -v b="$FRO" 'BEGIN{exit !(a < b - 0.005)}'; then
    echo "[real-run] PASS: Axis B cleared frozen by >0.005 — running the converged real QAT."
  else
    RUN=0
    echo "[real-run] INERT: Axis B did not clear frozen by >0.005 — skipping the longer run."
    echo "[real-run] (Finding: on the 4B, full-H reassignment is stable but adds nothing over Axis A.)"
  fi
else
  echo "[real-run] could not parse smoke finals — fail-safe: running the converged QAT anyway."
fi
[ "$RUN" = "1" ] || exit 0

echo "[real-run] $(date +%H:%M) === REAL QAT: frozen,gptq-interleave x150 steps ==="
systemd-run --user --scope -p MemoryHigh=9G -p MemoryMax=11G --unit mad281-4b-realrun \
  python smoke_fp8_qat.py --model "$MODEL" --gguf "$GGUF" \
    --arms frozen,gptq-interleave --steps 150 --eval-interval 10
echo "[real-run] $(date +%H:%M) === REAL QAT RUN COMPLETE ==="
