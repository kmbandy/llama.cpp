#!/usr/bin/env bash
# ml8_calibration_sweep.sh — MAD-223 Phase B.3 weight-side calibration sweep.
#
# Driver for the ml8-4 weight calibration matrix sweep, analog of the
# MAD-214 KV-side turbo_fp8_calibration_sweep.sh. Each cell runs:
#
#   1. python3 scripts/calibration/calibrate_ml8.py with the cell's args,
#      including --eval-ppl --ppl-max-tokens 100000 so PPL is captured.
#   2. Read the resulting manifest.json (ppl_baseline, ppl_quantized, ppl_delta,
#      per-layer Y_SNR/W_SNR).
#   3. Append cell row to results JSON.
#
# Output: tests/perf-baseline/ml8-calibration-sweep/<short_commit>-<UTC_ts>.json
#
# Locked-bench discipline: every cell uses identical model, calibration corpus
# size, and PPL eval window. The only variables across cells are the lever
# under test. Same f16 baseline (computed once per run) is reused for all
# Δ_PPL calculations within the run.
#
# Usage:
#   bash scripts/perf/ml8_calibration_sweep.sh             # full Pass 1 sweep
#   CELLS="baseline hadamard" bash scripts/perf/ml8_calibration_sweep.sh   # subset

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

RESULTS_DIR="${REPO_ROOT}/tests/perf-baseline/ml8-calibration-sweep"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/${SHORT_COMMIT}-${TIMESTAMP}-pass1.json"

WORK_ROOT="${WORK_ROOT:-/tmp/ml8-sweep-${SHORT_COMMIT}-${TIMESTAMP}}"
mkdir -p "$WORK_ROOT"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
N_SAMPLES="${N_SAMPLES:-32}"
SEQ_LEN="${SEQ_LEN:-1024}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-100000}"
DEVICE="${DEVICE:-cuda:0}"

# Cells are 'label|extra_args'. Args are appended to the common base; common
# args = --model, --n-samples, --seq-len, --eval-ppl, --device, --output-dir.
declare -a ALL_CELLS=(
  "baseline|--fit-loss mse --group-size 128 --n-centroids 16"
  "hadamard|--fit-loss mse --group-size 128 --n-centroids 16 --rotation kronecker"
  "h_mag_p0.5|--fit-loss mag_weighted --mag-weight-p 0.5 --group-size 128 --n-centroids 16 --rotation kronecker"
  "h_mag_p1|--fit-loss mag_weighted --mag-weight-p 1.0 --group-size 128 --n-centroids 16 --rotation kronecker"
  "h_mag_p2|--fit-loss mag_weighted --mag-weight-p 2.0 --group-size 128 --n-centroids 16 --rotation kronecker"
  "h_gs64|--fit-loss mse --group-size 64 --n-centroids 16 --rotation kronecker"
  "h_gs256|--fit-loss mse --group-size 256 --n-centroids 16 --rotation kronecker"
  "h_nc32|--fit-loss mse --group-size 128 --n-centroids 32 --rotation kronecker"
)

# Allow CELLS env var to restrict which labels run (whitespace-separated).
if [ -n "${CELLS:-}" ]; then
  declare -a CELL_FILTER=( $CELLS )
fi

cell_should_run() {
  local label="$1"
  if [ -z "${CELLS:-}" ]; then return 0; fi
  for c in "${CELL_FILTER[@]}"; do
    [ "$c" = "$label" ] && return 0
  done
  return 1
}

# Initialize results JSON
cat > "$RESULTS_FILE" <<EOF
{
  "commit": "${SHORT_COMMIT}",
  "timestamp": "${TIMESTAMP}",
  "model": "${MODEL}",
  "n_samples": ${N_SAMPLES},
  "seq_len": ${SEQ_LEN},
  "ppl_max_tokens": ${PPL_MAX_TOKENS},
  "cells": []
}
EOF

echo "[sweep] commit=${SHORT_COMMIT} ts=${TIMESTAMP}"
echo "[sweep] results: ${RESULTS_FILE}"
echo "[sweep] work:    ${WORK_ROOT}"
echo ""

for cell_spec in "${ALL_CELLS[@]}"; do
  IFS='|' read -r label args <<< "$cell_spec"
  if ! cell_should_run "$label"; then
    echo "[skip] $label (not in CELLS filter)"
    continue
  fi

  CELL_DIR="${WORK_ROOT}/${label}"
  rm -rf "$CELL_DIR"
  mkdir -p "$CELL_DIR"
  LOG="${CELL_DIR}/run.log"

  echo "===================================================================="
  echo "[cell] $label  args=$args"
  echo "[cell] $(date -u +%H:%M:%SZ) start"
  echo "===================================================================="

  python3 "${REPO_ROOT}/scripts/calibration/calibrate_ml8.py" \
    --model "$MODEL" \
    --output-dir "$CELL_DIR" \
    --n-samples "$N_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --eval-ppl --ppl-max-tokens "$PPL_MAX_TOKENS" \
    --device "$DEVICE" \
    $args 2>&1 | tee "$LOG" | tail -30

  T_END=$(date -u +%H:%M:%SZ)
  echo "[cell] $label done at $T_END"

  # Append cell row to results JSON
  python3 - "$RESULTS_FILE" "$label" "$CELL_DIR/manifest.json" "$args" <<'PY'
import json, sys, statistics
results_file, label, manifest_path, args = sys.argv[1:5]
with open(manifest_path) as f:
    m = json.load(f)
y_snrs = [r["y_snr_db"] for r in m.get("results", [])]
w_snrs = [r["w_snr_db"] for r in m.get("results", [])]
cell = {
    "label": label,
    "args": args,
    "ppl_baseline": m.get("ppl_baseline", {}).get("ppl"),
    "ppl_quantized": m.get("ppl_quantized", {}).get("ppl"),
    "ppl_delta": m.get("ppl_delta"),
    "n_layers": len(m.get("results", [])),
    "y_snr_median_db": statistics.median(y_snrs) if y_snrs else None,
    "y_snr_min_db": min(y_snrs) if y_snrs else None,
    "y_snr_max_db": max(y_snrs) if y_snrs else None,
    "w_snr_median_db": statistics.median(w_snrs) if w_snrs else None,
}
with open(results_file) as f:
    out = json.load(f)
out["cells"].append(cell)
with open(results_file, "w") as f:
    json.dump(out, f, indent=2)
print(f"  -> ppl_delta={cell['ppl_delta']:+.4f}  y_snr_median={cell['y_snr_median_db']:.1f}dB")
PY

  echo ""
done

echo "[sweep] ALL CELLS COMPLETE"
echo "[sweep] results: $RESULTS_FILE"
python3 -c "
import json
with open('$RESULTS_FILE') as f:
    out = json.load(f)
print()
print(f\"{'label':<14}{'ppl_baseline':<14}{'ppl_quant':<14}{'Δ_PPL':<10}{'Y_SNR_med':<12}\")
print('-' * 64)
for c in out['cells']:
    pb = f\"{c['ppl_baseline']:.4f}\" if c.get('ppl_baseline') else '—'
    pq = f\"{c['ppl_quantized']:.4f}\" if c.get('ppl_quantized') else '—'
    dp = f\"{c['ppl_delta']:+.4f}\" if c.get('ppl_delta') is not None else '—'
    ys = f\"{c['y_snr_median_db']:.1f}dB\" if c.get('y_snr_median_db') else '—'
    print(f\"{c['label']:<14}{pb:<14}{pq:<14}{dp:<10}{ys:<12}\")
"
