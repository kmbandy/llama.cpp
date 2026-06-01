#!/usr/bin/env bash
# ml8 INFERENCE smoke gate for gfx942 / CDNA3 (MI300X).
#
# The open question this answers: does our ml8 inference path (AITER Triton +
# ggml-hip ml8.cu) produce CORRECT perplexity on CDNA3 — or only on the RDNA4
# R9700 it was validated on? If yes, a single MI300X runs the WHOLE pipeline
# (calibrate + convert + ppl), no R9700 tail. If the numbers are off, the prime
# suspect is the ML8_FP8 tier: CDNA uses float8_e4m3fnuz, RDNA/Hopper use e4m3fn,
# and our scaled-FP8 bytes (embed + ssm α/β + MTP) are calibrated as e4m3fn.
#
# Cost: ~$0.05 of instance time. Run AFTER smoke.sh (which proves torch+calibration)
# and BEFORE trusting any ml8 PPL produced on the instance.
#
# Usage (on the MI300X instance, inside the container):
#   ML8_GGUF=/models/qwen35-0p8b/Qwen3.5-0.8B-ml8-fullcov.gguf \
#   EXPECTED_PPL=<value-from-R9700-on-the-SAME-gguf+eval> \
#   bash scripts/calibration/mi300x/smoke_ml8_inference.sh
#
# Env:
#   ML8_GGUF      (required) full-coverage ml8 GGUF — must contain BOTH the ml8_4
#                 tier AND the ML8_FP8 tier so this exercises the e4m3fnuz risk.
#   DEVICE        llama.cpp device (default ROCm0 — the single MI300X).
#   EVAL_FILE     eval text (default: the embedded deterministic passage below, so
#                 the R9700 reference and the gfx942 run use byte-identical input).
#   CTX, CHUNKS   perplexity window (default 256 / 2 → ~512 tokens, enough to catch
#                 a corrupted FP8 tier; bump for a tighter number).
#   EXPECTED_PPL  reference PPL from the R9700 on the SAME gguf+eval. If set, the
#                 comparison is e4m3fnuz-sensitive (a broken FP8 tier shifts PPL).
#                 If unset, only the sane-range check runs (still catches NaN/garbage).
#   TOL           relative tolerance for the match (default 0.05 = 5%).
set -uo pipefail

: "${ML8_GGUF:?set ML8_GGUF to a full-coverage ml8 GGUF (ml8_4 + ML8_FP8 tiers)}"
DEVICE="${DEVICE:-ROCm0}"
CTX="${CTX:-256}"
CHUNKS="${CHUNKS:-2}"
TOL="${TOL:-0.05}"
BIN="${BIN:-/opt/mad-lab/llama.cpp/build-hip/bin/llama-perplexity}"

[ -x "$BIN" ] || { echo "FAIL: $BIN not found/executable — image built without llama-perplexity?"; exit 2; }
[ -f "$ML8_GGUF" ] || { echo "FAIL: ML8_GGUF '$ML8_GGUF' not found (stage it via s5cmd first)"; exit 2; }

# Deterministic embedded eval — identical bytes on R9700 and gfx942 so PPL is
# directly comparable. Content is generic encyclopedic prose (no copyright); the
# absolute PPL is meaningless, only the cross-backend match matters.
EVAL_FILE="${EVAL_FILE:-/tmp/ml8_smoke_eval.txt}"
if [ ! -s "$EVAL_FILE" ]; then
  cat > "$EVAL_FILE" <<'EOF'
The transformer architecture processes sequences of tokens by repeatedly applying
self-attention and position-wise feed-forward transformations. Each layer reads a
residual stream, normalizes it, and writes an additive update back. Attention lets
every position gather information from every other position, weighted by learned
compatibility scores, while the feed-forward block expands the representation into a
wider intermediate space and projects it back. Quantization replaces the original
floating-point weights with a compact code: a small set of representative values, an
index per weight, and a scale per group. The reconstruction error introduced by this
substitution is not uniform across the model; some projections tolerate aggressive
compression while others, especially those whose activations carry heavy-tailed
outliers, demand more precision to preserve the output distribution. A calibration
corpus drives the process by producing activation statistics, and the covariance of
those activations determines which directions in weight space must be protected.
Larger and more representative corpora yield sharper statistics, but with diminishing
returns once the dominant directions are well estimated. Recurrent state-space layers
add a further wrinkle, because their internal state evolves across the sequence, so
the activation distribution at the end of a long context differs from its beginning.
Careful engineering balances memory footprint against fidelity, aiming for a model
that is dramatically smaller yet behaves, token for token, almost exactly like the
full-precision original from which it was derived. The measure of success is whether
held-out text, never seen during calibration, is predicted with the same confidence.
EOF
fi

echo "=== ml8 inference smoke on $DEVICE ==="
echo "  gguf   : $ML8_GGUF"
echo "  eval   : $EVAL_FILE  ($(wc -w < "$EVAL_FILE") words)"
echo "  window : -c $CTX --chunks $CHUNKS"
echo "  expect : ${EXPECTED_PPL:-<none — sane-range check only>}"
echo

LOG="$(mktemp)"
"$BIN" --no-mmap -m "$ML8_GGUF" -ngl 99 --device "$DEVICE" \
       -f "$EVAL_FILE" -c "$CTX" --chunks "$CHUNKS" >"$LOG" 2>&1
RC=$?
PPL="$(grep -oE 'Final estimate: PPL = [0-9.]+(e[+-]?[0-9]+)?' "$LOG" | grep -oE '[0-9.]+(e[+-]?[0-9]+)?$' | tail -1)"

echo "--- tail of run ---"; tail -5 "$LOG"; echo "-------------------"

# Classify.
if [ "$RC" -ne 0 ] || [ -z "$PPL" ] || grep -qiE 'nan|inf|dispatch failed|abort|HIP error|assert' "$LOG"; then
  echo "RESULT: FAIL — ml8 inference did not run cleanly on gfx942 (rc=$RC, ppl='${PPL:-none}')."
  echo "  => the ml8 kernel path does not execute on CDNA3. MI300X is calib-only;"
  echo "     PPL stays on the R9700. (Check the tail above for 'dispatch failed' / HIP errors.)"
  rm -f "$LOG"; exit 1
fi

# Garbage but non-crashing (e.g. PPL in the thousands) → numerics broken.
awk -v p="$PPL" 'BEGIN{exit !(p>1000)}' && {
  echo "RESULT: SUSPECT — ran but PPL=$PPL is implausibly high → corrupted numerics,"
  echo "  prime suspect the ML8_FP8 tier (e4m3fnuz vs e4m3fn). The 4-bit ml8_4 tier"
  echo "  likely dispatches; the scaled-FP8 embed/ssm bytes are being misread on CDNA."
  echo "  Fix path: calibrate the FP8 tier in the inference target's fp8 format."
  rm -f "$LOG"; exit 3
}

if [ -n "${EXPECTED_PPL:-}" ]; then
  REL="$(awk -v a="$PPL" -v b="$EXPECTED_PPL" 'BEGIN{d=(a-b); if(d<0)d=-d; printf "%.4f", d/b}')"
  echo "PPL=$PPL  expected=$EXPECTED_PPL  rel-diff=$REL  tol=$TOL"
  if awk -v r="$REL" -v t="$TOL" 'BEGIN{exit !(r>t)}'; then
    echo "RESULT: SUSPECT — ran and is finite, but off the R9700 reference by >$((100*${TOL%.*}))%."
    echo "  Most likely the ML8_FP8 tier (e4m3fnuz). ml8_4 dispatches; FP8 numerics drift."
    echo "  Fix path: calibrate the FP8 tier in the inference target's fp8 format, re-test."
    rm -f "$LOG"; exit 3
  fi
  echo "RESULT: PASS — ml8 PPL on gfx942 matches the R9700 within $TOL."
  echo "  => FULL ml8 inference (both ml8_4 + ML8_FP8 tiers) is CORRECT on CDNA3."
  echo "     A single MI300X can run the whole pipeline — no R9700 ppl tail."
else
  echo "RESULT: SANE — ml8 PPL=$PPL is finite and plausible on gfx942 (no R9700 reference given)."
  echo "  Provide EXPECTED_PPL (same gguf+eval on the R9700) to confirm the FP8 tier numerically."
fi
rm -f "$LOG"
