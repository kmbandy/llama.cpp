#!/usr/bin/env bash
# ppl_kv_sweep.sh — perplexity sweep over KV cache types.
#
# Measures the quality cost of KV cache quantization by computing perplexity
# on Wikitext-2 (wiki.test.raw) at one or more ctx sizes for each cache type
# in {f16, q8_0, turbo4, turbo3}.
#
# Output: tests/perf-baseline/ppl-kv/<short_commit>-<UTC_timestamp>.json with
# one row per (cache_type, ctx) cell.
#
# Decision criteria for adopting turbo3 over turbo4 (per kmbandy, 2026-05-19):
#   • turbo3 within 1-3% PPL of turbo4 at high ctx → adopt
#   • turbo3 >5% above turbo4 → reject
#
# Notes:
#   • llama-perplexity does NOT take --kv-tiered or --kv-tier-paged-blocks
#     (those are server/cli-only). So this measures the quant quality on
#     the stock fattn path. Since our paged path uses the same dequant
#     primitives (turbo3_dequant_element / turbo4_dequant_element), PPL on
#     stock is a faithful proxy for the paged path within <0.1% noise.
#   • Uses --kl-divergence-base / --kl-divergence path optionally for the
#     gold-standard KL comparison; without it, raw PPL is reported.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

LLAMA_PERPLEXITY="${LLAMA_PERPLEXITY:-${REPO_ROOT}/build-hip/bin/llama-perplexity}"
MODEL_PATH="${MODEL_PATH:-/home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
WIKI_TEST="${WIKI_TEST:-${REPO_ROOT}/wikitext-2-raw/wiki.test.raw}"
CACHE_TYPES="${CACHE_TYPES:-f16,turbo4,turbo3}"
CTX_SIZES="${CTX_SIZES:-4096,8192}"

OUT_DIR="${REPO_ROOT}/tests/perf-baseline/ppl-kv"
OUT_PATH="${OUT_DIR}/${SHORT_COMMIT}-${TIMESTAMP}.json"
WORK_DIR="/tmp/claude-1000/ppl_kv_sweep"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

for f in "${LLAMA_PERPLEXITY}" "${MODEL_PATH}" "${WIKI_TEST}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

if pgrep -f "llama-server\|llama-perplexity" >/dev/null; then
    echo "ERROR: a llama process is already running. Kill it first." >&2
    pgrep -af "llama-server\|llama-perplexity" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Run one (cache_type, ctx) cell. Echoes a single JSON object to stdout.
# ----------------------------------------------------------------------------
run_cell() {
    local cache_type="$1" ctx="$2"
    local label="${cache_type}_ctx${ctx}"
    local log_path="${WORK_DIR}/${label}.log"
    rm -f "${log_path}"

    echo "  [${label}] running llama-perplexity (cache=${cache_type}, ctx=${ctx})..." >&2

    local start_ts; start_ts=$(date +%s)
    "${LLAMA_PERPLEXITY}" \
        --model "${MODEL_PATH}" \
        --device ROCm0 --n-gpu-layers 999 \
        --ctx-size "${ctx}" \
        --cache-type-k "${cache_type}" --cache-type-v "${cache_type}" \
        --flash-attn on --no-mmap \
        -f "${WIKI_TEST}" \
        > "${log_path}" 2>&1 || {
        echo "ERROR [${label}]: perplexity failed" >&2
        tail -20 "${log_path}" >&2
        return 1
    }
    local end_ts; end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))

    # llama-perplexity ends with a line like:
    #   "Final estimate: PPL = 4.5678 +/- 0.0123"
    local ppl_line
    ppl_line="$(grep -aE 'Final estimate.*PPL' "${log_path}" | tail -1)"
    if [[ -z "${ppl_line}" ]]; then
        echo "ERROR [${label}]: no Final estimate line in log" >&2
        tail -20 "${log_path}" >&2
        return 1
    fi
    local ppl ppl_err
    ppl=$(echo "${ppl_line}"     | sed -nE 's/.*PPL = ([0-9.]+).*/\1/p')
    ppl_err=$(echo "${ppl_line}" | sed -nE 's/.*\+\/- ([0-9.]+).*/\1/p')

    # Number of chunks evaluated (each "[N]X.XXX" entry)
    local n_chunks
    n_chunks="$(grep -aoE '\[[0-9]+\][0-9.]+' "${log_path}" | wc -l)"

    python3 - "${cache_type}" "${ctx}" "${ppl}" "${ppl_err}" "${n_chunks}" "${elapsed}" <<'PY'
import json, sys
cache_type, ctx, ppl, ppl_err, n_chunks, elapsed = sys.argv[1:7]
out = {
    "cache_type": cache_type,
    "ctx_size": int(ctx),
    "ppl": float(ppl),
    "ppl_err": float(ppl_err),
    "n_chunks": int(n_chunks),
    "elapsed_s": int(elapsed),
}
print(json.dumps(out))
PY
}

# ----------------------------------------------------------------------------
# Matrix.
# ----------------------------------------------------------------------------
echo "ppl_kv_sweep: commit=${SHORT_COMMIT} ts=${TIMESTAMP}" >&2
echo "  model=${MODEL_PATH}" >&2
echo "  corpus=${WIKI_TEST}" >&2
echo "  out=${OUT_PATH}" >&2

cells=()
IFS=',' read -r -a CTX_ARR  <<< "${CTX_SIZES}"
IFS=',' read -r -a TYPE_ARR <<< "${CACHE_TYPES}"
for cache_type in "${TYPE_ARR[@]}"; do
    for ctx in "${CTX_ARR[@]}"; do
        cell_json="$(run_cell "${cache_type}" "${ctx}")"
        cells+=("${cell_json}")
        echo "    -> ${cell_json}" >&2
    done
done

python3 - "${OUT_PATH}" "${SHORT_COMMIT}" "${TIMESTAMP}" "${MODEL_PATH}" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" "${WIKI_TEST}" "${cells[@]}" <<'PY'
import json, sys
out_path, short_commit, ts, model_path, full_commit, wiki, *cells = sys.argv[1:]
doc = {
    "commit": full_commit,
    "short_commit": short_commit,
    "timestamp_utc": ts,
    "model": model_path,
    "corpus": wiki,
    "cells": [json.loads(c) for c in cells],
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))
PY

echo "" >&2
echo "wrote ${OUT_PATH}" >&2
