#!/usr/bin/env bash
# bench_paged_paths.sh — bench AITER on/off × prefill+decode × 8K/64K.
#
# Mechanizes the "test both paged attention paths every commit" rule. Unlike
# profile_prefill.sh this does NOT use rocprof (which slows the GPU ~2x and
# hides production-curve regressions); it runs llama-server clean and reads
# timings off the /v1/completions response.
#
# Two axes:
#   aiter ∈ {0, 1}     — MAD_USE_AITER=1 forces the AITER unified_attention
#                        path. AITER currently asserts F16 KV (MAD-199), so
#                        the AITER row uses --cache-type-k f16; the non-AITER
#                        row uses the production TURBO4 cache type.
#   ctx   ∈ {8K, 64K}  — short prompt + long prompt, covers the bend in the
#                        prefill curve where the tile kernel kicks in.
#
# Output: tests/perf-baseline/paged-paths/<short_commit>-<UTC_timestamp>.json
#
# Env vars (optional):
#   LLAMA_SERVER       override path to llama-server binary
#                      (default: <repo>/build-hip/bin/llama-server)
#   MODEL_PATH         override model gguf
#                      (default: /home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)
#   N_PREDICT          decode tokens to generate (default: 32)
#   PORT               server port (default: 8090)
#   CTX_SIZES          comma-separated target prompt tokens (default: 8192,65536)

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_ROOT}/build-hip/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
N_PREDICT="${N_PREDICT:-32}"
PORT="${PORT:-8090}"
CTX_SIZES="${CTX_SIZES:-8192,65536}"

OUT_DIR="${REPO_ROOT}/tests/perf-baseline/paged-paths"
OUT_PATH="${OUT_DIR}/${SHORT_COMMIT}-${TIMESTAMP}.json"
WORK_DIR="/tmp/claude-1000/bench_paged_paths"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

for f in "${LLAMA_SERVER}" "${MODEL_PATH}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

if pgrep -f "llama-server" >/dev/null; then
    echo "ERROR: a llama-server is already running. Kill it first." >&2
    pgrep -af "llama-server" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Synthesize prompts.
#
# We don't have the model tokenizer here; we generate a deterministic English
# string sized at ~3.6 chars/token (Qwen BPE empirical average). Actual token
# count comes back in the response's `timings.prompt_n` and is what we record.
# ----------------------------------------------------------------------------
prompt_file_for() {
    local target_tokens="$1"
    local path="${WORK_DIR}/prompt_${target_tokens}.json"
    if [[ -f "${path}" ]]; then
        echo "${path}"; return
    fi
    python3 - "${target_tokens}" "${path}" "${N_PREDICT}" <<'PY'
import json, sys
target_tokens, out_path, n_predict = sys.argv[1:4]
target_tokens = int(target_tokens); n_predict = int(n_predict)
# ~3.6 chars/token avg on Qwen for English prose. Pad a bit so we land at
# or just above the target.
char_target = int(target_tokens * 3.8)
seed = (
    "The pagedattn benchmark needles its way through tiered KV cache, "
    "stitching tile kernels to flash-decode and watching where the bytes "
    "actually go. Tungsten marmot four nineteen lives in section twelve. "
)
prompt = (seed * ((char_target // len(seed)) + 2))[:char_target]
body = {
    "prompt": prompt,
    "n_predict": n_predict,
    "temperature": 0.0,
    "cache_prompt": False,
    "stream": False,
}
with open(out_path, "w") as f:
    json.dump(body, f)
PY
    echo "${path}"
}

# ----------------------------------------------------------------------------
# Run one (aiter, ctx) cell. Echoes a single JSON object to stdout.
# ----------------------------------------------------------------------------
run_cell() {
    local aiter="$1" target_tokens="$2"
    local prompt_path; prompt_path="$(prompt_file_for "${target_tokens}")"

    local label="aiter${aiter}_ctx${target_tokens}"
    local server_log="${WORK_DIR}/${label}.server.log"
    local resp_path="${WORK_DIR}/${label}.resp.json"
    rm -f "${server_log}" "${resp_path}"

    # AITER currently requires F16 KV (MAD-199). Non-AITER uses prod TURBO4.
    local cache_type
    if [[ "${aiter}" == "1" ]]; then
        cache_type="f16"
    else
        cache_type="turbo4"
    fi
    # ctx-size: must be >= target tokens + n_predict. Round up for headroom.
    local ctx_size=$(( target_tokens + N_PREDICT + 256 ))
    if (( ctx_size < 8192 )); then ctx_size=8192; fi

    echo "  [${label}] launching server (cache=${cache_type}, ctx=${ctx_size})..." >&2

    MAD_USE_AITER="${aiter}" setsid nohup "${LLAMA_SERVER}" \
        --model "${MODEL_PATH}" \
        --device ROCm0 --n-gpu-layers 999 \
        --ctx-size "${ctx_size}" --parallel 1 \
        --kv-tiered 100,0,0 \
        --cache-type-k "${cache_type}" --cache-type-v "${cache_type}" \
        --flash-attn on --no-mmap --no-warmup \
        --cache-ram 0 --ctx-checkpoints 0 \
        --host 127.0.0.1 --port "${PORT}" \
        --timeout 3600 --alias bench \
        > "${server_log}" 2>&1 < /dev/null &
    disown
    local server_pid=""
    for _ in $(seq 1 30); do
        server_pid="$(pgrep -f "${LLAMA_SERVER} --model ${MODEL_PATH}" | head -1 || true)"
        [[ -n "${server_pid}" ]] && break
        sleep 1
    done
    if [[ -z "${server_pid}" ]]; then
        echo "ERROR [${label}]: server PID not found" >&2
        tail -30 "${server_log}" >&2
        return 1
    fi

    # Wait for /health.
    local ready=0
    for i in $(seq 1 300); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" -m 2 >/dev/null 2>&1; then
            ready=1; break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "ERROR [${label}]: server died during startup" >&2
            tail -40 "${server_log}" >&2
            return 1
        fi
        sleep 1
    done
    if (( ready == 0 )); then
        echo "ERROR [${label}]: server not ready after 300s" >&2
        kill -TERM "${server_pid}" 2>/dev/null || true
        return 1
    fi

    # Generous timeout: prefill at the secondary-path low end is ~200 t/s
    # at 400K so even 64K should finish under 5 min. Add slack for decode.
    local timeout=$(( target_tokens / 50 + N_PREDICT * 2 + 120 ))
    local http_rc=0
    curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        --data-binary @"${prompt_path}" \
        -o "${resp_path}" -m "${timeout}" || http_rc=$?

    kill -TERM "${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "${server_pid}" 2>/dev/null || break
        sleep 1
    done

    if (( http_rc != 0 )) || [[ ! -s "${resp_path}" ]]; then
        echo "ERROR [${label}]: completion request failed (curl rc=${http_rc})" >&2
        tail -30 "${server_log}" >&2
        return 1
    fi

    python3 - "${aiter}" "${target_tokens}" "${cache_type}" "${ctx_size}" "${resp_path}" <<'PY'
import json, sys
aiter, target_tokens, cache_type, ctx_size, resp_path = sys.argv[1:6]
r = json.load(open(resp_path))
t = r.get("timings", {})
out = {
    "aiter": int(aiter),
    "target_tokens": int(target_tokens),
    "cache_type": cache_type,
    "ctx_size": int(ctx_size),
    "prompt_n": t.get("prompt_n"),
    "prompt_ms": t.get("prompt_ms"),
    "prompt_per_second": t.get("prompt_per_second"),
    "predicted_n": t.get("predicted_n"),
    "predicted_ms": t.get("predicted_ms"),
    "predicted_per_second": t.get("predicted_per_second"),
}
print(json.dumps(out))
PY
}

# ----------------------------------------------------------------------------
# Matrix.
# ----------------------------------------------------------------------------
echo "bench_paged_paths: commit=${SHORT_COMMIT} ts=${TIMESTAMP}" >&2
echo "  out=${OUT_PATH}" >&2

cells=()
IFS=',' read -r -a CTX_ARR <<< "${CTX_SIZES}"
for aiter in 0 1; do
    for ctx in "${CTX_ARR[@]}"; do
        cell_json="$(run_cell "${aiter}" "${ctx}")"
        cells+=("${cell_json}")
    done
done

# Assemble final JSON.
python3 - "${OUT_PATH}" "${SHORT_COMMIT}" "${TIMESTAMP}" "${MODEL_PATH}" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" "${cells[@]}" <<'PY'
import json, sys
out_path, short_commit, ts, model_path, full_commit, *cells = sys.argv[1:]
doc = {
    "commit": full_commit,
    "short_commit": short_commit,
    "timestamp_utc": ts,
    "model": model_path,
    "cells": [json.loads(c) for c in cells],
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))
PY

echo "" >&2
echo "wrote ${OUT_PATH}" >&2
