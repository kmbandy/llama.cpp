#!/usr/bin/env bash
# Profile a single prefill request under rocprofv3 --hip-trace, then summarize.
#
# Why this script exists: PMC counters on gfx1201/RDNA4 are broken in current
# ROCm releases (silently return 0 — see ROCm/rocm-systems#5953). --hip-trace
# bypasses hardware counters and gives us reliable per-kernel timing.
#
# Usage:
#   profile_prefill.sh <target_tokens> [label]
#       target_tokens   approximate token count for the prompt (8192, 32768, ...)
#       label           optional baseline label; defaults to ctx-COMMIT
#
# Outputs:
#   /tmp/claude-1000/perf_traces/<label>/        raw rocprof CSV files
#   tests/perf-baseline/mt_pagedattn/<label>.json   parsed summary
#
# Requirements (assumed already set up on this workstation):
#   - rocprofv3 in PATH
#   - llama-server built at build-hip/bin/llama-server with Q_TILES=6 multi-warp
#   - /tmp/claude-1000/build_prompt.py + scale_corpus.txt
#   - Qwen3.6-35B model + bge-small embedder paths as below

set -euo pipefail

TARGET_TOKENS="${1:-8192}"
LABEL="${2:-${TARGET_TOKENS}-$(git -C "$(dirname "$0")/../.." rev-parse --short HEAD)}"

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
TRACE_DIR="/tmp/claude-1000/perf_traces/${LABEL}"
BASELINE_DIR="${REPO_ROOT}/tests/perf-baseline/mt_pagedattn"
BASELINE_PATH="${BASELINE_DIR}/${LABEL}.json"
PROMPT_PATH="/tmp/claude-1000/prompt_${TARGET_TOKENS}_p1.json"
SERVER_LOG="${TRACE_DIR}/server.log"
RESP_PATH="${TRACE_DIR}/response.json"

mkdir -p "${TRACE_DIR}" "${BASELINE_DIR}"
rm -f "${TRACE_DIR}"/trace_*.csv "${RESP_PATH}"

# Build prompt with n_predict=1 (we only want prefill timings, not decode).
if [[ ! -f "${PROMPT_PATH}" ]]; then
    python3 /tmp/claude-1000/build_prompt.py "${TARGET_TOKENS}" "${PROMPT_PATH}.tmp"
    python3 -c "
import json
p='${PROMPT_PATH}.tmp'
r=json.load(open(p))
r['n_predict']=1
json.dump(r, open('${PROMPT_PATH}', 'w'))
"
    rm -f "${PROMPT_PATH}.tmp"
fi

# Refuse to run if a server or rocprof is already alive — we'd hit port-bind.
if pgrep -f "llama-server.*Qwen3.6" >/dev/null; then
    echo "ERROR: a llama-server is already running. Kill it first." >&2
    pgrep -af "llama-server.*Qwen3.6" >&2
    exit 1
fi

echo "[1/5] launching llama-server under rocprofv3 --hip-trace (label=${LABEL})..."
setsid nohup /opt/rocm/bin/rocprofv3 \
    --hip-trace --kernel-trace --memory-copy-trace \
    --kernel-include-regex 'mt_paged_attention_tile_mw_kernel' \
    -o trace -d "${TRACE_DIR}" \
    -f csv \
    -- "${REPO_ROOT}/build-hip/bin/llama-server" \
       --model /home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
       --n-gpu-layers 999 --ctx-size 1048576 --parallel 1 \
       --cache-type-k turbo4 --cache-type-v turbo4 \
       --kv-tiered 65,20,15 --kv-tier-ssd-path /home/kmbandy/kv-cold/qwen36-35b \
       --kv-tier-paged-blocks \
       --kv-tier-semantic-index /home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf \
       --flash-attn on --metrics --jinja --no-mmap --no-warmup \
       --host 0.0.0.0 --port 8090 --cache-ram 0 --ctx-checkpoints 0 \
       --timeout 3600 --alias qwen36 --device ROCm0 > "${SERVER_LOG}" 2>&1 < /dev/null &
disown
sleep 3

# Find the actual llama-server child of rocprof (not the rocprof parent).
SERVER_PID=""
for _ in $(seq 1 60); do
    SERVER_PID=$(pgrep -f "build-hip/bin/llama-server.*Qwen3.6" | head -1 || true)
    [[ -n "${SERVER_PID}" ]] && break
    sleep 1
done
if [[ -z "${SERVER_PID}" ]]; then
    echo "ERROR: failed to find llama-server PID" >&2
    tail -20 "${SERVER_LOG}" >&2
    exit 1
fi
echo "  server PID: ${SERVER_PID}"

echo "[2/5] waiting for server ready..."
for i in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8090/health -m 2 >/dev/null 2>&1; then
        echo "  ready after ${i}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server died during startup" >&2
        tail -20 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 1
done

echo "[3/5] sending prefill (target=${TARGET_TOKENS} tokens, n_predict=1)..."
TIMEOUT=$((TARGET_TOKENS / 50 + 60))  # generous: ~50 t/s under rocprof + 60s slack
curl -fsS -X POST http://127.0.0.1:8090/v1/completions \
    -H "Content-Type: application/json" \
    --data-binary @"${PROMPT_PATH}" \
    -o "${RESP_PATH}" -m "${TIMEOUT}"

PREFILL_INFO=$(python3 -c "
import json
r = json.load(open('${RESP_PATH}'))
t = r.get('timings', {})
print(f\"{t.get('prompt_n', 0)} {t.get('prompt_ms', 0):.2f} {t.get('prompt_per_second', 0):.2f}\")
")
read -r ACTUAL_TOKENS PREFILL_MS PREFILL_TPS <<< "${PREFILL_INFO}"
echo "  prefill: tokens=${ACTUAL_TOKENS}  ms=${PREFILL_MS}  t/s=${PREFILL_TPS}"

echo "[4/5] killing server cleanly (so rocprof finalizes)..."
kill -TERM "${SERVER_PID}"
for i in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  exited after ${i}s"
        break
    fi
    sleep 1
done
# Give rocprof a few seconds to flush CSVs.
sleep 4

KERNEL_CSV="${TRACE_DIR}/trace_kernel_trace.csv"
if [[ ! -f "${KERNEL_CSV}" ]]; then
    echo "ERROR: rocprof did not produce ${KERNEL_CSV}" >&2
    ls -la "${TRACE_DIR}" >&2
    exit 1
fi

echo "[5/5] parsing trace + writing baseline..."
COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
python3 "${REPO_ROOT}/scripts/perf/parse_prefill_trace.py" \
    "${KERNEL_CSV}" \
    --out "${BASELINE_PATH}" \
    --ctx-tokens "${ACTUAL_TOKENS}" \
    --prefill-ms "${PREFILL_MS}" \
    --prefill-tps "${PREFILL_TPS}" \
    --commit "${COMMIT}" \
    --label "${LABEL}"
