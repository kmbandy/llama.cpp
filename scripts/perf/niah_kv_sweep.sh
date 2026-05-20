#!/usr/bin/env bash
# niah_kv_sweep.sh — needle-in-a-haystack sweep over KV cache types.
#
# Measures long-context retrieval quality of KV cache quantization. For each
# (cache_type, ctx_size) cell:
#   1. Build a haystack from wikitext-2 train corpus sized to roughly fill ctx.
#   2. Insert a known needle phrase at ~50% depth.
#   3. Append a query asking for the secret passphrase.
#   4. POST to llama-server (with paged framework engaged via --kv-tier-paged-blocks).
#   5. Pass/fail = response contains the needle phrase exactly.
#
# Output: tests/perf-baseline/niah-kv/<short_commit>-<UTC_timestamp>.json
#
# Why this complements PPL: PPL averages over all tokens — a quant scheme can
# preserve average likelihood while breaking a single critical retrieval. NIAH
# is the binary "did the model find the needle" signal that long-ctx production
# users actually care about.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_ROOT}/build-hip/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
HAYSTACK_CORPUS="${HAYSTACK_CORPUS:-${REPO_ROOT}/wikitext-2-raw/wiki.train.raw}"
PORT="${PORT:-8090}"
NEEDLE="${NEEDLE:-The secret passphrase is verdant-quokka-7831.}"
NEEDLE_TOKEN="${NEEDLE_TOKEN:-verdant-quokka-7831}"
QUERY="${QUERY:-Question: What is the secret passphrase mentioned in the document? Reply with only the passphrase, nothing else. Answer:}"

# Tier config crossover. Targets <= TIER_HOT_MAX use --kv-tiered 100,0,0
# (everything in VRAM). Larger targets engage the production tiered config
# (65,20,15 + ssd-path + semantic-index) — the only way 256K+ prompts fit
# alongside the 35B-A3B model on a 32GB R9700.
TIER_HOT_MAX="${TIER_HOT_MAX:-65536}"
KV_TIER_SSD_PATH="${KV_TIER_SSD_PATH:-/home/kmbandy/kv-cold/qwen36-35b}"
KV_TIER_SEMANTIC_INDEX="${KV_TIER_SEMANTIC_INDEX:-/home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf}"

OUT_DIR="${REPO_ROOT}/tests/perf-baseline/niah-kv"
OUT_PATH="${OUT_DIR}/${SHORT_COMMIT}-${TIMESTAMP}.json"
WORK_DIR="/tmp/claude-1000/niah_kv_sweep"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

for f in "${LLAMA_SERVER}" "${MODEL_PATH}" "${HAYSTACK_CORPUS}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

if pgrep -f "llama-server\|llama-perplexity" >/dev/null; then
    echo "ERROR: a llama process is already running. Kill it first." >&2
    pgrep -af "llama-server\|llama-perplexity" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Build a prompt JSON for a given target ctx. Caches per-ctx so the same
# haystack is reused across cache types (apples-to-apples comparison).
# ----------------------------------------------------------------------------
prompt_file_for() {
    local target_tokens="$1"
    local path="${WORK_DIR}/niah_prompt_${target_tokens}.json"
    if [[ -f "${path}" ]]; then
        echo "${path}"; return
    fi
    python3 - "${target_tokens}" "${path}" "${HAYSTACK_CORPUS}" "${NEEDLE}" "${QUERY}" <<'PY'
import json, sys
target_tokens, out_path, corpus_path, needle, query = sys.argv[1:6]
target_tokens = int(target_tokens)
# Target haystack size in chars. Qwen BPE on English prose is ~4.78 chars/tok
# empirically; leave ~5% headroom for the needle + query suffix tokens.
# Aim for 80% of target so the prompt itself doesn't run up against ctx-size
# (admission machinery + decode budget needs headroom).
char_target = int(target_tokens * 4.78 * 0.80)
with open(corpus_path, "rb") as f:
    raw = f.read().decode("utf-8", errors="replace")
# Trim leading blank lines / wikitext metadata
raw = raw.lstrip()
if len(raw) < char_target:
    # Tile the corpus so we have enough text
    raw = (raw * ((char_target // len(raw)) + 2))[:char_target]
else:
    raw = raw[:char_target]
# Insert needle at ~50% depth. Split on the nearest paragraph break for
# naturalness; fall back to absolute position if no break found nearby.
mid = len(raw) // 2
para_break = raw.find("\n\n", mid)
if para_break < 0 or para_break > mid + 4000:
    para_break = mid
needle_block = f"\n\n{needle}\n\n"
haystack = raw[:para_break] + needle_block + raw[para_break:]
# Chat-format prompt: wrap the haystack + question as a user message so the
# chat-tuned model actually treats it as a question to answer (instead of
# continuing the wikipedia-style haystack). Use /v1/chat/completions endpoint
# in run_cell, which auto-applies the model's chat template (Qwen3 ChatML).
user_content = (
    "I am going to give you a long passage of text. Somewhere inside it, "
    "a sentence reveals a secret passphrase. Read the whole passage, then "
    "answer my question at the end.\n\n"
    "=== BEGIN PASSAGE ===\n"
    + haystack +
    "\n=== END PASSAGE ===\n\n"
    + query
)
body = {
    "messages": [{"role": "user", "content": user_content}],
    "max_tokens": 64,
    # Tiny non-zero temperature to break the degenerate-repetition loops
    # we observed at temperature=0 on long haystacks. NIAH cares about
    # retrieval correctness, not bit-exact reproducibility.
    "temperature": 0.1,
    "top_p": 0.9,
    "stream": False,
    "cache_prompt": False,
    # Qwen3 ChatML defaults to enabling the <think>...</think> reasoning
    # block, which gobbles our token budget before any actual answer is
    # emitted. Disable it so the model goes straight to the response.
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(out_path, "w") as f:
    json.dump(body, f)
PY
    echo "${path}"
}

# ----------------------------------------------------------------------------
# Run one (cache_type, ctx) cell. Starts server, posts prompt, parses result.
# Echoes a single JSON object to stdout.
# ----------------------------------------------------------------------------
run_cell() {
    local cache_type="$1" target_tokens="$2"
    local prompt_path; prompt_path="$(prompt_file_for "${target_tokens}")"
    local label="${cache_type}_ctx${target_tokens}"
    local server_log="${WORK_DIR}/${label}.server.log"
    local resp_path="${WORK_DIR}/${label}.resp.json"
    rm -f "${server_log}" "${resp_path}"

    # Per-cell tier config. Short cells (<=TIER_HOT_MAX) stay all-hot so the
    # measurement is isolated to the kernel and KV format; long cells engage
    # the production tiered stack (warm→RAM, cold→SSD, semantic-index) which
    # is the only way 256K+ prompts fit alongside the 35B-A3B model.
    local tier_flags ctx_size tier_mode
    if (( target_tokens <= TIER_HOT_MAX )); then
        tier_mode="100,0,0"
        tier_flags="--kv-tiered 100,0,0 --kv-tier-paged-blocks --ctx-checkpoints 0"
        ctx_size=$(( target_tokens * 2 ))
        if (( ctx_size < 32768 )); then ctx_size=32768; fi
    else
        tier_mode="65,20,15"
        # ctx-size scales with target so the hot-pool layer allocation fits in
        # the R9700's 32GB alongside the 35B model (model ~21GB → ~9GB free).
        # At ~25KB hot KV per token, hot=65% of (target*2) keeps the hot pool
        # under that budget for targets up to ~262K. The warm→RAM + cold→SSD
        # wiring still loads so the tier infrastructure is exercised.
        tier_flags="--kv-tiered 65,20,15 --kv-tier-paged-blocks --ctx-checkpoints 0 --kv-tier-ssd-path ${KV_TIER_SSD_PATH} --kv-tier-semantic-index ${KV_TIER_SEMANTIC_INDEX}"
        ctx_size=$(( target_tokens * 2 ))
    fi

    echo "  [${label}] launching server (cache=${cache_type}, ctx=${ctx_size}, tier=${tier_mode})..." >&2

    # --jinja is REQUIRED for /v1/chat/completions to honor the model's
    # chat template + chat_template_kwargs in the request body. Without it
    # the model produces degenerate output (observed: "The leFallrésrates..."
    # on Qwen3.6 with a 7K-token haystack prompt).
    # shellcheck disable=SC2086 # tier_flags is intentionally word-split
    setsid nohup "${LLAMA_SERVER}" \
        --model "${MODEL_PATH}" \
        --device ROCm0 --n-gpu-layers 999 \
        --ctx-size "${ctx_size}" --parallel 1 \
        ${tier_flags} \
        --cache-type-k "${cache_type}" --cache-type-v "${cache_type}" \
        --flash-attn on --no-mmap --no-warmup \
        --cache-ram 0 --jinja \
        --host 127.0.0.1 --port "${PORT}" \
        --timeout 3600 --alias niah \
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
    (( ready == 0 )) && { kill -KILL "${server_pid}" 2>/dev/null || true; return 1; }

    local timeout=$(( target_tokens / 50 + 240 ))
    local http_rc=0
    local start_ts; start_ts=$(date +%s)
    curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @"${prompt_path}" \
        -o "${resp_path}" -m "${timeout}" || http_rc=$?
    local end_ts; end_ts=$(date +%s)

    kill -KILL "${server_pid}" 2>/dev/null || true
    sleep 2

    if (( http_rc != 0 )) || [[ ! -s "${resp_path}" ]]; then
        echo "ERROR [${label}]: completion request failed (curl rc=${http_rc})" >&2
        tail -30 "${server_log}" >&2
        return 1
    fi

    python3 - "${cache_type}" "${target_tokens}" "${ctx_size}" "${resp_path}" "${NEEDLE_TOKEN}" "$(( end_ts - start_ts ))" <<'PY'
import json, sys
cache_type, target_tokens, ctx_size, resp_path, needle_token, elapsed = sys.argv[1:7]
r = json.load(open(resp_path))
t = r.get("timings", {})
# /v1/chat/completions returns choices[0].message.content (vs /v1/completions
# which has choices[0].text). Fall back to "text" in case llama-server's chat
# endpoint shape ever diverges.
choice = r.get("choices", [{}])[0]
text = choice.get("message", {}).get("content") or choice.get("text", "") or ""
hit = needle_token in text
out = {
    "cache_type": cache_type,
    "target_tokens": int(target_tokens),
    "ctx_size": int(ctx_size),
    "prompt_n": t.get("prompt_n"),
    "prompt_per_second": t.get("prompt_per_second"),
    "predicted_n": t.get("predicted_n"),
    "response": text.strip(),
    "needle_hit": hit,
    "elapsed_s": int(elapsed),
}
print(json.dumps(out))
PY
}

# ----------------------------------------------------------------------------
# Matrix.
# ----------------------------------------------------------------------------
echo "niah_kv_sweep: commit=${SHORT_COMMIT} ts=${TIMESTAMP}" >&2
echo "  model=${MODEL_PATH}" >&2
echo "  needle=${NEEDLE}" >&2
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

python3 - "${OUT_PATH}" "${SHORT_COMMIT}" "${TIMESTAMP}" "${MODEL_PATH}" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" "${NEEDLE}" "${cells[@]}" <<'PY'
import json, sys
out_path, short_commit, ts, model_path, full_commit, needle, *cells = sys.argv[1:]
doc = {
    "commit": full_commit,
    "short_commit": short_commit,
    "timestamp_utc": ts,
    "model": model_path,
    "needle": needle,
    "cells": [json.loads(c) for c in cells],
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))
PY

echo "" >&2
echo "wrote ${OUT_PATH}" >&2
