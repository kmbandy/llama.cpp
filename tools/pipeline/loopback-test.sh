#!/usr/bin/env bash
# loopback-test.sh -- Phase 2 pipeline correctness gate.
#
# THIS SCRIPT IS WRITTEN, NOT RUN, by the implementing agent. A human executes
# it on mad-lab-main once the tree builds. Do not run it from an agent.
#
# Given a model and a split point K, this:
#   1. splits the model into head [0, K-1] and tail [K, n_layer-1] stage GGUFs
#      with tools/wp-stage-split,
#   2. runs the two stage processes on 127.0.0.1 (different ports),
#   3. runs the SAME model as a single process on CPU,
#   4. asserts an identical greedy token sequence for N tokens at a fixed seed.
#
# Greedy + fixed seed makes this exact, not statistical. The first generated
# token is the greedy argmax of the first-token prompt logits, so an identical
# token sequence implies the prompt-logit argmax matched at the boundary; the
# full sequence then checks every subsequent token. If the outputs differ,
# that is the finding: report it as a failure. Do NOT soften this to a
# similarity threshold or a perplexity comparison.
#
# Requirements (already built by a human, CPU-only build):
#   build/bin/llama-wp-stage-split
#   build/bin/llama-pipeline
#   build/bin/llama-cli            (the single-process reference)
#
# Usage:
#   tools/pipeline/loopback-test.sh <model.gguf> <K> [n_tokens]
#
# Suggested models (run weight paging OFF first to isolate the protocol):
#   ~/models/E2Rank-0.6B.Q8_0.gguf        (dense, fast gate)
#   ~/models/LFM2.5-8B-A1B-Q8_0.gguf      (MoE, covers the expert path)

set -euo pipefail

MODEL="${1:?usage: $0 <model.gguf> <K> [n_tokens]}"
K="${2:?usage: $0 <model.gguf> <K> [n_tokens]}"
N_TOKENS="${3:-32}"

BIN="${BIN:-build/bin}"
SPLIT="${BIN}/llama-wp-stage-split"
PIPE="${BIN}/llama-pipeline"
CLI="${BIN}/llama-cli"

WORK="$(mktemp -d /tmp/pipe-loopback.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

HEAD_GGUF="${WORK}/head.gguf"
TAIL_GGUF="${WORK}/tail.gguf"

HOST="127.0.0.1"
TAIL_PORT=9911

PROMPT="${PROMPT:-The meaning of life is}"
SEED=0

echo "== loopback: model=${MODEL} K=${K} n=${N_TOKENS} work=${WORK} =="

# ---------------------------------------------------------------------------
# 0. discover n_layer so the tail's --last is exactly n_layer-1.
#    wp-stage-split --dry-run prints "layers [F, L] of N (...)" on stderr.
N_LAYER="$("${SPLIT}" --model "${MODEL}" --first 0 --last 0 --dry-run 2>&1 \
            | sed -n 's/.* of \([0-9][0-9]*\) .*/\1/p' | head -n1)"
if [[ -z "${N_LAYER}" ]]; then
    echo "FAIL: could not read n_layer from ${MODEL} via wp-stage-split" >&2
    exit 1
fi
LAST=$((N_LAYER - 1))
if (( K < 1 || K > LAST )); then
    echo "FAIL: split point K=${K} out of range for n_layer=${N_LAYER}" >&2
    exit 1
fi
echo "== n_layer=${N_LAYER}: head [0,$((K-1))] tail [${K},${LAST}] =="

# ---------------------------------------------------------------------------
# 1. split into stage GGUFs
"${SPLIT}" --model "${MODEL}" --out "${HEAD_GGUF}" --first 0      --last $((K - 1))
"${SPLIT}" --model "${MODEL}" --out "${TAIL_GGUF}" --first "${K}" --last "${LAST}"

# ---------------------------------------------------------------------------
# CPU-only, weight paging OFF, greedy, fixed seed. --no-warmup on the tail is
# mandatory (a token warmup on a stage without token_embd fails by design).
# --log-disable/--no-display-prompt/--simple-io keep stdout to ONLY the
# generated continuation, so the diff below compares generation, not logs.
COMMON_CPU=(-t 4 -ngl 0)
GREEDY=(--seed "${SEED}" --temp 0)
# IO-cleaning flags. -no-cnv is llama-cli-only (the pipeline tool never enters
# conversation mode and its arg parser would reject the flag).
CLEAN_IO=(--log-disable --no-display-prompt --simple-io)
CLI_IO=(-no-cnv)

# ---------------------------------------------------------------------------
# 2. single-process reference: greedy token sequence
echo "== reference (single process) =="
"${CLI}" -m "${MODEL}" "${COMMON_CPU[@]}" "${GREEDY[@]}" "${CLEAN_IO[@]}" "${CLI_IO[@]}" \
    -p "${PROMPT}" -n "${N_TOKENS}" \
    >"${WORK}/ref.txt" 2>"${WORK}/ref.err"

# ---------------------------------------------------------------------------
# 3. pipeline: start the tail, then the head driver
echo "== pipeline (2 stages on ${HOST}) =="
# NOTE: the tail is NOT given CLEAN_IO so its "listening" log line appears and
# the readiness wait below is fast; only the head and reference stdout are diffed.
"${PIPE}" -m "${TAIL_GGUF}" "${COMMON_CPU[@]}" "${GREEDY[@]}" \
    --pipeline-listen "${HOST}:${TAIL_PORT}" --no-warmup \
    >"${WORK}/tail.out" 2>"${WORK}/tail.err" &
TAIL_PID=$!
# wait for the tail to finish its HELLO-less listen (it logs "listening")
for _ in $(seq 1 100); do
    if grep -q "listening" "${WORK}/tail.err" "${WORK}/tail.out" 2>/dev/null; then break; fi
    sleep 0.2
done

"${PIPE}" -m "${HEAD_GGUF}" "${COMMON_CPU[@]}" "${GREEDY[@]}" "${CLEAN_IO[@]}" \
    --pipeline-peer "${HOST}:${TAIL_PORT}" \
    -p "${PROMPT}" -n "${N_TOKENS}" \
    >"${WORK}/pipe.txt" 2>"${WORK}/pipe.err" || true
# the head's canonical greedy token-id stream (ground truth) is on stderr
grep '^PIPELINE-TOKENS:' "${WORK}/pipe.err" >"${WORK}/pipe.tokens" || true

kill "${TAIL_PID}" 2>/dev/null || true
wait "${TAIL_PID}" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. assert identical greedy token sequences (exact, not statistical)
#
# Both ref.txt and pipe.txt should contain ONLY the generated continuation
# (CLEAN_IO strips prompt echo and logs). If a llama-cli/log routing change
# ever lets load logs leak into stdout, the diff will show them; in that case
# confirm whether the generated block is identical (formatting artifact) and,
# if it is not, that is a REAL divergence -- report it. The head's canonical
# token-id stream is in pipe.tokens for ground truth.
echo "== comparing =="
if [[ ! -s "${WORK}/ref.txt" ]]; then
    echo "WARN: reference stdout is empty. llama-cli may route generation through" >&2
    echo "  the log system that --log-disable paused. Re-run the reference WITHOUT" >&2
    echo "  --log-disable and diff its generated block against pipe.txt manually." >&2
    echo "  Head token stream (ground truth): ${WORK}/pipe.tokens" >&2
    trap - EXIT
    exit 2
fi
if diff -u "${WORK}/ref.txt" "${WORK}/pipe.txt"; then
    echo "PASS: pipeline greedy output matches single-process for ${N_TOKENS} tokens"
    exit 0
else
    echo "FAIL: pipeline output diverges from single-process reference" >&2
    echo "  kept for inspection (not cleaned on failure):" >&2
    echo "    ref:  ${WORK}/ref.txt" >&2
    echo "    pipe: ${WORK}/pipe.txt" >&2
    echo "    tail: ${WORK}/tail.err" >&2
    echo "    head: ${WORK}/pipe.err" >&2
    trap - EXIT   # keep $WORK for post-mortem
    echo "This is the correctness gate failing -- report it, do not tune it away." >&2
    exit 1
fi
