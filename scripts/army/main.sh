#!/usr/bin/env bash
# Army boot script: main box (R9700 + 6900XT).
# Customize the model paths, ports, and any per-machine specifics.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────
LLAMA_BIN="${LLAMA_BIN:-$HOME/GitHub/llama.cpp/build-hip/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SSD_PATH="${SSD_PATH:-/var/lib/army/ssd}"
LOG_DIR="${LOG_DIR:-/var/log/army}"
BGE_SMALL="${BGE_SMALL:-$MODELS_DIR/bge-small-en-v1.5-q8_0.gguf}"

mkdir -p "$SSD_PATH" "$LOG_DIR"

# ── Wait for GPU + network ────────────────────────────────────────────
# rocm-smi is a quick health check; bail early if HIP isn't up yet.
until rocm-smi >/dev/null 2>&1; do sleep 2; done

# ── Instance 1: R9700 (32 GB RDNA4) — Qwen3.6-27B ─────────────────────
INSTANCE=main-r9700
PORT=11435
exec_log="$LOG_DIR/${INSTANCE}.log"

"$LLAMA_BIN" \
    -m "$MODELS_DIR/Qwen3.6-27B-Q6_K.gguf" \
    --device ROCm0 -ngl 99 \
    --parallel 4 -c 524288 \
    --no-mmap \
    --kv-tier-paged-blocks \
    --kv-tiered 25,75,0 \
    --cache-type-k turbo4 --cache-type-v turbo4 \
    --kv-tier-ssd-path "$SSD_PATH" \
    --kv-tier-semantic-index "$BGE_SMALL" \
    --instance-id "$INSTANCE" \
    --kv-tier-cold-budget-mb 10000 \
    --port "$PORT" \
    >> "$exec_log" 2>&1 &
PID_R9700=$!
echo "[army-main] R9700 instance pid=$PID_R9700 → $exec_log"

# ── Instance 2: 6900XT (16 GB RDNA2 eGPU) — gpt-oss-20B ───────────────
INSTANCE=main-6900xt
PORT=11436
exec_log="$LOG_DIR/${INSTANCE}.log"

"$LLAMA_BIN" \
    -m "$MODELS_DIR/gpt-oss-20B-MXFP4.gguf" \
    --device ROCm1 -ngl 99 \
    --parallel 4 -c 262144 \
    --no-mmap \
    --kv-tier-paged-blocks \
    --kv-tiered 30,70,0 \
    --kv-tier-ssd-path "$SSD_PATH" \
    --kv-tier-semantic-index "$BGE_SMALL" \
    --instance-id "$INSTANCE" \
    --kv-tier-cold-budget-mb 5000 \
    --port "$PORT" \
    >> "$exec_log" 2>&1 &
PID_6900=$!
echo "[army-main] 6900XT instance pid=$PID_6900 → $exec_log"

# Wait for both. systemd will SIGTERM us on shutdown; both children
# get the signal and clean up their lockfiles via the dtor.
wait $PID_R9700 $PID_6900
