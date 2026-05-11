#!/usr/bin/env bash
# Army boot script: mad-lab box (1070 Pascal + RX 480 Polaris in ROCm 6.3 docker).
# Customize paths, ports, and docker image name.

set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/GitHub/llama.cpp/build-cuda/bin/llama-server}"
LLAMA_BIN_HIP="${LLAMA_BIN_HIP:-$HOME/GitHub/llama.cpp/build-hip-gfx803/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SSD_PATH="${SSD_PATH:-/var/lib/army/ssd}"
LOG_DIR="${LOG_DIR:-/var/log/army}"
BGE_SMALL="${BGE_SMALL:-$MODELS_DIR/bge-small-en-v1.5-q8_0.gguf}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm-6.3-gfx803:latest}"

mkdir -p "$SSD_PATH" "$LOG_DIR"

# ── Instance 1: 1070 (Pascal CUDA, 8 GB) — Qwen3.5-9B-Omnicoder ───────
INSTANCE=mad-lab-1070
PORT=11437
exec_log="$LOG_DIR/${INSTANCE}.log"

"$LLAMA_BIN" \
    -m "$MODELS_DIR/Qwen3.5-9B-Omnicoder-Q5_K_S.gguf" \
    --device CUDA0 -ngl 99 \
    --parallel 4 -c 524288 \
    --no-mmap \
    --kv-tier-paged-blocks \
    --kv-tiered 25,25,50 \
    --cache-type-k turbo4 --cache-type-v turbo4 \
    --kv-tier-ssd-path "$SSD_PATH" \
    --kv-tier-semantic-index "$BGE_SMALL" \
    --instance-id "$INSTANCE" \
    --kv-tier-cold-budget-mb 8000 \
    --port "$PORT" \
    >> "$exec_log" 2>&1 &
PID_1070=$!
echo "[army-mad-lab] 1070 instance pid=$PID_1070 → $exec_log"

# ── Instance 2: RX 480 (Polaris gfx803 in docker) — Qwen3.5-9B ────────
# Run llama-server inside the gfx803 docker since native ROCm 7.x
# dropped Polaris support. The image is built with ROCm 6.3/6.4.
INSTANCE=mad-lab-rx480
PORT=11438
exec_log="$LOG_DIR/${INSTANCE}.log"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v "$MODELS_DIR:/models:ro" \
    -v "$SSD_PATH:/ssd" \
    -v "$LOG_DIR:/logs" \
    -p $PORT:$PORT \
    "$DOCKER_IMAGE" \
    /opt/llama-server \
        -m /models/Qwen3.5-9B-Q5_K_S.gguf \
        -ngl 99 \
        --parallel 4 -c 524288 \
        --no-mmap \
        --kv-tier-paged-blocks \
        --kv-tiered 25,25,50 \
        --cache-type-k turbo4 --cache-type-v turbo4 \
        --kv-tier-ssd-path /ssd \
        --kv-tier-semantic-index "/models/bge-small-en-v1.5-q8_0.gguf" \
        --instance-id "$INSTANCE" \
        --kv-tier-cold-budget-mb 8000 \
        --port $PORT \
    >> "$exec_log" 2>&1 &
PID_RX480=$!
echo "[army-mad-lab] RX 480 docker pid=$PID_RX480 → $exec_log"

wait $PID_1070 $PID_RX480
