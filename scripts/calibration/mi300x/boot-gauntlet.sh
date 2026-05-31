#!/usr/bin/env bash
#
# POD BOOT SCRIPT — paste into the provider's "startup script" field.
#
# Turns a fresh 8×MI300X pod into a fire-and-forget ml8 gauntlet appliance:
#   boot → pull image → S3-sync model bundles → run the dispatcher on all GPUs
#        → ship results back to S3 → power off (stop the $/hr meter).
#
# Override via the provider's env / cloud-init vars:
#   IMAGE          registry/mad-lab-mi300x:v1   (required)
#   S3_BUCKET      s3://mad-lab-mi300x          (model bundles in/results out)
#   GPUS           8                            (saturate the pod)
#   MANIFEST       gauntlet_tier1.json
#   AUTO_SHUTDOWN  1                            (0 to keep the box up for inspection)
#
set -euo pipefail
exec > >(tee -a /var/log/gauntlet-boot.log) 2>&1

: "${IMAGE:?set IMAGE=<registry>/mad-lab-mi300x:v1}"
S3_BUCKET="${S3_BUCKET:-s3://mad-lab-mi300x}"
GPUS="${GPUS:-8}"
MANIFEST="${MANIFEST:-gauntlet_tier1.json}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"

REPO=/opt/mad-lab/llama.cpp                       # image WORKDIR root
MODELS=/models
OUT=/gauntlet-out
STAMP=$(date +%Y%m%d-%H%M%S)

echo "[boot] $(date -Is) gauntlet bring-up  image=$IMAGE gpus=$GPUS"

# 1. image
podman pull "$IMAGE"

# 2. model bundles: S3 → local NVMe (cp, NOT mount — calibration does full-file
#    reads into VRAM; a FUSE mount would be pathologically slow). Bundle = bf16
#    GGUF + HF config/tokenizer (no safetensors; weights load resident from GGUF).
mkdir -p "$MODELS/qwen36-27b" "$MODELS/qwen36-35b-a3b" "$OUT"
s5cmd cp "$S3_BUCKET/qwen36-27b/*"     "$MODELS/qwen36-27b/"
s5cmd cp "$S3_BUCKET/qwen36-35b-a3b/*" "$MODELS/qwen36-35b-a3b/"

# 2b. OVERLAY latest calibration scripts from S3 over the image's baked copy. This
#     is why we never rebuild the image for a Python change: the heavy C++ base is
#     baked once; the fast-moving calibration code (resident-MoE, new manifests,
#     lever tweaks) is pulled fresh here. No-op if the prefix is empty.
SCRIPTS_DST="$REPO/scripts/calibration"
if s5cmd ls "$S3_BUCKET/scripts-latest/" >/dev/null 2>&1; then
    echo "[boot] overlaying latest calibration scripts from S3"
    # Overlay INTO the container's baked tree at run time via a bind mount below;
    # here we stage them on the host so the same -v mount picks them up.
    mkdir -p /scripts-latest
    s5cmd cp "$S3_BUCKET/scripts-latest/*" /scripts-latest/
fi

# 3. dispatcher across all GPUs (resume-aware; rerun continues on a crashed pod)
podman run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --ipc=host \
    -v "$MODELS":/models -v "$OUT":/gauntlet-out \
    "$IMAGE" \
    python3 "$REPO/scripts/calibration/mi300x/run_gauntlet.py" \
        --manifest "$REPO/scripts/calibration/mi300x/$MANIFEST" \
        --gpus "$GPUS" --models-root /models --out-root /gauntlet-out \
  || echo "[boot] dispatcher returned nonzero — results still uploaded below"

# 4. ship everything back (blobs + per-cell manifest.json/Y_SNR + run logs)
s5cmd cp "$OUT/" "$S3_BUCKET/gauntlet-out/$STAMP/"
echo "[boot] $(date -Is) results at $S3_BUCKET/gauntlet-out/$STAMP/"

# 5. stop the meter
if [ "$AUTO_SHUTDOWN" = "1" ]; then
    echo "[boot] powering off to end billing"
    shutdown -h now
fi
