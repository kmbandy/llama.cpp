#!/usr/bin/env bash
# Build the MI300X (gfx942) calibration image with podman, from the repo root.
#
#   scripts/calibration/mi300x/build-image.sh [image-tag]
#
# llama.cpp cross-compiles for gfx942 without a gfx942 GPU present, so this builds
# fully on the local box. Push the result to a registry; the MI300X just pulls it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:-mad-lab-mi300x:v1}"
DOCKERFILE="scripts/calibration/mi300x/Dockerfile"

# Resource caps (overridable via env). Defaults are tuned to build SAFELY alongside
# a running calibration on a 15 GB host: the build runs in a memory-capped cgroup so
# its compiles can never exhaust host RAM and trip the host OOM-killer (which would
# take the calibration). JOBS bounds clang concurrency to fit that cgroup.
JOBS="${JOBS:-3}"
MEM="${MEM:-7g}"

# podman streams the multi-GB base image through TMPDIR during pull. Default /tmp
# is often tmpfs (RAM-backed) or a small partition — a 30 GB base image overflows
# it AND pressures host RAM. Force temp onto the big /home disk.
export TMPDIR="${PODMAN_TMPDIR:-$HOME/.cache/podman-build-tmp}"
mkdir -p "$TMPDIR"
echo "[build] TMPDIR = $TMPDIR ($(df -h "$TMPDIR" | awk 'NR==2{print $4}') free)"

cd "$REPO_ROOT"
[ -f "$DOCKERFILE" ] || { echo "missing $DOCKERFILE (run from the repo)"; exit 1; }

echo "[build] context = $REPO_ROOT"
echo "[build] image   = $IMAGE"
echo "[build] caps    = JOBS=$JOBS  --memory=$MEM (cgroup-isolated from the host run)"
echo "[build] context size (after .containerignore):"
du -sh --exclude=.git --exclude='build*' --exclude=models --exclude=wikitext-2-raw \
       --exclude=bin --exclude='*.gguf' . 2>/dev/null | tail -1 || true

# --memory == --memory-swap → no swap, so a runaway compile can't swap the host to death.
podman build \
    --memory="$MEM" --memory-swap="$MEM" \
    --build-arg JOBS="$JOBS" \
    -f "$DOCKERFILE" -t "$IMAGE" .

echo
echo "[build] done: $IMAGE"
echo "Next:"
echo "  podman push $IMAGE <registry>/$IMAGE       # e.g. ghcr.io / a private registry"
echo "  # on the MI300X instance:"
echo "  podman pull <registry>/$IMAGE"
echo "  podman run --rm -it --device=/dev/kfd --device=/dev/dri \\"
echo "      --group-add video --group-add render --security-opt seccomp=unconfined \\"
echo "      -v /mnt/models:/models $IMAGE bash scripts/calibration/mi300x/smoke.sh"
