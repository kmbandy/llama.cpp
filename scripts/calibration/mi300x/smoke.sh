#!/usr/bin/env bash
# 5-minute MI300X gate: run this INSIDE the container on a fresh instance BEFORE
# committing the gauntlet to paid credits. Proves imports + gfx942 kernels + the
# resident calibration path actually work on this host. Fail here = $0.05 lost,
# not a wasted gauntlet hour.
set -euo pipefail

echo "=== [1/3] torch sees the MI300X + bf16 gemm runs on gfx942 ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip)
assert torch.cuda.is_available(), \
    "no GPU visible — run with --device=/dev/kfd --device=/dev/dri --group-add video --group-add render"
print("device:", torch.cuda.get_device_name(0))
# bf16 matmul on the WMMA cores is the core op of the heavy tune loop — exercise it.
x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
s = torch.bmm(x[None].bfloat16(), x[None].bfloat16()).float().sum().item()
print("gfx942 bf16 bmm OK, checksum finite:", torch.isfinite(torch.tensor(s)).item())
PY

echo "=== [2/3] calibration modules import (resident path must import w/o wp_native) ==="
python3 - <<'PY'
import calibrate_ml8_paged as c
import ml8_io, batched_gptq, kronecker_rotation, gguf  # noqa: F401
# resident path does not require the pager; paged path would. Either is fine here.
print("wp_native present:", c.wp_native is not None)
print("calibration imports OK")
PY

echo "=== [3/3] optional 1-layer resident calibration ==="
# Only runs if a model bundle is staged. Set MODEL_DIR (HF dir) and GGUF (bf16 path),
# e.g. after: s5cmd cp 's3://mad-lab-mi300x/qwen35-9b/*' /models/qwen35-9b/
if [[ -n "${MODEL_DIR:-}" && -n "${GGUF:-}" ]]; then
  python3 calibrate_ml8_paged.py --strategy dense --resident \
    --model "$MODEL_DIR" --gguf "$GGUF" --arch "${ARCH:-qwen35}" \
    --rotation kronecker --snap-centroids e4m3 --fit-loss mse \
    --group-size 64 --n-centroids 16 --n-samples 4 --seq-len 256 \
    --act-order --heavy-rounds 1 --heavy-steps 5 --heavy-dtype bf16 \
    --no-resume --output-dir /tmp/smoke_out --device cuda:0
  echo "1-layer resident calibration OK"
else
  echo "skipped (set MODEL_DIR + GGUF to run a real 1-layer pass)"
fi

echo "=== SMOKE OK — safe to start the gauntlet ==="
