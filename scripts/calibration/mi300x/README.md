# MI300X calibration image

A self-contained container for running ml8 calibration gauntlets on AMD MI300X
(gfx942 / CDNA3) instances — e.g. AMD Dev Cloud at ~$2/hr. Built locally with
podman, pushed to a registry, pulled by each instance. No on-instance builds.

## Why an image (and why these versions)

The worry this design kills: *"spin up an instance, then discover we need rebuilds."*
In a container the host's ROCm/torch versions don't matter — the container carries
its own. We pin everything and verify locally first:

- **Base `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.7.1`** — ROCm
  7.2.4 matches the local build's ROCm (HIP 7.2), gfx942 is first-class, torch 2.7.1
  is AMD-validated on MI300X. torch 2.7.1 covers the entire ~40-op surface our
  calibration uses (newest op stabilized in torch 1.9), so the local torch-2.13-dev
  version is irrelevant to our code.
- **We build only our own bits**: llama.cpp for gfx942, the `wp_native` pager
  (links libllama/ggml-hip, *not* libtorch → torch-version-independent), our scripts,
  and the repo's `gguf-py`. torch/numpy come from the base — never reinstalled.
- ml8 *inference* is gfx1201-only, so PPL eval stays on the R9700; only
  **calibration** runs on the MI300X.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | The image. Build context is the repo root. |
| `requirements-mi300x.txt` | Python deps (NOT torch). The one hard pin is `transformers==5.6.1` (Qwen3.5 VL arch). |
| `build-image.sh` | `podman build` from the repo root + push instructions. |
| `smoke.sh` | 5-min on-instance gate: torch sees the GPU, gfx942 bf16 gemm runs, calibration imports, optional 1-layer pass. Run BEFORE the gauntlet. |
| `verify_torch_surface.py` | $0 local pre-flight: prove the pinned torch+transformers surface in a CPU venv before building. |

## Workflow

```bash
# 0. ($0) prove the API surface against the pinned versions, locally:
python3 -m venv /tmp/v && . /tmp/v/bin/activate
pip install torch==2.7.1 transformers==5.6.1 scipy sentencepiece && pip install ./gguf-py
python3 scripts/calibration/mi300x/verify_torch_surface.py      # -> SURFACE OK

# 1. build the gfx942 image locally (llama.cpp cross-compiles without a gfx942 GPU):
scripts/calibration/mi300x/build-image.sh mad-lab-mi300x:v1

# 2. push to a registry, then on each MI300X instance:
podman pull <registry>/mad-lab-mi300x:v1
podman run --rm -it --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --security-opt seccomp=unconfined \
    -v /mnt/models:/models mad-lab-mi300x:v1 \
    bash scripts/calibration/mi300x/smoke.sh                     # -> SMOKE OK

# 3. stage a model bundle from S3 and run a gauntlet stage:
s5cmd cp 's3://mad-lab-mi300x/qwen35-9b/*' /models/qwen35-9b/
```

## Notes

- The `.so` for gfx942 can only be validated on a real MI300X (we can't run it on the
  R9700/gfx1201), so step 0 + the build prove everything except gfx942 kernel
  execution, and `smoke.sh` closes that last gap for ~$0.05.
- Resident calibration fits every model on 192 GB (35B bf16 = 67 GB), so the pager is
  optional on MI300X; `--resident` is the default path there.
- The base image tag can be bumped (2.8.0 / 2.9.1 also exist for rocm7.2.4) via the
  `FROM` line; 2.7.1 is the conservative pick.
