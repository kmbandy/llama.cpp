# syntax=docker/dockerfile:1
#
# MI300X (gfx942 / CDNA3) calibration image — full mirror of the local resident +
# paged calibration toolchain, so a fresh MI300X instance runs the gauntlet with
# zero on-instance builds and zero version surprises.
#
# Base: AMD's own ROCm 7.2.4 + PyTorch image. This is deliberate:
#   - ROCm 7.2.4 == the local build's ROCm major.minor (HIP 7.2) → our llama.cpp
#     compiles against an identical ROCm, no reconciliation.
#   - gfx942 is a first-class target in this image; AMD validates torch on MI300X here.
#   - torch 2.7.1 covers the entire ~40-op surface our calibration uses (newest op
#     stabilized in torch 1.9), so the local torch-2.13-dev version is irrelevant.
# We add ONLY our own bits: llama.cpp built for gfx942, the wp_native pager, python
# deps (NOT torch), and the calibration scripts.
FROM docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

ARG GFX=gfx942
ARG S5CMD_VERSION=2.2.2
ARG JOBS=
ENV ROCM_PATH=/opt/rocm
SHELL ["/bin/bash", "-lc"]

# --- build tooling for the llama.cpp HIP build ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# --- s5cmd: fast S3 model-bundle pulls (cp to local NVMe, NOT mount) ---
RUN curl -fsSL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
      | tar -xz -C /usr/local/bin s5cmd && s5cmd version

WORKDIR /opt/mad-lab/llama.cpp
# Build context is the repo ROOT (see build-image.sh). .containerignore trims the
# local build trees, model blobs, and datasets so the context stays ~800 MB.
COPY . .

# --- build llama.cpp HIP libs for gfx942 (libllama + ggml-hip + ggml-base + ggml-cpu) ---
# GGML_NATIVE=OFF: the local build bakes -march=native for THIS host CPU; off here
# so the image runs on whatever CPU the MI300X host has.
RUN cmake -S . -B build-hip -G Ninja \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=${GFX} -DGPU_TARGETS=${GFX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ && \
    cmake --build build-hip --target llama -j"${JOBS:-$(nproc)}" && \
    test -e build-hip/bin/libllama.so

# --- build wp_native pager against the gfx942 llama.cpp ---
# Links libllama/ggml-hip/amdhip64 (NOT libtorch) → torch-version-independent.
# Only the PAGED path needs it; --resident imports fine even if this were absent.
RUN python3 -m pip install --no-cache-dir pybind11 && \
    cd python_bindings/wp && python3 setup.py build_ext --inplace && \
    test -e "$(ls wp_native*.so | head -1)"

# --- python deps (torch is already in the base image — do NOT reinstall it) ---
# Install the repo's own gguf-py so the GGUF reader matches our writer exactly.
RUN python3 -m pip install --no-cache-dir -r scripts/calibration/mi300x/requirements-mi300x.txt && \
    python3 -m pip install --no-cache-dir ./gguf-py

ENV PYTHONPATH=/opt/mad-lab/llama.cpp/scripts/calibration:/opt/mad-lab/llama.cpp/python_bindings/wp
ENV LD_LIBRARY_PATH=/opt/mad-lab/llama.cpp/build-hip/bin:/opt/rocm/lib

WORKDIR /opt/mad-lab/llama.cpp/scripts/calibration
CMD ["bash"]
