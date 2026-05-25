"""Build the wp_native pybind11 module.

Usage:
    cd python_bindings/wp
    python3 setup.py build_ext --inplace

Produces wp_native*.so in this directory, importable as `wp_native`.

Links against the libllama.so built at <repo_root>/build-hip/bin/.
"""
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_HIP_BIN = REPO_ROOT / "build-hip" / "bin"

assert REPO_ROOT.is_dir(), REPO_ROOT
assert BUILD_HIP_BIN.is_dir(), f"missing build-hip/bin — did you cmake --build build-hip? ({BUILD_HIP_BIN})"
assert (BUILD_HIP_BIN / "libllama.so").exists() or (BUILD_HIP_BIN / "libllama.so.0").exists(), \
    f"missing libllama.so in {BUILD_HIP_BIN}"

ROCM_INCLUDE = "/opt/rocm/include"
ROCM_LIB = "/opt/rocm/lib"

ext = Pybind11Extension(
    "wp_native",
    sources=["wp_bindings.cpp"],
    include_dirs=[
        str(REPO_ROOT),                       # for "src/weight-pager/..." headers
        str(REPO_ROOT / "ggml" / "include"),  # for forward decls of ggml types
        ROCM_INCLUDE,                          # hip/hip_runtime.h
    ],
    library_dirs=[str(BUILD_HIP_BIN), ROCM_LIB],
    libraries=["llama", "ggml-hip", "ggml-base", "amdhip64"],
    runtime_library_dirs=[str(BUILD_HIP_BIN), ROCM_LIB],
    cxx_std=17,
    extra_compile_args=[
        "-O2", "-Wall",
        "-D__HIP_PLATFORM_AMD__=1",  # required for hip_runtime.h
    ],
)

setup(
    name="wp_native",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
