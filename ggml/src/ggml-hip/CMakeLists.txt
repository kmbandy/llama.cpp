if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

# Guard: HIP host code MUST be compiled with the ROCm-bundled clang++. Building
# with GCC silently produces a binary that runs ~40% slower on the host hot path
# (kernel dispatch, batch prep, server loop). See the 2026-05-18 build-config
# drift incident in mad-lab notes.
if (NOT CXX_IS_HIPCC)
    if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(FATAL_ERROR
            "GGML_HIP=ON requires a Clang C++ compiler (got '${CMAKE_CXX_COMPILER_ID}' "
            "at '${CMAKE_CXX_COMPILER}'). Reconfigure with "
            "-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++.")
    endif()
    get_filename_component(_ggml_hip_cxx_real "${CMAKE_CXX_COMPILER}" REALPATH)
    string(FIND "${_ggml_hip_cxx_real}" "${ROCM_PATH}" _ggml_hip_cxx_match)
    if (_ggml_hip_cxx_match EQUAL -1)
        message(FATAL_ERROR
            "GGML_HIP=ON requires the ROCm-bundled clang++; got "
            "'${CMAKE_CXX_COMPILER}' (resolved '${_ggml_hip_cxx_real}') which is "
            "not under ROCM_PATH '${ROCM_PATH}'. Reconfigure with "
            "-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++.")
    endif()
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
# MAD-305 Phase 5: hand-written gfx1201 WMMA "trfeed" block-scale GEMM for
# GGML_TYPE_ML8_FP8 (rdna4_fp8_gemm/RESULT.md). Not picked up by the *.cu glob
# above (lives under aiter-integration/rdna4_fp8_gemm/ with a .hip extension,
# matching this repo's convention for the other hand-written kernels in that
# dir). Compiled unconditionally -- it has no Triton/AITER dependency of its
# own (only ml8.cu's USE of the GEMM dispatch is gated on GGML_HIP_AITER, to
# match the existing FP8_B128 dispatch shape); its production kernel body is
# internally guarded with `#if defined(__gfx1201__)` so other offload-arch
# device-code passes in the same fat binary (e.g. gfx1030) compile it down to
# an inert stub instead of failing on an unknown gfx12 matrix-core intrinsic.
list(APPEND GGML_SOURCES_ROCM
    ../ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_blockscale.hip)
# MAD-305 Phase 5 (round 3): production entry point for the FROZEN Phase-1
# "trfeed" kernel (rdna4_gemm_fp8_trfeed, gemm_capi.h) -- the kernel GGML_OP_
# FP8_MUL_MAT's HIP dispatch (ml8.cu) now uses by default; gemm_blockscale.hip
# above stays in the build (its rdna4_[un]preshuffle_b_ml8fp8 byte-shuffle
# helpers are still reused here) but its two in-loop-scale-fold GEMM kernels
# are no longer called from ml8.cu's default dispatch (abandoned, RESULT.md
# Phase 2/gemm_capi.h). Same gfx1201-only / other-arch-stub guard convention.
list(APPEND GGML_SOURCES_ROCM
    ../ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_trfeed_prod.hip)
# MAD-305 ML8_4 (4.5 bpw) decode + prefill entry points (gemm_capi.h's
# "ML8_4 decode + prefill" section) -- now the DEFAULT dispatch for dense
# GGML_TYPE_ML8_4 weights on gfx1201 (ml8.cu's ML8_4_LAYOUT_RDNA4_TRFEED,
# MT_ML8_4_LAYOUT=triton opts back into the old Triton path). Same
# gfx1201-only / other-arch-stub guard convention as gemm_trfeed_prod.hip
# above; reuses the UNCHANGED frozen fp8 trfeed kernel (gemm_trfeed_prod.hip,
# already in this list) for its prefill path via rdna4_expand_ml84_to_trfeed.
list(APPEND GGML_SOURCES_ROCM
    ../ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_ml84_prod.hip)

# R4D vendored attention kernels (ggml-cuda/r4d/, see r4d/README-ORIGIN.txt for provenance --
# radiance-libr4d commit b9e42ab-rx6). fp8-KV and bf16-KV paged, causal, GQA-6 attention at
# head_dim 256: r4d_attn_{prefill,decode}_h256_gqa6_{fp8kv,bf16kv} plus
# r4d_attn_decode_h256_gqa6_scratch_bytes / r4d_attn_dims. Reachable from other ggml-cuda
# translation units through ggml-cuda/r4d/ggml-r4d.h, which also declares
# ggml_cuda_r4d_available() (true only when this is compiled in AND the current device is
# gfx1201). Compiled ONLY for gfx1201, both because the vendored kernels are hand-tuned for
# exactly that WMMA ISA (they reject anything else at the call site, but there is no reason to
# pay the compile time or fat-binary size for an arch that can never dispatch into them) and
# because gfx1201 is the only arch this vendoring has been validated against.
#
# Default ON iff gfx1201 is one of the requested offload archs (AMDGPU_TARGETS is what the
# CMAKE_HIP_ARCHITECTURES forwarding above reads; GPU_TARGETS is its CMake-native alias -- check
# both, matching the forwarding logic itself). GGML_HIP_R4D can still be forced OFF explicitly, or
# forced ON, but forcing it ON without gfx1201 in the arch list is refused below: r4d_attn_paged_
# h256_gqa6.hip has no other-arch stub (unlike the rdna4_fp8_gemm units above, whose production
# kernel bodies are individually `#if defined(__gfx1201__)`-guarded) and would either fail to
# build or build dead code for an offload target gfx1201-only WMMA intrinsics don't exist on.
set(_r4d_targets "${AMDGPU_TARGETS}${GPU_TARGETS}")
if (_r4d_targets MATCHES "gfx1201")
    set(GGML_HIP_R4D_DEFAULT ON)
else()
    set(GGML_HIP_R4D_DEFAULT OFF)
endif()
option(GGML_HIP_R4D "Enable vendored R4D fp8/bf16-KV paged attention kernels (gfx1201 only)" ${GGML_HIP_R4D_DEFAULT})
if (GGML_HIP_R4D)
    if (NOT _r4d_targets MATCHES "gfx1201")
        message(FATAL_ERROR
            "GGML_HIP_R4D=ON but gfx1201 is not in AMDGPU_TARGETS/GPU_TARGETS "
            "('${AMDGPU_TARGETS}' / '${GPU_TARGETS}'). The vendored R4D kernels are compiled "
            "only for gfx1201 and have no stub for other architectures. Add gfx1201 to the "
            "offload-arch list or reconfigure with -DGGML_HIP_R4D=OFF.")
    endif()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/r4d/r4d_attn_paged_h256_gqa6.hip)
    add_compile_definitions(GGML_HIP_R4D)
    # Upstream (r4d/build.sh) compiles this unit with:
    #   -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off
    # -O3 / -std=c++17 / -fPIC / --offload-arch already match what the rest of ggml-hip's build
    # uses (CMAKE_BUILD_TYPE=Release, CMAKE_CXX_STANDARD 17 set in ggml/CMakeLists.txt, shared
    # libs implying -fPIC, CMAKE_HIP_ARCHITECTURES from AMDGPU_TARGETS) so nothing extra is
    # needed for those. -Wno-unused-result is harmless and applied below for parity.
    # -ffp-contract=off is NOT applied: it exists upstream only for the quantized all-reduce unit
    # (r4d_ar_oneshot_2rank_wht6.hip, not vendored here) whose correctness depends on FMA NOT
    # merging two independently-scaled products; the attention kernels have no such site, so
    # applying it here would just forgo FMA contraction on this TU's ordinary float math for no
    # reason. Left off to match the rest of ggml-hip, which does not set it either.
    set_source_files_properties(../ggml-cuda/r4d/r4d_attn_paged_h256_gqa6.hip
        PROPERTIES COMPILE_OPTIONS "-Wno-unused-result")
endif()

file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
# dkq640 tile kernels exceed HIP local memory limit (67584 > 65536) at ncols=32
list(FILTER SRCS EXCLUDE REGEX "dkq640")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo4_0.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# gemm_ml84_prod.hip (transitively, via bench/trfeed_ml84_kernels.h) includes
# bench/ml84_trfeed_layout.h, whose own `#include "../../../ggml-common.h"`
# resolves relative to ITS directory (rdna4_fp8_gemm/bench/), which only
# reaches ggml/src/ggml-common.h through the compiler's include search path,
# not the literal 3-dot relative path (that lands one directory short, in
# ggml-cuda/, which has no ggml-common.h of its own). ml8.cu ALSO now
# includes this same header directly (for the ML8_4 in-place unpack path),
# so both need this on the include path -- the standalone bench/build.sh has
# always passed the identical `-I"$HERE"` (this same directory) for exactly
# this reason.
target_include_directories(ggml-hip PRIVATE
    ../ggml-cuda/aiter-integration/rdna4_fp8_gemm)

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

# NOTE(fork): our -ffast-math was dropped here in the 2026-07-31 upstream sync.
# Upstream now sets -funsafe-math-optimizations above instead, deliberately:
# -ffast-math implies -ffinite-math-only, which breaks ggml's INFINITY masking
# and produces NaNs. Do not reinstate it.

# AITER Triton AOT integration (MAD-188). Off by default; opt in with
# -DGGML_HIP_AITER=ON. Adds AOT-compiled AITER unified_attention kernels plus
# the mt_aiter_unified_attn wrapper to ggml-hip. Source code dispatches on the
# GGML_HIP_AITER compile definition; the runtime call site is further gated by
# the MAD_USE_AITER env var so we can A/B against the existing tile kernel
# without a rebuild.
option(GGML_HIP_AITER "Enable AITER Triton AOT attention kernels (MAD-188)" OFF)
if (GGML_HIP_AITER)
    add_subdirectory(../ggml-cuda/aiter-integration aiter-integration)
    # The aiter-integration subdir only creates the aiter_triton_aot target when
    # Triton AOT is actually available (Python can import triton). If Triton is
    # unavailable it silently skips, leaving no target — and an unconditional
    # `target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)` then degrades to
    # a bare `-laiter_triton_aot` with no search path, which only links if the
    # build dir happens to be on LIBRARY_PATH (the old env-var workaround) and
    # otherwise fails at link time with a confusing error. Guard on the target
    # actually existing, and fail loud at configure with an actionable message
    # otherwise — ml8 GPU inference REQUIRES these kernels (the ml8 ops are
    # #ifdef GGML_HIP_AITER), so a silent AITER-less build would abort at runtime.
    if (TARGET aiter_triton_aot)
        target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)
        # Out-of-tree STATIC target: pin the link search path to the target's own
        # output dir so it resolves with no LIBRARY_PATH env dance, any generator/arch.
        target_link_directories(ggml-hip PRIVATE $<TARGET_FILE_DIR:aiter_triton_aot>)
        target_compile_definitions(ggml-hip PRIVATE GGML_HIP_AITER)
        # The aiter_triton_aot target exposes wrappers/ as PUBLIC includes, but
        # under multi-arch HIP cross-compilation the propagation through PRIVATE
        # linkage does not always reach every per-arch compile pass (gfx1030
        # specifically). Add the include path explicitly so mt_pagedattn_aiter.cu
        # — and any other .cu in ggml-cuda/ that consumes wrappers/ headers —
        # builds for every requested AMDGPU_TARGETS arch.
        target_include_directories(ggml-hip PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/aiter-integration/wrappers)
    else()
        message(FATAL_ERROR
            "GGML_HIP_AITER=ON but the aiter_triton_aot target was not created. "
            "The configure log above shows Triton AOT is unavailable (\"cannot "
            "import triton\"). ml8 GPU inference REQUIRES the AITER kernels (the "
            "ml8 ops are #ifdef GGML_HIP_AITER and abort otherwise), so this would "
            "build a broken library. Fix Triton availability — point the build at a "
            "Python that has triton, e.g. -DPython3_EXECUTABLE=/path/to/venv/bin/python "
            "— or, to build an ml8-less HIP backend on purpose, reconfigure with "
            "-DGGML_HIP_AITER=OFF.")
    endif()
endif()
