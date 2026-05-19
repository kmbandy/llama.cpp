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

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
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

# AITER Triton AOT integration (MAD-188). Off by default; opt in with
# -DGGML_HIP_AITER=ON. Adds AOT-compiled AITER unified_attention kernels plus
# the mt_aiter_unified_attn wrapper to ggml-hip. Source code dispatches on the
# GGML_HIP_AITER compile definition; the runtime call site is further gated by
# the MAD_USE_AITER env var so we can A/B against the existing tile kernel
# without a rebuild.
option(GGML_HIP_AITER "Enable AITER Triton AOT attention kernels (MAD-188)" OFF)
if (GGML_HIP_AITER)
    add_subdirectory(../ggml-cuda/aiter-integration aiter-integration)
    target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)
    target_compile_definitions(ggml-hip PRIVATE GGML_HIP_AITER)
endif()
