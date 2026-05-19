# aiter-integration/CMakeLists.txt
#
# Triton AOT integration scaffolding for AITER (ROCm/aiter) + vLLM TurboQuant
# kernels. See docs/aiter-integration/ARCHITECTURE.md for the strategic
# context.
#
# STATUS: DRAFT — this CMakeLists is NOT included from the parent build yet.
# That happens after MAD-187 milestone 1 (vector-add POC) demonstrates
# the toolchain end-to-end on gfx1201. To include manually for development:
#
#     # in ggml/src/ggml-cuda/CMakeLists.txt, after the main HIP setup:
#     option(LLAMA_AITER_TRITON "Enable AITER + vLLM Triton AOT kernel path" OFF)
#     if(LLAMA_AITER_TRITON)
#         add_subdirectory(aiter-integration)
#     endif()
#
# Then configure with -DLLAMA_AITER_TRITON=ON.

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(TritonAOT)

if(NOT TRITON_AOT_AVAILABLE)
    message(STATUS "aiter-integration: Triton AOT not available, skipping integration build")
    return()
endif()

# ────────────────────────────────────────────────────────────────────────
# Target arches to AOT-compile for. Default: whatever the user gave
# GPU_TARGETS / AMDGPU_TARGETS / similar. Override via -DAITER_AOT_ARCHES=...
# ────────────────────────────────────────────────────────────────────────

if(NOT AITER_AOT_ARCHES)
    if(GPU_TARGETS)
        set(AITER_AOT_ARCHES ${GPU_TARGETS})
    elseif(AMDGPU_TARGETS)
        set(AITER_AOT_ARCHES ${AMDGPU_TARGETS})
    else()
        # Conservative default: just gfx1201 (R9700). Users with other
        # hardware can override.
        set(AITER_AOT_ARCHES gfx1201)
    endif()
endif()

message(STATUS "aiter-integration: AOT target arches = ${AITER_AOT_ARCHES}")

# ────────────────────────────────────────────────────────────────────────
# Output directory for all generated kernel artifacts.
# Lives under the CMake binary dir, NOT in the source tree — so a clean
# rebuild starts fresh.
# ────────────────────────────────────────────────────────────────────────

set(AITER_AOT_OUT_DIR ${CMAKE_BINARY_DIR}/aiter-integration/triton-out)

# ────────────────────────────────────────────────────────────────────────
# vector_add — proof-of-concept kernel (MAD-187 milestone 1).
# Compile one specialization per target arch. All N hardware archs share
# the same Triton signature; only the AOT --target string differs.
# ────────────────────────────────────────────────────────────────────────

set(_vector_add_specs)
foreach(arch ${AITER_AOT_ARCHES})
    list(APPEND _vector_add_specs
        SIGNATURE  "*fp32:16, *fp32:16, *fp32:16, i32, 1024"
        # C-syntax grid required: Triton inlines this verbatim into the
        # generated C launcher (no Python→C translation). `//` would
        # become a C line comment. Verified 2026-05-17.
        GRID       "n_elements / 1024, 1, 1"
        NUM_WARPS  4
        NUM_STAGES 1
        ARCH       ${arch}
        # No runtime fp32 scalars — the only fp32 in this signature is
        # the constexpr `1024` (BLOCK_SIZE), which is baked at AOT.
        FP32_SCALARS NONE
    )
endforeach()

add_triton_aot_kernel(
    NAME         vector_add
    SOURCE       kernels/vector_add.py
    KERNEL_NAME  add_kernel
    OUT_DIR      ${AITER_AOT_OUT_DIR}
    SPECS        ${_vector_add_specs}
)

# ────────────────────────────────────────────────────────────────────────
# unified_attention 3D + reduce_segments (MAD-188).
#
# Vendored from ROCm/aiter — see kernels/unified_attention.py header for
# upstream commit + patch info.
#
# Shape: Qwen3.6 (num_query_heads=16, num_queries_per_kv=8, head_size=128,
# block_size=16). All bool/feature constexprs disabled (no alibi, no qq_bias,
# no softcap, no sinks, no sliding-window). ALL_DECODE=1 (q_len=1 per seq).
# These match the validated launcher hashes from 2026-05-17:
#   3D     gfx1201: uattn_3d.11808415_0d3d4d5d
#   reduce gfx1201: uattn_reduce.ecdee0c8_0d1d
#
# Grid expressions are C-syntax (Triton inlines them verbatim).
# ────────────────────────────────────────────────────────────────────────

set(_uattn_3d_signature
    "*fp32:16, *fp32, *fp32, *fp16:16, *fp16:16, *fp16:16, *fp32, *i32, *i32, *fp32, *fp16, fp32, *fp32, *fp32, *fp32, fp32, 16, 8, i64, i64, 128, i64, 16, 32, 128, 128, 0, 0, 0, 0, 0, i64, i64, i64, 1, i64, i64, i64, 1, *i32, 2, i32, 16, 32, 1"
)
set(_uattn_3d_grid       "num_seqs / 2 + num_seqs, 2, 32")

set(_uattn_reduce_signature
    "*fp16:16, *fp32:16, *fp32, *fp32, *i32, i32, 16, *fp32, i64, 128, i64, 32, 128, 128, *i32, 2, 32, -448.0, 448.0"
)
set(_uattn_reduce_grid   "num_seqs, 16, 1")

set(_uattn_3d_specs)
set(_uattn_reduce_specs)
foreach(arch ${AITER_AOT_ARCHES})
    list(APPEND _uattn_3d_specs
        SIGNATURE    "${_uattn_3d_signature}"
        GRID         "${_uattn_3d_grid}"
        NUM_WARPS    4
        NUM_STAGES   1
        ARCH         ${arch}
        # scale + softcap are runtime fp32 scalars in the 3D kernel sig —
        # both must be patched per MAD-188 (Triton AOT emits `double`).
        FP32_SCALARS "scale,softcap"
    )
    list(APPEND _uattn_reduce_specs
        SIGNATURE    "${_uattn_reduce_signature}"
        GRID         "${_uattn_reduce_grid}"
        NUM_WARPS    4
        NUM_STAGES   1
        ARCH         ${arch}
        # No runtime fp32 scalars — FP8_MIN/MAX are tl.constexpr (baked at AOT).
        FP32_SCALARS NONE
    )
endforeach()

add_triton_aot_kernel(
    NAME         uattn_3d
    SOURCE       kernels/unified_attention.py
    KERNEL_NAME  kernel_unified_attention_3d
    OUT_DIR      ${AITER_AOT_OUT_DIR}
    SPECS        ${_uattn_3d_specs}
)

add_triton_aot_kernel(
    NAME         uattn_reduce
    SOURCE       kernels/unified_attention.py
    KERNEL_NAME  reduce_segments
    OUT_DIR      ${AITER_AOT_OUT_DIR}
    SPECS        ${_uattn_reduce_specs}
)

# ────────────────────────────────────────────────────────────────────────
# Pull the generated .c files into a static library. The HEADERS need to
# be discoverable too via include path.
# ────────────────────────────────────────────────────────────────────────

# {kernel}_AOT_C_FILES was set in PARENT_SCOPE by add_triton_aot_kernel.
# At configure time these are empty (no build has run); at re-configure
# after build, they'll be populated via CONFIGURE_DEPENDS glob.

add_library(aiter_triton_aot STATIC
    wrappers/vector_add_wrapper.cpp
    wrappers/mt_aiter_unified_attn.cpp
    wrappers/aiter_runtime_compiler.cpp
    # Generated sources added at build time via the AOT_C_FILES vars:
    ${vector_add_AOT_C_FILES}
    ${uattn_3d_AOT_C_FILES}
    ${uattn_reduce_AOT_C_FILES}
)

# This static lib gets linked into libggml-hip.so (a SHARED library). Without
# PIC, ld.lld rejects with `R_X86_64_PC32 cannot be used against symbol …;
# recompile with -fPIC`. GNU ld + GCC silently let this slide; ld.lld is
# strict and correct to enforce it.
set_target_properties(aiter_triton_aot PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Surface the path to the runtime-compile helper as a compile definition so
# the registry's default doesn't require an env-var dance just to run tests.
target_compile_definitions(aiter_triton_aot PUBLIC
    AITER_COMPILE_SCRIPT_DEFAULT=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_aiter_kernel.py\"
    AITER_KERNEL_SOURCE_DEFAULT=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/unified_attention.py\"
)

# Ensure the AOT custom_targets run before the library/wrapper compiles —
# the wrapper #includes the generated headers, so they must exist on disk
# by the time the C++ compiler reads the .cpp.
add_dependencies(aiter_triton_aot
    triton_aot_vector_add
    triton_aot_uattn_3d
    triton_aot_uattn_reduce
)

target_include_directories(aiter_triton_aot
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/wrappers
        # Each per-arch output dir contains the .h for that specialization.
        # Multiple archs may produce same-named headers — we trust the
        # link-stage merger or namespace via subdir.
)

foreach(arch ${AITER_AOT_ARCHES})
    target_include_directories(aiter_triton_aot PUBLIC
        ${AITER_AOT_OUT_DIR}/vector_add/${arch}
        ${AITER_AOT_OUT_DIR}/uattn_3d/${arch}
        ${AITER_AOT_OUT_DIR}/uattn_reduce/${arch}
    )
endforeach()

# Triton's generated .c files use HIP's runtime API (hipModuleLoad,
# hipModuleLaunchKernel). When this subdir is built as part of ggml-hip,
# `hip::host` (from find_package(hip)) is already available and carries the
# right include dirs + libamdhip64 linkage. When built standalone (for the
# wrapper smoke tests), we fall back to find_library so the tests don't need
# the full HIP cmake config.
#
# Important: don't call find_library for amdhip64 when hip::host is already
# imported — the parent's find_package may have stitched amdhip64 together
# with `rocclr` (a transitive that doesn't exist as a standalone .so on
# some HIP versions), and a redundant find_library here produces a link
# command that ends up looking for `-lrocclr` and failing.
if (TARGET hip::host)
    target_link_libraries(aiter_triton_aot PUBLIC hip::host)
else()
    find_path(HIP_INCLUDE hip/hip_runtime.h PATHS /opt/rocm/include REQUIRED)
    find_library(HIP_RUNTIME amdhip64        PATHS /opt/rocm/lib     REQUIRED)
    target_include_directories(aiter_triton_aot PUBLIC ${HIP_INCLUDE})
    target_link_libraries(aiter_triton_aot      PUBLIC ${HIP_RUNTIME})
endif()

# HIP's API entry points are declared with `__host__ __device__` etc by default
# (HIP runtime headers assume hipcc). Compiling with plain gcc/clang requires
# __HIP_PLATFORM_AMD__ to select the right declarations.
target_compile_definitions(aiter_triton_aot PUBLIC __HIP_PLATFORM_AMD__)

# Make this library available to the parent ggml-hip target.
# (The actual `target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)`
# happens in the parent CMakeLists.txt once this is wired up.)

# ────────────────────────────────────────────────────────────────────────
# Standalone wrapper test executables. Built only when this directory is
# configured directly (CMAKE_SOURCE_DIR == CMAKE_CURRENT_SOURCE_DIR) — when
# included via add_subdirectory() from the parent ggml-hip build, we skip
# them to avoid leaking test binaries into the production build.
# ────────────────────────────────────────────────────────────────────────

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(test_mt_aiter_unified_attn
        wrappers/test_mt_aiter_unified_attn.cpp
    )
    target_link_libraries(test_mt_aiter_unified_attn PRIVATE aiter_triton_aot)

    add_executable(test_aiter_registry
        wrappers/test_aiter_registry.cpp
    )
    target_link_libraries(test_aiter_registry PRIVATE aiter_triton_aot)
endif()
