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
# Pull the generated .c files into a static library. The HEADERS need to
# be discoverable too via include path.
# ────────────────────────────────────────────────────────────────────────

# vector_add_AOT_C_FILES was set in PARENT_SCOPE by add_triton_aot_kernel.
# At configure time it'll be empty (no build has run); at re-configure
# after build, it'll be populated via CONFIGURE_DEPENDS glob.

add_library(aiter_triton_aot STATIC
    wrappers/vector_add_wrapper.cpp
    # Generated sources added at build time via target_sources below:
    ${vector_add_AOT_C_FILES}
)

target_include_directories(aiter_triton_aot
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/wrappers
        # Each per-arch output dir contains the .h for that specialization.
        # Multiple archs may produce same-named headers — we trust the
        # link-stage merger or namespace via subdir.
)

foreach(arch ${AITER_AOT_ARCHES})
    target_include_directories(aiter_triton_aot
        PUBLIC ${AITER_AOT_OUT_DIR}/vector_add/${arch}
    )
endforeach()

# Triton's generated .c files use HIP's runtime API (hipModuleLoad,
# hipModuleLaunchKernel). We need to link against amdhip64.
find_library(HIP_RUNTIME amdhip64 PATHS /opt/rocm/lib REQUIRED)
target_link_libraries(aiter_triton_aot PUBLIC ${HIP_RUNTIME})

# Make this library available to the parent ggml-hip target.
# (The actual `target_link_libraries(ggml-hip PRIVATE aiter_triton_aot)`
# happens in the parent CMakeLists.txt once this is wired up.)
