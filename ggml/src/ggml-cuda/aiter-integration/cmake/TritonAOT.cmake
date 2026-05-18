# TritonAOT.cmake
#
# CMake helpers for ahead-of-time (AOT) compilation of Triton @jit kernels.
#
# Wraps `python -m triton.tools.compile` (and `triton.tools.link` for
# multi-specialization aggregation). Each invocation produces a .c file
# with the kernel binary embedded as a static byte array plus a .h with a
# clean C launcher function:
#
#     hipError_t kernel_NAME_SUFFIX(
#         hipStream_t stream, unsigned gX, unsigned gY, unsigned gZ,
#         <typed args...>);
#
# Status: DRAFT, not yet wired into the main ggml-hip build. The main
# CMakeLists.txt deliberately does NOT include this module until MAD-187
# milestone 1 (vector-add POC) verifies the toolchain end-to-end on
# gfx1201. See docs/aiter-integration/ARCHITECTURE.md and the CMAKE-GLUE
# design notes alongside this file.
#
# Author: Kurt + Claude (design session 2026-05-17)

if(TARGET TritonAOT::Helpers)
    return()  # already loaded
endif()

# ────────────────────────────────────────────────────────────────────────
# Prerequisites — find the Python interpreter that has Triton installed.
# We don't make Triton a hard CMake requirement (development-only dep);
# instead, this module is silently a no-op if Triton isn't importable.
# ────────────────────────────────────────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import triton, sys; sys.stdout.write(triton.__version__)"
    OUTPUT_VARIABLE TRITON_VERSION
    RESULT_VARIABLE TRITON_IMPORT_RC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT TRITON_IMPORT_RC EQUAL 0)
    message(STATUS "TritonAOT: Python ${Python3_EXECUTABLE} cannot import triton — disabling AOT path")
    set(TRITON_AOT_AVAILABLE FALSE CACHE INTERNAL "Triton AOT not available")
    return()
endif()

message(STATUS "TritonAOT: Triton ${TRITON_VERSION} available via ${Python3_EXECUTABLE}")

# Verify the .compile entry point exists (it's a moving target across versions).
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import triton.tools.compile"
    RESULT_VARIABLE TRITON_AOT_RC
    ERROR_QUIET
)
if(NOT TRITON_AOT_RC EQUAL 0)
    message(WARNING "TritonAOT: triton.tools.compile module missing — AOT compile won't work")
    set(TRITON_AOT_AVAILABLE FALSE CACHE INTERNAL "Triton AOT not available")
    return()
endif()

set(TRITON_AOT_AVAILABLE TRUE CACHE INTERNAL "Triton AOT available")
add_library(TritonAOT::Helpers INTERFACE IMPORTED GLOBAL)

# ────────────────────────────────────────────────────────────────────────
# Target string construction. Triton's AOT tool expects:
#     hip:<gfx-arch>:<warp-size>
# E.g. hip:gfx1201:32 for RDNA4, hip:gfx1030:32 for RDNA2, hip:gfx942:64 for CDNA3.
# ────────────────────────────────────────────────────────────────────────

function(_triton_aot_target out_var arch)
    if(arch MATCHES "^gfx9[0-9][0-9]$")
        # CDNA / Instinct — wavefront size 64
        set(${out_var} "hip:${arch}:64" PARENT_SCOPE)
    elseif(arch MATCHES "^gfx1[0-9][0-9][0-9]$")
        # RDNA — wavefront size 32 (wave32 default on RDNA1+)
        set(${out_var} "hip:${arch}:32" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "TritonAOT: unrecognized AMD arch '${arch}'")
    endif()
endfunction()

# ────────────────────────────────────────────────────────────────────────
# add_triton_aot_kernel — main entry point.
#
# Compiles a single Triton @jit kernel for a list of (signature, grid,
# num_warps, num_stages, arch) tuples. Each tuple yields one specialization
# (one .c/.h pair). Multiple specializations of the same kernel can later
# be combined via add_triton_aot_link().
#
# Usage:
#   add_triton_aot_kernel(
#       NAME           vector_add                       # logical name
#       SOURCE         kernels/vector_add.py            # .py file w/ @jit
#       KERNEL_NAME    add_kernel                       # @jit function name inside .py
#       OUT_DIR        ${CMAKE_BINARY_DIR}/aiter-integration/triton-out
#       SPECS
#           SIGNATURE  "*fp32:16, *fp32:16, *fp32:16, i32"
#           GRID       "n_elements // BLOCK_SIZE"
#           NUM_WARPS  4
#           NUM_STAGES 1
#           ARCH       gfx1201
#       # ... more SPECS blocks allowed for additional specializations
#   )
#
# Outputs (per spec block):
#   ${OUT_DIR}/${NAME}.<suffix>.c
#   ${OUT_DIR}/${NAME}.<suffix>.h
# where <suffix> is Triton's deterministic specialization hash.
#
# Caller can then `target_sources(<their-target> PRIVATE ${${NAME}_AOT_C_FILES})`.
# ────────────────────────────────────────────────────────────────────────

function(add_triton_aot_kernel)
    set(options)
    set(oneValueArgs NAME SOURCE KERNEL_NAME OUT_DIR)
    set(multiValueArgs SPECS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT TRITON_AOT_AVAILABLE)
        message(WARNING "TritonAOT: skipping ${ARG_NAME} (toolchain unavailable)")
        return()
    endif()

    foreach(req NAME SOURCE KERNEL_NAME OUT_DIR)
        if(NOT ARG_${req})
            message(FATAL_ERROR "add_triton_aot_kernel: missing required arg ${req}")
        endif()
    endforeach()

    if(NOT IS_ABSOLUTE ${ARG_SOURCE})
        set(ARG_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SOURCE})
    endif()
    if(NOT EXISTS ${ARG_SOURCE})
        message(FATAL_ERROR "add_triton_aot_kernel: source not found: ${ARG_SOURCE}")
    endif()

    file(MAKE_DIRECTORY ${ARG_OUT_DIR})

    # Parse SPECS — it's a flat list that we walk in 5-key blocks:
    #   SIGNATURE "...", GRID "...", NUM_WARPS N, NUM_STAGES N, ARCH arch
    set(c_files)
    set(h_files)
    set(i 0)
    list(LENGTH ARG_SPECS specs_len)
    while(i LESS specs_len)
        set(signature "")
        set(grid "")
        set(num_warps 4)
        set(num_stages 1)
        set(arch "")
        # Read 10 elements: 5 keys + 5 values
        foreach(_unused RANGE 0 4)
            list(GET ARG_SPECS ${i} key)
            math(EXPR i_plus "${i} + 1")
            list(GET ARG_SPECS ${i_plus} value)
            if(key STREQUAL "SIGNATURE")
                set(signature "${value}")
            elseif(key STREQUAL "GRID")
                set(grid "${value}")
            elseif(key STREQUAL "NUM_WARPS")
                set(num_warps ${value})
            elseif(key STREQUAL "NUM_STAGES")
                set(num_stages ${value})
            elseif(key STREQUAL "ARCH")
                set(arch ${value})
            else()
                message(FATAL_ERROR "add_triton_aot_kernel: unknown SPEC key '${key}'")
            endif()
            math(EXPR i "${i} + 2")
        endforeach()

        if(NOT signature OR NOT grid OR NOT arch)
            message(FATAL_ERROR "add_triton_aot_kernel: incomplete SPEC block (need SIGNATURE, GRID, ARCH at minimum)")
        endif()

        _triton_aot_target(target_str ${arch})

        # Suffix is the triton-generated hash (we can't predict it offline
        # for byte-perfect path matching), so we emit into a per-arch
        # subdirectory and glob the result.
        set(out_subdir ${ARG_OUT_DIR}/${ARG_NAME}/${arch})
        file(MAKE_DIRECTORY ${out_subdir})

        # Stable stamp file we DO control — used as the custom_command output.
        # The .c/.h files are co-located but their exact names are
        # specialization-suffix-dependent; we glob them at link time.
        set(stamp ${out_subdir}/.${ARG_NAME}.${arch}.stamp)

        add_custom_command(
            OUTPUT ${stamp}
            COMMAND ${CMAKE_COMMAND} -E rm -f ${out_subdir}/${ARG_NAME}.*.c ${out_subdir}/${ARG_NAME}.*.h
            COMMAND ${Python3_EXECUTABLE} -m triton.tools.compile
                ${ARG_SOURCE}
                --kernel-name ${ARG_KERNEL_NAME}
                --target ${target_str}
                --signature ${signature}
                --grid ${grid}
                --num-warps ${num_warps}
                --num-stages ${num_stages}
                --out-name ${ARG_NAME}
                --out-path ${out_subdir}/${ARG_NAME}
            COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
            DEPENDS ${ARG_SOURCE}
            COMMENT "Triton AOT: ${ARG_NAME} → ${target_str} (warps=${num_warps}, stages=${num_stages})"
            VERBATIM
        )

        # Per spec, we WILL produce .c/.h with a deterministic naming. Since
        # the suffix depends on triton's internal hash, we collect them via
        # a wrapper target that globs at build time. CMake's POST_BUILD
        # custom_command on the same custom target lets us assemble the lists.
        list(APPEND stamps ${stamp})
        list(APPEND out_subdirs ${out_subdir})
    endwhile()

    # One custom target gathering all specializations of this kernel.
    add_custom_target(triton_aot_${ARG_NAME} ALL DEPENDS ${stamps})

    # Expose the discovered .c files via a variable in PARENT_SCOPE so the
    # caller can attach them to their library target. We use a CMake glob
    # that re-evaluates at configure time AFTER the custom_command runs —
    # this works because configure happens before build, but glob results
    # for generated files are populated post-build using CONFIGURE_DEPENDS.
    set(glob_pattern)
    foreach(d ${out_subdirs})
        list(APPEND glob_pattern "${d}/${ARG_NAME}.*.c")
    endforeach()
    file(GLOB ${ARG_NAME}_AOT_C_FILES CONFIGURE_DEPENDS ${glob_pattern})
    set(${ARG_NAME}_AOT_C_FILES "${${ARG_NAME}_AOT_C_FILES}" PARENT_SCOPE)
    set(${ARG_NAME}_AOT_OUT_DIRS "${out_subdirs}" PARENT_SCOPE)
endfunction()

# ────────────────────────────────────────────────────────────────────────
# add_triton_aot_link — link multiple specializations of one logical
# kernel into a unified entry point via `triton.tools.link`.
#
# Optional — only needed when we have many specializations per kernel and
# want a single C entry that runtime-dispatches. For the vector-add POC
# this is overkill; for unified_attention with multiple HEAD_SIZE values
# it's essential.
#
# Stub left as a placeholder — implement in MAD-187 milestone 2 once the
# specialization matrix for unified_attention is finalized.
# ────────────────────────────────────────────────────────────────────────

function(add_triton_aot_link)
    message(FATAL_ERROR "add_triton_aot_link: not yet implemented — see TritonAOT.cmake comment block")
endfunction()
