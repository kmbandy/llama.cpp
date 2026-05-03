llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")
file(GLOB LLAMA_WEIGHT_PAGER_SOURCES "weight-pager/*.cpp")
file(GLOB LLAMA_MEMORY_TIER_SOURCES "memory-tier/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-tiered.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-weight-pager.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            ${LLAMA_WEIGHT_PAGER_SOURCES}
            ${LLAMA_MEMORY_TIER_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
if(GGML_HIP)
    target_include_directories(llama PRIVATE /opt/rocm/include)
    target_compile_definitions(llama PRIVATE __HIP_PLATFORM_AMD__)
endif()
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# Optional liburing for the weight pager's async file IO path. When found,
# the IoUringAsync FileIOLayer is compiled and selected at runtime when
# --weight-paging-prefetch is on. Without it, the pager falls back to
# SyncPread which serialises all pread()s on the eval-callback thread.
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(LIBURING IMPORTED_TARGET liburing)
    if (LIBURING_FOUND)
        message(STATUS "Found liburing ${LIBURING_VERSION} - enabling async weight-pager IO")
        target_compile_definitions(llama PRIVATE LLAMA_HAVE_IO_URING)
        target_link_libraries(llama PRIVATE PkgConfig::LIBURING)
    else()
        message(STATUS "liburing not found - weight pager will use SyncPread fallback")
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
