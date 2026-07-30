set(TARGET llama-wp-expert-shard)

add_executable(${TARGET}
    wp-expert-shard.cpp
    wp-expert-shard-lib.cpp
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
