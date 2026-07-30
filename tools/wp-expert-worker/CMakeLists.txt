set(TARGET llama-wp-expert-worker-lib)

add_library(${TARGET}
    wp-expert-worker.cpp
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
)
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps
)
target_link_libraries(${TARGET} PUBLIC llama-common llama llama-expert-dispatch ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

set(TARGET llama-wp-expert-worker)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-wp-expert-worker-lib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
