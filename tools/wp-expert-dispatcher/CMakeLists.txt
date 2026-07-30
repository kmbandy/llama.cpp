set(TARGET llama-wp-expert-dispatcher-lib)

add_library(${TARGET}
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-expert-dispatcher.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-transport.cpp
)
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
)
target_link_libraries(${TARGET} PUBLIC llama)
target_compile_features(${TARGET} PUBLIC cxx_std_17)

set(TARGET llama-wp-expert-dispatcher)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-wp-expert-dispatcher-lib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(TARGET test-wp-expert-dispatcher)
    add_dependencies(test-wp-expert-dispatcher llama-wp-expert-worker)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
