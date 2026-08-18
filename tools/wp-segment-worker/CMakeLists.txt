set(TARGET llama-wp-segment-worker-lib)

add_library(${TARGET} wp-segment-worker.cpp)
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/tools/pipeline
    ${CMAKE_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PUBLIC
    llama-pipeline-stage
    llama-expert-dispatch
    llama-common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PUBLIC cxx_std_17)

set(TARGET llama-wp-segment-worker)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-wp-segment-worker-lib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
