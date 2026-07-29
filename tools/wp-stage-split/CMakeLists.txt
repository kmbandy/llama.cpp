set(TARGET llama-wp-stage-split)

add_executable(${TARGET}
    wp-stage-split.cpp
    wp-stage-split-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-pipeline.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
