set(TARGET llama-wp-worker-replay)

add_executable(${TARGET} wp-worker-replay.cpp)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pipeline
)
# llama-expert-dispatch calls llama_pipeline_validate_stages, which lives in
# libllama (see tools/wp-expert-worker/CMakeLists.txt for the same note) --
# link llama too or the symbol goes unresolved on the HIP/CUDA build.
target_link_libraries(${TARGET} PRIVATE llama-expert-dispatch llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
