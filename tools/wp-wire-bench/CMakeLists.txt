set(TARGET llama-wp-wire-bench)

add_executable(${TARGET} wp-wire-bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama-pipeline-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
