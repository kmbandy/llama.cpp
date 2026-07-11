set(TARGET llama-pagedattn-dump)
add_executable(${TARGET} pagedattn-dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
