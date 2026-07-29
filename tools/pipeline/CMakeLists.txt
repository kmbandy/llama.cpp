# llama-pipeline-impl: stage driver logic, reusable by app

set(TARGET llama-pipeline-impl)

# pipe-transport/pipe-protocol are compiled here (they are NOT part of the
# llama library, keeping the core single-process path untouched). llama-pipeline
# is NOT compiled here: the linked `llama` library already provides it, and
# compiling it again would duplicate its symbols at link time.
add_library(${TARGET}
    pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-transport.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-protocol.cpp
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
)
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-pipeline executable

set(TARGET llama-pipeline)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-pipeline-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
