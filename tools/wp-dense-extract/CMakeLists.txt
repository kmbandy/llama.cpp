set(TARGET llama-wp-dense-extract)

add_executable(${TARGET}
    wp-dense-extract.cpp
    wp-dense-extract-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-router.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
