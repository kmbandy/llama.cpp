set(TARGET llama-wp-repack)

add_executable(${TARGET}
    wp-repack.cpp
    wp-repack-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-page-catalog.cpp
    ${CMAKE_SOURCE_DIR}/vendor/hash/sha256/sha256.c
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/hash
)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
