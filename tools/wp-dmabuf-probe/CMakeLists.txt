set(TARGET wp-dmabuf-probe)

add_executable(${TARGET} main.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /opt/rocm/include)
target_compile_definitions(${TARGET} PRIVATE __HIP_PLATFORM_AMD__)

if (TARGET hip::host)
    target_link_libraries(${TARGET} PRIVATE hip::host dl)
else()
    find_library(WP_PROBE_HIP_RUNTIME amdhip64 PATHS /opt/rocm/lib REQUIRED)
    target_link_libraries(${TARGET} PRIVATE ${WP_PROBE_HIP_RUNTIME} dl)
endif()
