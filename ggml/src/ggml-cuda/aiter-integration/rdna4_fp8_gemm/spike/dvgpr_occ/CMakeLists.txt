#
# dram_meter: standalone VRAM read-traffic meter for gfx1201 (R9700)
# using the rocprofiler-sdk device-counting service.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(dram_meter LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/lib/cmake)

find_package(rocprofiler-sdk REQUIRED)
find_package(hsa-runtime64 REQUIRED HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(Threads REQUIRED)

add_executable(dram_meter dram_meter.cpp)

target_compile_features(dram_meter PRIVATE cxx_std_17)
target_compile_options(dram_meter PRIVATE -W -Wall -Wextra)

target_link_libraries(
    dram_meter
    PRIVATE rocprofiler-sdk::rocprofiler-sdk
            hsa-runtime64::hsa-runtime64
            Threads::Threads
            ${CMAKE_DL_LIBS})
