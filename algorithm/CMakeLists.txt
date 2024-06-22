cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER nvcc)

set(project_name "CUDA CCL")
project(${project_name} LANGUAGES CUDA CXX)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)


if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_OMPILE_FLAGS
    -XCudafe=--display_error_number
    -lineinfo
    --expt-extended-lambda
    -use_fast_math
    --expt-relaxed-constexpr
    -Xptax -warn-spills -res-usage
    --ptaxs-options=-v
    -Xcompiler -Wall -fopenmp -O3
)

# verbose
set(CMAKE_CUDA_VERBOSE_BUILD ON)
add_subdirectory(src)
