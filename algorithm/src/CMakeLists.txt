cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "CCL")

add_executable(${exec_name} main.cu)

target_include_directories(${exec_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/alg")

set_target_properties(${exec_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON COMPILE_OPTIONS -G)

add_executable("serial" serial.cxx)
target_include_directories("serial" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")