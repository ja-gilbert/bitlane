# Builds libbitlane.so: the C reference always, and the CUDA kernel once kernel.cu exists.
cmake_minimum_required(VERSION 3.24)
project(bitlane C)

add_library(bitlane SHARED refsim.c)
target_compile_options(bitlane PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3;-march=native;-Wall;-Wextra>)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu)
  enable_language(CUDA)
  target_sources(bitlane PRIVATE kernel.cu)
  set_target_properties(bitlane PROPERTIES CUDA_ARCHITECTURES 89)  # RTX 4060: Ada, sm_89
  target_compile_options(bitlane PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()
