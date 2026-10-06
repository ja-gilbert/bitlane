# Builds libbitlane.so: the C reference and the CUDA kernel.
cmake_minimum_required(VERSION 3.24)
project(bitlane C CUDA)

add_library(bitlane SHARED refsim.c kernel.cu)
set_target_properties(bitlane PROPERTIES CUDA_ARCHITECTURES 89)  # RTX 4060: Ada, sm_89
target_compile_options(bitlane PRIVATE $<$<COMPILE_LANGUAGE:C>:-O3;-march=native;-Wall;-Wextra>)
target_compile_options(bitlane PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
