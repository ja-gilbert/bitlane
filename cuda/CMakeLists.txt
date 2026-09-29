# Builds libbitlane.co: the C reference simulator now, the CUDA kernel later.
cmake_minimum_required(VERSION 3.24)
project(bitlane C)

add_library(bitlane SHARED refsim.c)
target_compile_options(bitlane PRIVATE -O3 -march=native -Wall -Wextra)
