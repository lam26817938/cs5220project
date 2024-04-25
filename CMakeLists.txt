cmake_minimum_required(VERSION 3.28)
project(cs5220project CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cs5220project main.cu)

set_target_properties(cs5220project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
