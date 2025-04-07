cmake_minimum_required(VERSION 3.28)
project(lab13 CUDA)

find_package(CUDAToolkit REQUIRED)

set(TARGET1_SRC "matrix_multiplication/cublas_example.cu")
set(TARGET2_SRC "matrix_multiplication/wmma_example.cu")

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(cublas_example ${TARGET1_SRC})
add_executable(wmma_example ${TARGET2_SRC})

set_target_properties(cublas_example PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(wmma_example PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(cublas_example PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(wmma_example PRIVATE CUDA::cudart CUDA::cublas)