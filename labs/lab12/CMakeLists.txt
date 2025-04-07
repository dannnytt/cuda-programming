cmake_minimum_required(VERSION 3.28)
project(lab12 CUDA)

find_package(CUDAToolkit REQUIRED)

set(TARGET1_SRC "matrix_multiplication/simple_example.cu")
set(TARGET2_SRC "matrix_multiplication/cublas_example.cu")

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(simple_example ${TARGET1_SRC})
add_executable(cublas_example ${TARGET2_SRC})

set_target_properties(simple_example PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(cublas_example PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(simple_example PRIVATE CUDA::cudart)
target_link_libraries(cublas_example PRIVATE CUDA::cudart CUDA::cublas)