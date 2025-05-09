cmake_minimum_required(VERSION 3.28)
project(rgz CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(TARGET1_SRC "src/cublas/fp16_example.cu")
set(TARGET2_SRC "src/cublas/int8_example.cu")
set(TARGET3_SRC "src/wmma/fp16_example.cu")
set(TARGET4_SRC "src/wmma/int8_example.cu")


add_executable(cublas_fp16_example ${TARGET1_SRC})
add_executable(cublas_int8_example ${TARGET2_SRC})
add_executable(wmma_fp16_example ${TARGET3_SRC})
add_executable(wmma_int8_example ${TARGET4_SRC})

set_target_properties(cublas_fp16_example PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(cublas_int8_example PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(wmma_fp16_example PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(wmma_int8_example PROPERTIES CUDA_ARCHITECTURES "75")


target_link_libraries(cublas_fp16_example PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(cublas_int8_example PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(wmma_int8_example PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(wmma_int8_example PRIVATE CUDA::cudart CUDA::cublas)
