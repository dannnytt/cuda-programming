cmake_minimum_required(VERSION 3.28)
project(lab11 CUDA)


set(TARGET1_SRC "src/memory_copy/mem_copy_test.cu")
set(TARGET2_SRC "src/vector_addition/vector_addition.cu")
set(TARGET3_SRC "src/vector_product/vector_product.cu")

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(memory_test ${TARGET1_SRC})
add_executable(vector_addition ${TARGET2_SRC})
add_executable(vector_product ${TARGET3_SRC})

set_target_properties(memory_test PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(vector_addition PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(vector_product PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(memory_test  ${CUDA_LIBRARIES})
target_link_libraries(vector_addition  ${CUDA_LIBRARIES})
target_link_libraries(vector_product  ${CUDA_LIBRARIES})
