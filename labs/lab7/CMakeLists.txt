cmake_minimum_required(VERSION 3.28)
project(lab6 CUDA)

set(SRC_DIR "src/")

set(TARGET_SRC "${SRC_DIR}/integrate.cu")


set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(integrate ${TARGET_SRC})


set_target_properties(integrate PROPERTIES CUDA_ARCHITECTURES "75")


target_link_libraries(integrate ${CUDA_LIBRARIES})
