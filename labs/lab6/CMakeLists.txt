cmake_minimum_required(VERSION 3.28)
project(lab6 CUDA)

set(SRC_DIR "src/")

set(TARGET1_SRC "${SRC_DIR}/transpose_global.cu")
set(TARGET2_SRC "${SRC_DIR}/transpose_shared.cu")
set(TARGET3_SRC "${SRC_DIR}/transpose_shared_opt.cu")

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(transpose_global ${TARGET1_SRC})
add_executable(transpose_shared ${TARGET2_SRC})
add_executable(transpose_shared_opt ${TARGET3_SRC})

set_target_properties(transpose_global PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(transpose_shared PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(transpose_shared_opt PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(transpose_global ${CUDA_LIBRARIES})
target_link_libraries(transpose_shared ${CUDA_LIBRARIES})
target_link_libraries(transpose_shared_opt ${CUDA_LIBRARIES})
