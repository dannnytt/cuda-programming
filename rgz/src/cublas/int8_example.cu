#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

void fill_random_int8(int8_t* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = (rand() % 255) - 127;
    }
}

void mm_int8(
    int8_t* A,
    int8_t* B,
    int32_t* C,
    int M,
    int N,
    int K,
    bool use_tensor_cores
) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    int32_t alpha = 1;
    int32_t beta = 0;

    cublasGemmAlgo_t algo = use_tensor_cores
        ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
        : CUBLAS_GEMM_DEFAULT;

    cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_8I, N,
        A, CUDA_R_8I, K,
        &beta,
        C, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I,
        algo
    );

    cublasDestroy(cublas_handle);
}

void process_mm_int8(int M, int N, int K) {
    int a_size = M * K;
    int b_size = K * N;
    int c_size = M * N;

    int8_t* h_a = new int8_t[a_size];
    int8_t* h_b = new int8_t[b_size];
    int32_t* h_c = new int32_t[c_size];

    fill_random_int8(h_a, a_size);
    fill_random_int8(h_b, b_size);

    int8_t *d_a, *d_b;
    int32_t *d_c;

    CUDA_CHECK_RETURN(cudaMalloc(&d_a, a_size * sizeof(int8_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, b_size * sizeof(int8_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_c, c_size * sizeof(int32_t)));
    CUDA_CHECK_RETURN(cudaMemset(d_c, 0, c_size * sizeof(int32_t)));


    CUDA_CHECK_RETURN(cudaMemcpy(d_a, h_a, a_size * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, h_b, b_size * sizeof(int8_t), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();
    mm_int8(d_a, d_b, d_c, M, N, K, false);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> mm_no_tc_time = stop - start;

    CUDA_CHECK_RETURN(cudaMemcpy(h_c, d_c, c_size * sizeof(int32_t), cudaMemcpyDeviceToHost));

    start = std::chrono::high_resolution_clock::now();
    mm_int8(d_a, d_b, d_c, M, N, K, true);
    stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> mm_tc_time = stop - start;

    CUDA_CHECK_RETURN(cudaMemcpy(h_c, d_c, c_size * sizeof(int32_t), cudaMemcpyDeviceToHost));

    printf("Время выполнения cuBLAS INT8 (no tensor cores): %.4f мс\n", mm_no_tc_time.count());
    printf("Время выполнения cuBLAS INT8 (tensor cores):   %.4f мс\n\n", mm_tc_time.count());

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    
    int sizes[][3] = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {16384, 16384, 16384}
    };

    for (auto& size : sizes) {
        printf("Тестирование INT8 для M = %d, N = %d, K = %d\n", size[0], size[1], size[2]);
        process_mm_int8(size[0], size[1], size[2]);
    }

    return 0;
}
