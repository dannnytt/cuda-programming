#include<cuda_runtime.h>
#include <cublas_v2.h>
#include<iostream>
#include<chrono>

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}


void fill_random(double *vec, int size) {
    for (int i = 0; i < size; i++)
        vec[i] = (double) rand() / RAND_MAX;
}

void mm_no_tc(
    double *A, 
    double *B,
    double *C,
    int M,
    int N,
    int K
) {

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    const double alpha = 1.0;
    const double beta = 0.0;

    cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                             
        &alpha,
        B, CUDA_R_64F, N,                   
        A, CUDA_R_64F, K,                   
        &beta,
        C, CUDA_R_64F, N,                   
        CUDA_R_64F,                         
        CUBLAS_GEMM_DEFAULT                 
    );

    cublasDestroy(cublas_handle);
}


void process_mm(int M, int N, int K) {
    int a_size = M * K;  
    int b_size = K * N;
    int c_size = M * N;

    double *h_a, *h_b, *h_c;

    h_a = new double[a_size];
    h_b = new double[b_size];
    h_c = new double[c_size];

    fill_random(h_a, a_size);
    fill_random(h_b, b_size);

    double *d_a, *d_b, *d_c;
    CUDA_CHECK_RETURN(cudaMalloc(&d_a, a_size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, b_size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_c, c_size * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMemset(d_c, 0, c_size * sizeof(double)));


    CUDA_CHECK_RETURN(cudaMemcpy(d_a, h_a, a_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, h_b, b_size * sizeof(double), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();
    mm_no_tc(d_a, d_b, d_c, M, N, K);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> mm_no_tc_time = stop - start;
    CUDA_CHECK_RETURN(cudaMemcpy(h_c, d_c, c_size * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Время выполнения cuBLAS FLOAT32 (no tensor cores): %.4f мс\n\n", mm_no_tc_time.count());

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char *argv[]) {
    
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
        {8192, 8192, 8192}
    };

    for (auto& size : sizes) {
        printf("Тестирование для M = %d, N = %d, K = %d\n", size[0], size[1], size[2]);
        process_mm(size[0], size[1], size[2]);
    }

    return 0;
}