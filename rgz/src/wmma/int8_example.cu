#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <chrono>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

void fill_random(int8_t *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 256 - 128;
    }
}

__global__ void wmma_gemm_kernel_int8(int8_t *a, int8_t *b, int32_t *c, int m, int n, int k) {
    using namespace nvcuda;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        const int8_t *tile_a = a + aRow * k + aCol;
        const int8_t *tile_b = b + bRow * n + bCol;

        wmma::load_matrix_sync(a_frag, tile_a, k);
        wmma::load_matrix_sync(b_frag, tile_b, n);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int *tile_c = c + warpM * WMMA_M * n + warpN * WMMA_N;
    wmma::store_matrix_sync(tile_c, c_frag, n, wmma::mem_row_major);
}

void process_mm_int8(int M, int N, int K) {
    int a_size =  M * K;
    int b_size =  K * N;
    int c_size =  M * N;

    int8_t *h_a, *h_b;
    int32_t *h_c;
    h_a = new int8_t[a_size];
    h_b = new int8_t[b_size];
    h_c = new int32_t[c_size];

    fill_random(h_a, a_size);
    fill_random(h_b, b_size);

    int8_t *d_a, *d_b;
    int *d_c;
    CUDA_CHECK_RETURN(cudaMalloc(&d_a, a_size * sizeof(int8_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, b_size * sizeof(int8_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_c, c_size * sizeof(int32_t)));
    CUDA_CHECK_RETURN(cudaMemset(d_c, 0, c_size * sizeof(int32_t)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_a, h_a, a_size * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_b, h_b, b_size * sizeof(int8_t), cudaMemcpyHostToDevice));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (M + (WMMA_M * 4 - 1)) / (WMMA_M * 4), 
        (N + WMMA_N - 1) / WMMA_N
    );

    auto start = std::chrono::high_resolution_clock::now();
    wmma_gemm_kernel_int8<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    printf("Время выполнения WMMA для INT8: %.4f\n\n", duration.count());

    cudaMemcpy(h_c, d_c, c_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
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
        {16384, 16384, 16384},
    };

    for (auto& size : sizes) {
        printf("Тестирование WMMA для INT8: M = %d, N = %d, K = %d\n", size[0], size[1], size[2]);
        process_mm_int8(size[0], size[1], size[2]);
    }

    return 0;
}
