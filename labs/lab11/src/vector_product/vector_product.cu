#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

#define CUDA_CHECK_RETURN(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

__global__ void dot_product_kernel(int *a, int *b, int *partial_results, int n) {
    extern __shared__ int shared_cache[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_index = threadIdx.x;
    
    int temp = 0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    shared_cache[cache_index] = temp;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (cache_index < s) {
            shared_cache[cache_index] += shared_cache[cache_index + s];
        }
        __syncthreads();
    }
    
    if (cache_index == 0) {
        partial_results[blockIdx.x] = shared_cache[0];
    }
}

void benchmark_dot_product(size_t N, size_t chunk_size) {
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_partial = new int[N];

    for (size_t i = 0; i < N; ++i) {
        h_a[i] = rand() % RAND_MAX;
        h_b[i] = rand() % RAND_MAX;
    }

    int *d_a, *d_b, *d_partial;
    CUDA_CHECK_RETURN(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, N * sizeof(int)));
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    CUDA_CHECK_RETURN(cudaMalloc(&d_partial, grid_size * sizeof(int)));

    const int num_streams = (N + chunk_size - 1) / chunk_size;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; i += chunk_size) {
        size_t current_chunk = std::min(chunk_size, N - i);
        int stream_idx = i / chunk_size;
        
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_a + i, h_a + i, 
                                  current_chunk * sizeof(int),
                                  cudaMemcpyHostToDevice, streams[stream_idx]));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_b + i, h_b + i, 
                                  current_chunk * sizeof(int),
                                  cudaMemcpyHostToDevice, streams[stream_idx]));
    }

    dot_product_kernel<<<grid_size, block_size, block_size*sizeof(int)>>>(
        d_a, d_b, d_partial, N);
    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_partial, d_partial, 
                              grid_size * sizeof(int),
                              cudaMemcpyDeviceToHost, streams[0]));

    for (auto& stream : streams) {
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Chunk size: %zu KB, (%zu elem)\n", chunk_size / 1024, chunk_size);
    printf("Time: %.4f ms\n\n", elapsed.count());

    delete[] h_a;
    delete[] h_b;
    delete[] h_partial;
    CUDA_CHECK_RETURN(cudaFree(d_a));
    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_partial));
    for (auto& stream : streams) {
        CUDA_CHECK_RETURN(cudaStreamDestroy(stream));
    }
}

int main() {
    srand(time(nullptr));
    const size_t N = 1 << 24;
    std::cout << "Testing scalar product with N = " << N << " elements" << std::endl;

    std::vector<size_t> chunk_sizes = {
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24
    };

    for (size_t chunk : chunk_sizes) {
        benchmark_dot_product(N, chunk);
    }

    return 0;
}