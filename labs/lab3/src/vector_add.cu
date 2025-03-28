#include <iostream>
#include <vector>
#include <random>

#define VECTOR_SIZE 10'000'000

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

__global__ void vectors_add(double *vector1, double *vector2, double *vector3) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;    
    vector3[idx]  = vector1[idx] + vector2[idx];
}

void initialize_random_vectors(std::vector<double> &A, std::vector<double> &B, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
}

void process_vectors_addition(int threads_per_block) {
    std::vector<double> h_vector1(VECTOR_SIZE), 
                        h_vector2(VECTOR_SIZE), 
                        h_vector3(VECTOR_SIZE);
        
    double *d_vector1, *d_vector2, *d_vector3;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector1, VECTOR_SIZE * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector2, VECTOR_SIZE * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector3, VECTOR_SIZE * sizeof(double)));

    initialize_random_vectors(h_vector1, h_vector2, VECTOR_SIZE);
    CUDA_CHECK_RETURN(cudaMemcpy(d_vector1, h_vector1.data(), VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_vector2, h_vector2.data(), VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    int block_count = VECTOR_SIZE / threads_per_block;
    vectors_add<<<block_count, threads_per_block>>>(d_vector1, d_vector2, d_vector3);
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    CUDA_CHECK_RETURN(cudaMemcpy(h_vector3.data(), d_vector3, VECTOR_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    double sum = std::accumulate(h_vector3.begin(), h_vector3.end(), 0.0);
    fprintf(stdout, "Threads: %d, Sum = %.2f\n\n", threads_per_block, sum);
    
    CUDA_CHECK_RETURN(cudaFree(d_vector1));
    CUDA_CHECK_RETURN(cudaFree(d_vector2));
    CUDA_CHECK_RETURN(cudaFree(d_vector3));
}

int main() {
    
    process_vectors_addition(1024);
    process_vectors_addition(1025);
    return 0;
}