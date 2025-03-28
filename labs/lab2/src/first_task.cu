#include <iostream>
#include <vector>
#include <random>
#include <chrono>

const size_t VECTOR_SIZE = 100000000;

std::vector<double> VECTOR_A(VECTOR_SIZE);
std::vector<double> VECTOR_B(VECTOR_SIZE);
std::vector<double> VECTOR_C(VECTOR_SIZE, 0.0);

__global__ void vector_add_cuda(const double *a, const double *b, double *c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void initialize_random_vectors() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    for (size_t i = 0; i < VECTOR_SIZE; i++) {
        VECTOR_A[i] = dist(gen);
        VECTOR_B[i] = dist(gen);
    }
}

int main() {
    initialize_random_vectors();

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < VECTOR_SIZE; i++) {
        VECTOR_C[i] = VECTOR_A[i] + VECTOR_B[i];
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential CPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu).count()
              << " ms" << std::endl << std::endl;


    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&d_B, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&d_C, VECTOR_SIZE * sizeof(double));

    cudaMemcpy(d_A, VECTOR_A.data(), VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, VECTOR_B.data(), VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (VECTOR_SIZE + block_size - 1) / block_size;
    std::cout << "GPU threads count = " << grid_size * block_size << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vector_add_cuda<<<grid_size, block_size>>>(d_A, d_B, d_C, VECTOR_SIZE);
    cudaDeviceSynchronize();
    auto stop_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(VECTOR_C.data(), d_C, VECTOR_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "CUDA GPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop_gpu - start_gpu).count()
              << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
