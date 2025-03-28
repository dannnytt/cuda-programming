#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <random>

__global__ void vector_add(double *A, double *B, double *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
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

void process_vector_addition(int vector_size, int threadsPerBlock, std::ofstream &output_file) {
    std::vector<double> h_A(vector_size), h_B(vector_size), h_C(vector_size);
    initialize_random_vectors(h_A, h_B, vector_size);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, vector_size * sizeof(double));
    cudaMalloc((void**)&d_B, vector_size * sizeof(double));
    cudaMalloc((void**)&d_C, vector_size * sizeof(double));

    cudaMemcpy(d_A, h_A.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice);

    int blockCount = vector_size / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add<<<blockCount, threadsPerBlock>>>(d_A, d_B, d_C, vector_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C.data(), d_C, vector_size * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Threads: " << threadsPerBlock << ", Time: " << milliseconds << " ms" << std::endl;

    output_file << threadsPerBlock << "," << milliseconds << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const int N = 10000000;
    std::ofstream output_file("data/results.csv");
    output_file << "Threads,Time(ms)\n";

    for (int threadsPerBlock = 2; threadsPerBlock <= 1024; threadsPerBlock *= 2) {
        process_vector_addition(N, threadsPerBlock, output_file);
    }

    output_file.close();
    return 0;
}
