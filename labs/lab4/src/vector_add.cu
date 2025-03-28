#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#define VECTOR_SIZE (1 << 20)

std::ofstream output_file("data/results.csv");

std::vector<double> h_vector1(VECTOR_SIZE), 
                    h_vector2(VECTOR_SIZE), 
                    h_vector3(VECTOR_SIZE); 

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}


__global__ void vectors_addition(double *vector1, double *vector2, double *vector3) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    vector3[idx] = vector1[idx] + vector2[idx];
}

void initialize_random_vectors(std::vector<double> &vector1,
                                std::vector<double> &vector2) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector1[i] = dist(gen);
        vector2[i] = dist(gen);
    }
}

void process_vectors_additon(int threads_per_block) {

    size_t vector_size_bytes = (VECTOR_SIZE) * sizeof(double);

    double *d_vector1, *d_vector2, *d_vector3;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector1, vector_size_bytes));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector2, vector_size_bytes));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector3, vector_size_bytes));

    initialize_random_vectors(h_vector1, h_vector2);
    CUDA_CHECK_RETURN(cudaMemcpy(d_vector1, h_vector1.data(), vector_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_vector2, h_vector2.data(), vector_size_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    
    int blocks_count = VECTOR_SIZE / threads_per_block;
    CUDA_CHECK_RETURN(cudaEventRecord(start));
    vectors_addition<<<blocks_count, threads_per_block>>>(d_vector1, d_vector2, d_vector3);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    cudaFree(d_vector1);
    cudaFree(d_vector2); 
    cudaFree(d_vector3);

    float elapsed_time = 0.0;
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));
    
    std::cout << "Threads: " << threads_per_block << ", Time: " << elapsed_time << " ms" << std::endl;
    output_file << threads_per_block << "," << elapsed_time << std::endl;
}

int main(int argc, char *argv[]) {
    output_file << "Threads,Time(ms)\n";
    int threads_per_block[] = {
        1, 16, 
        32, 64,
        128, 256, 
        512, 1024
    };

    for (int i : threads_per_block) {
        process_vectors_additon(i);
    }

    std::system("./run_python.sh");
    return 0;
}
