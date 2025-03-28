#include <cuda.h>
#include <iostream>
#include <vector>
#include<random>

#define CUDA_CHECK_CALL(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA error: " << errStr << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define VECTOR_SIZE 10


void initialize_random_vectors(std::vector<double> &vector1, std::vector<double> &vector2)  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        vector1[i] = dist(gen);
        vector2[i] = dist(gen);
    }
}

int main() {
    
    CUDA_CHECK_CALL(cuInit(0));
    
    CUdevice device;
    CUDA_CHECK_CALL(cuDeviceGet(&device, 0));

    CUcontext context;
    CUDA_CHECK_CALL(cuCtxCreate(&context, 0, device));

    CUmodule module;
    CUDA_CHECK_CALL(cuModuleLoad(&module, "ptx/kernel.ptx"));

    CUfunction kernel;
    CUDA_CHECK_CALL(cuModuleGetFunction(&kernel, module, "kernel"));

    std::vector<double> h_vector1(VECTOR_SIZE, 5);
    std::vector<double> h_vector2(VECTOR_SIZE, 10);
    std::vector<double> h_result(VECTOR_SIZE, 0);

    initialize_random_vectors(h_vector1, h_vector2);

    CUdeviceptr d_vector1, d_vector2, d_result;
    CUDA_CHECK_CALL(cuMemAlloc(&d_vector1, VECTOR_SIZE * sizeof(double)));
    CUDA_CHECK_CALL(cuMemAlloc(&d_vector2, VECTOR_SIZE * sizeof(double)));
    CUDA_CHECK_CALL(cuMemAlloc(&d_result, VECTOR_SIZE * sizeof(double)));

    CUDA_CHECK_CALL(cuMemcpyHtoD(d_vector1, h_vector1.data(), VECTOR_SIZE * sizeof(double)));
    CUDA_CHECK_CALL(cuMemcpyHtoD(d_vector2, h_vector2.data(), VECTOR_SIZE * sizeof(double)));

    int size = VECTOR_SIZE;
    void *args[] = { &d_vector1, &d_vector2, &d_result, &size };
    CUDA_CHECK_CALL(cuLaunchKernel(kernel, 1, 1, 1, VECTOR_SIZE, 1, 1, 0, 0, args, 0));

    CUDA_CHECK_CALL(cuMemcpyDtoH(h_result.data(), d_result, VECTOR_SIZE * sizeof(double)));

    std::cout << "Результат: " << std::endl;
    for (double val : h_result) 
        printf("%-8.2f", val);
    std::cout << std::endl << std::endl;
    
    std::cout << "Cумма: " << std::accumulate(h_result.begin(), h_result.end(), 0) << std::endl;


    CUDA_CHECK_CALL(cuMemFree(d_vector1));
    CUDA_CHECK_CALL(cuMemFree(d_vector2));
    CUDA_CHECK_CALL(cuMemFree(d_result));
    CUDA_CHECK_CALL(cuModuleUnload(module));
    CUDA_CHECK_CALL(cuCtxDestroy(context));

    return 0;
}