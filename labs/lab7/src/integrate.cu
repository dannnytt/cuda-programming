#include <iostream>
#include <cuda_runtime.h>
#include <functional>
#include <cmath>

#define CHECK_CUDA_CALL(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

#define WIDTH 512
#define HEIGHT 512
#define DEPTH 512

__constant__ float c_radius;
// __constant__ float c_input[WIDTH * HEIGHT * DEPTH];

__device__ float stepInterpolation(const float* input, int width, int height, int depth, float x, float y, float z) {
    int ix = static_cast<int>(x + 0.5f);
    int iy = static_cast<int>(y + 0.5f);
    int iz = static_cast<int>(z + 0.5f);

    ix = max(0, min(ix, width - 1));
    iy = max(0, min(iy, height - 1));
    iz = max(0, min(iz, depth - 1));

    return input[ix + iy * width + iz * width * height];
}

__device__ float linearInterpolation(const float* input, int width, int height, int depth, float x, float y, float z) {
    int ix = x;
    int iy = y;
    int iz = z;

    float tx = x - ix;
    float ty = y - iy;
    float tz = z - iz;

    ix = max(0, min(ix, width - 1));
    iy = max(0, min(iy, height - 1));
    iz = max(0, min(iz, depth - 1));

    int ix1 = min(ix + 1, width - 1);
    int iy1 = min(iy + 1, height - 1);
    int iz1 = min(iz + 1, depth - 1);

    float c000 = input[ix + iy * width + iz * width * height];
    float c100 = input[ix1 + iy * width + iz * width * height];
    float c010 = input[ix + iy1 * width + iz * width * height];
    float c110 = input[ix1 + iy1 * width + iz * width * height];
    float c001 = input[ix + iy * width + iz1 * width * height];
    float c101 = input[ix1 + iy * width + iz1 * width * height];
    float c011 = input[ix + iy1 * width + iz1 * width * height];
    float c111 = input[ix1 + iy1 * width + iz1 * width * height];

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

__global__ void integrateWithoutTexture(const float* input, float* output, int width, int height, int depth, float radius, bool useLinearInterpolation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        int index = x + y * width + z * width * height;
        float val = useLinearInterpolation ? 
                    linearInterpolation(input, width, height, depth, x, y, z) :
                    stepInterpolation(input, width, height, depth, x, y, z);
        output[index] = val * radius;
    }
}

// __global__ void integrateWithConstantMemory(float* output, int width, int height, int depth, float radius) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;
    
//     if (x < width && y < height && z < depth) {
//         int index = x + y * width + z * width * height;
//         output[index] = c_input[index] * radius;
//     }
// }

__global__ void integrateWithTexture(cudaTextureObject_t texObj, float* output, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < width && y < height && z < depth) {
        float val = tex3D<float>(texObj, x + 0.5f, y + 0.5f, z + 0.5f);
        int index = x + y * width + z * width * height;
        output[index] = val * c_radius;
    }
}

cudaTextureObject_t createTextureObject(float* d_data, int width, int height, int depth, cudaArray_t& d_array) {
    cudaTextureObject_t texObj;
    cudaResourceDesc resDesc = {};
    cudaTextureDesc texDesc = {};

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CHECK_CUDA_CALL(cudaMalloc3DArray(&d_array, &channelDesc, make_cudaExtent(width, height, depth)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(d_data, width * sizeof(float), width, height);
    copyParams.dstArray = d_array;
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;

    CHECK_CUDA_CALL(cudaMemcpy3D(&copyParams));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    CHECK_CUDA_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    return texObj;
}

float sumArray(float* d_array, int size) {
    float* h_array = new float[size];
    CHECK_CUDA_CALL(cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += h_array[i];
    }

    delete[] h_array;
    return sum;
}

float measureKernelTime(std::function<void()> kernelLaunch, int repeats = 10) {
    cudaEvent_t start, stop;
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));
    
    float totalTime = 0.0f;
    for (int i = 0; i < repeats; i++) {
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        CHECK_CUDA_CALL(cudaEventRecord(start));

        kernelLaunch();
        CHECK_CUDA_CALL(cudaGetLastError());

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }

    CHECK_CUDA_CALL(cudaEventDestroy(start));
    CHECK_CUDA_CALL(cudaEventDestroy(stop));

    return totalTime / repeats;
}

int main() {
    CHECK_CUDA_CALL(cudaSetDevice(0));

    int width = WIDTH, height = HEIGHT, depth = DEPTH;
    size_t size = width * height * depth * sizeof(float);

    float *h_input = new float[width * height * depth];
    float *d_input, *d_output;
    CHECK_CUDA_CALL(cudaMalloc(&d_input, size));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, size));

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = (x - width / 2.0f) / (width / 2.0f);
                float ny = (y - height / 2.0f) / (height / 2.0f);
                float nz = (z - depth / 2.0f) / (depth / 2.0f);

                h_input[x + y * width + z * width * height] = nx * nx + ny * ny + nz * nz;
            }
        }
    }
    // CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_input, h_input, size));
    CHECK_CUDA_CALL(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaArray_t d_array;
    cudaTextureObject_t texObj = createTextureObject(d_input, width, height, depth, d_array);

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (depth + blockSize.z - 1) / blockSize.z);

    float radius = 1.0f;
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_radius, &radius, sizeof(float)));

    float avgTimeWithoutTextureStep = measureKernelTime([&]() {
        integrateWithoutTexture<<<gridSize, blockSize>>>(d_input, d_output, width, height, depth, radius, false);
    });
    float integralWithoutTextureStep = sumArray(d_output, width * height * depth);

    float avgTimeWithoutTextureLinear = measureKernelTime([&]() {
        integrateWithoutTexture<<<gridSize, blockSize>>>(d_input, d_output, width, height, depth, radius, true);
    });
    float integralWithoutTextureLinear = sumArray(d_output, width * height * depth);

    // Измерение времени и вычисление интеграла с использованием константной памяти
    // float avgTimeWithConstantMem = measureKernelTime([&]() {
    //     integrateWithConstantMemory<<<gridSize, blockSize>>>(d_output, width, height, depth, radius);
    // });
    // float integralWithConstantMem = sumArray(d_output, width * height * depth);

    float avgTimeWithTexture = measureKernelTime([&]() {
        integrateWithTexture<<<gridSize, blockSize>>>(texObj, d_output, width, height, depth);
    });
    float integralWithTexture = sumArray(d_output, width * height * depth);

    std::cout << "Время без использования текстурной и константной памяти (линейная интерполяция): " << avgTimeWithoutTextureLinear << " ms" << std::endl;
    std::cout << "Интеграл: " << integralWithoutTextureLinear << std::endl << std::endl;
  
    std::cout << "Время без использования текстурной и константной памяти (ступенчатая интерполяция): " << avgTimeWithoutTextureStep << " ms" << std::endl;
    std::cout << "Интеграл: " << integralWithoutTextureStep << std::endl << std::endl;

    // std::cout << "Время с использованием константной памяти: " << avgTimeWithConstantMem << " ms" << std::endl;
    // std::cout << "Интеграл: " << integralWithConstantMem << std::endl << std::endl;

    std::cout << "Время с использованием текстурной и константной памяти: " << avgTimeWithTexture << " ms" << std::endl;
    std::cout << "Интеграл: " << integralWithTexture << std::endl;

    // Освобождение памяти
    delete[] h_input;
    CHECK_CUDA_CALL(cudaFree(d_input));
    CHECK_CUDA_CALL(cudaFree(d_output));
    CHECK_CUDA_CALL(cudaDestroyTextureObject(texObj));
    CHECK_CUDA_CALL(cudaFreeArray(d_array));
    
    return 0;
}