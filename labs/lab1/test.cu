#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void my_kernel() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    my_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}