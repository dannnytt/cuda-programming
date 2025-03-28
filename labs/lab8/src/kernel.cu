#include <cuda_runtime.h>
#include <stdio.h>


extern "C" __global__ void kernel(double *vector1, double *vector2, double *result, int size) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size)
        result[idx] = vector1[idx] + vector2[idx];
}