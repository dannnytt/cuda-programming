import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda

from pycuda import gpuarray
from pycuda.driver import Device
from pycuda.compiler import SourceModule



cuda_code = """
    __global__ void matrix_multiplication(float *vec1, float *vec2, float *res, int size) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += vec1[row * size + i] * vec2[i * size + col];
        }
        res[row * size + col] = sum;
    }
}
"""

def process_matrix_multiplication():
    SIZE = 1024
    h_vec1 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_vec2 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_res = np.zeros(SIZE * SIZE, dtype=np.float32)

    d_vec1 = gpuarray.to_gpu(h_vec1)
    d_vec2 = gpuarray.to_gpu(h_vec2)
    d_res = gpuarray.zeros_like(d_vec1)

    module = SourceModule(
        cuda_code,
        options=['-arch=sm_75']
    )

    matrix_multiplication = module.get_function("matrix_multiplication")

    threads_per_block = (16, 16, 1)
    num_blocks = (
        (SIZE + threads_per_block[0] - 1) // threads_per_block[0],
        (SIZE + threads_per_block[1] - 1) // threads_per_block[1],
        1
    )

    matrix_multiplication(
        d_vec1,
        d_vec2,
        d_res,
        np.int32(SIZE),
        block=threads_per_block,
        grid=num_blocks
    )
    cuda.Context.synchronize()
    
    h_res = d_res.get()



if __name__ == "__main__":
    process_matrix_multiplication()
    
