import numpy as np

from numba import cuda

@cuda.jit
def matrix_multiplication(vec1, vec2, res, size):    
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (row < size and col < size):
        sum = 0.0
        for i in range(size):
            sum += vec1[row * size + i] * vec2[i * size + col]
        res[row * size + col] = sum


def process_matrix_multiplication():
    SIZE = 1024
    h_vec1 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_vec2 = np.random.randn(SIZE * SIZE).astype(np.float32)
    h_res = np.zeros(SIZE * SIZE, dtype=np.float32)

    d_vec1 = cuda.to_device(h_vec1)
    d_vec2 = cuda.to_device(h_vec2)
    d_res = cuda.device_array_like(h_res)

    threads_per_block = (16, 16)
    num_blocks = (
        SIZE + threads_per_block[0] // threads_per_block[0],
        SIZE + threads_per_block[1] // threads_per_block[1]
    )

    matrix_multiplication[num_blocks, threads_per_block](d_vec1, d_vec2, d_res, SIZE)
    cuda.synchronize()

    h_res = d_res.copy_to_host()



if __name__ == "__main__":
    process_matrix_multiplication()
    pass