#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 8192
#define COLS 8192
#define TILE_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    }}

__global__ void transpose(double *d_in, double *d_out, uint width, uint height) {

    __shared__ double tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        d_out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
      

void measure_time(double *d_in, double *d_out, uint width, uint height) {
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEIL_DIV(width, TILE_SIZE), CEIL_DIV(height, TILE_SIZE));

    CUDA_CHECK_RETURN(cudaEventRecord(start));
    transpose<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK_RETURN(cudaEventRecord(stop));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Время выполнения: %f ms\n", milliseconds);

    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));
}

void matrix_init(double *matrix, uint rows, uint cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)(rand() % 1000) / 100.0;
    }
}

int main(int argc, char* argv[]) {

    double *h_in, *h_out;
    double *d_in, *d_out;
    size_t size = ROWS * COLS * sizeof(double);
    
    h_in = (double *)malloc(size);
    h_out = (double *)malloc(size);

    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_out, size));

    matrix_init(h_in, ROWS, COLS);
    CUDA_CHECK_RETURN(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    measure_time(d_in, d_out, ROWS, COLS);
    CUDA_CHECK_RETURN(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    free(h_in);
    free(h_out);
    CUDA_CHECK_RETURN(cudaFree(d_in));
    CUDA_CHECK_RETURN(cudaFree(d_out));
    return 0;
}
