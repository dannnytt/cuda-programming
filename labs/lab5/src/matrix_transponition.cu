#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK_RETURN(value) { \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) { \
        fprintf(stderr, "Ошибка %s в строке %d в файле %s\n", \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
        exit(1); \
    } \
}

void fill_matrix(vector<int> &matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * columns + j] = (rand() % 100) + 1;
        }
    }
}

__global__ void matrix_transponition(int *A, int *B, int rows, int columns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < columns) {
        B[col * rows + row] = A[row * columns + col];
    }
}

// __global__ void matrix_transponition(int *A, int *B, int rows, int columns) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     // Эмулируем недостаток регистров
//     // Искусственно создаём большой массив
//     float local_data[512];  

//     if (row < rows && col < columns) {
//         local_data[threadIdx.x] = A[row * columns + col];
//         B[col * rows + row] = local_data[threadIdx.x];
//     }
// }


void print_matrix(vector<int> matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            cout << matrix[i * columns + j] << " ";
        }
        cout << endl;
    }
}

void process_matrix_transponition() {
    srand(time(nullptr));
    int rows = 3;
    int columns = 4;

    vector<int> h_matrix(rows * columns);
    vector<int> h_matrix2(columns * rows, -1);

    fill_matrix(h_matrix, rows, columns);

    cout << "Исходная матрица:\n";
    print_matrix(h_matrix, rows, columns);

    int *d_vector1, *d_vector2;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector1, rows * columns * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector2, columns * rows * sizeof(int))); 

    CUDA_CHECK_RETURN(cudaMemcpy(d_vector1, h_matrix.data(), rows * columns * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((columns + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    matrix_transponition<<<gridSize, blockSize>>>(d_vector1, d_vector2, rows, columns);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(h_matrix2.data(), d_vector2, columns * rows * sizeof(int), cudaMemcpyDeviceToHost));

    cout << "Транспонированная матрица:\n";
    print_matrix(h_matrix2, columns, rows);

    cudaFree(d_vector1);
    cudaFree(d_vector2);
}

int main() {
    process_matrix_transponition();
    return 0;
}
