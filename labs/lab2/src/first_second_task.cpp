#include <iostream>
#include <thread>
#include <pthread.h>
#include <vector>
#include <random>
#include <chrono>

const size_t VECTOR_SIZE = 10000000;

std::vector<double> VECTOR_A(VECTOR_SIZE);
std::vector<double> VECTOR_B(VECTOR_SIZE);
std::vector<double> VECTOR_C(VECTOR_SIZE, 0.0f);

struct thread_data {
    size_t start;
    size_t end;
    std::vector<double> *vector_a;
    std::vector<double> *vector_b;
    std::vector<double> *vector_c;
};

void vector_add_cpu() {
    for (size_t i = 0; i < VECTOR_SIZE; i++) {
        VECTOR_C[i] = VECTOR_A[i] + VECTOR_B[i]; 
    }
}

void* vector_add_posix(void* params) {
    thread_data *data = static_cast<thread_data*>(params);
    for (size_t i = data->start; i < data->end; i++) {
        (*data->vector_c)[i] = (*data->vector_a)[i] + (*data->vector_b)[i]; 
    }
    return nullptr;
}

void vector_add_multithreaded(size_t num_threads) {
    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_data> thread_data(num_threads);
    size_t chunk_size = VECTOR_SIZE / num_threads;
    
    for (size_t i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? VECTOR_SIZE : (i + 1) * chunk_size;
        thread_data[i].vector_a = &VECTOR_A;
        thread_data[i].vector_b = &VECTOR_B;
        thread_data[i].vector_c = &VECTOR_C;

        pthread_create(&threads[i], nullptr, vector_add_posix, &thread_data[i]);
    }
        
    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
}

void initialize_random_vectors() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (size_t i = 0; i < VECTOR_SIZE; i++) {
        VECTOR_A[i] = dist(gen);
        VECTOR_B[i] = dist(gen);
    } 
}

int main(int argc, char *argv[]) {
    initialize_random_vectors();

    std::cout << "Vector's size: " << VECTOR_SIZE << std::endl 
                                                << std::endl;

    // Последовательное сложение векторов
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_cpu();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential vector addition CPU Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() 
              << " ms" << std::endl << std::endl;

    
    // Многопоточное сложение векторов
    std::cout << "Multithreaded vector addition CPU Time: " << std::endl;
    size_t n_cores = std::thread::hardware_concurrency();
    for (size_t num_threads = 1; num_threads <= n_cores; ++num_threads) {
        std::fill(VECTOR_C.begin(), VECTOR_C.end(), 0.0f); // Очистить вектор C
        auto start_threaded = std::chrono::high_resolution_clock::now();
        vector_add_multithreaded(num_threads);
        auto stop_threaded = std::chrono::high_resolution_clock::now();
        std::cout << "Time with " << num_threads << " threads: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop_threaded - start_threaded).count() 
                  << " ms" << std::endl;
    }

    return 0;
}