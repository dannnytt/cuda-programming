#include <cuda.h>
#include <iostream>

int main() {
    int driverVersion;
    cuDriverGetVersion(&driverVersion);
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
    return 0;
}