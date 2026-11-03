#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    std::printf("Number of CUDA devices: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);

        if (err != cudaSuccess) {
            std::fprintf(stderr, "Device query failed: %s\n",
                         cudaGetErrorString(err));
            return 1;
        }

        std::printf("GPU %d: %s\n", i, prop.name);
        std::printf("Compute capability: %d.%d\n",
                    prop.major, prop.minor);
        std::printf("Global memory: %.2f GiB\n",
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    return 0;
}
