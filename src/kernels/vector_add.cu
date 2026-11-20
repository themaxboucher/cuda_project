#include "kernels.cuh"

// Each thread adds one pair of elements.
__global__ void vector_add(const int* a, const int* b, int* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
