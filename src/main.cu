#include <stdio.h>
#include <stdlib.h>
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include "kernels.cuh"


void vector_add_example() {
    int size = 1000000;
    size_t bytes = size * sizeof(int);

    // Allocate memory on the host
    int* host_a = (int*) malloc(bytes);
    int* host_b = (int*) malloc(bytes);
    int* host_c = (int*) malloc(bytes);

    for (int i = 0; i < size; i++) {
        host_a[i] = i;
        host_b[i] = i * 2;
    }

    // Allocate memory on the device
    int* device_a;
    int* device_b;
    int* device_c;
    CUDA_CHECK(cudaMalloc(&device_a, bytes));
    CUDA_CHECK(cudaMalloc(&device_b, bytes));
    CUDA_CHECK(cudaMalloc(&device_c, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, size);
    CUDA_CHECK_LAUNCH();

    // Copy data from device to host (this also waits for the kernel to finish)
    CUDA_CHECK(cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost));

    // Print the result
    printf("Vector addition, first 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }
    printf("\n");

    // Free memory
    CUDA_CHECK(cudaFree(device_a));
    CUDA_CHECK(cudaFree(device_b));
    CUDA_CHECK(cudaFree(device_c));
    free(host_a);
    free(host_b);
    free(host_c);
}


int main() {
    vector_add_example();

    int M = 1024; // Rows of A
    int N = 1024; // Columns of B
    int K = 1024; // Columns of A and rows of B
    printf("Matrix multiplication benchmark (%d x %d x %d):\n", M, N, K);
    matmul_benchmark(MATMUL_NAIVE, M, N, K);
    matmul_benchmark(MATMUL_TILED, M, N, K);
    matmul_benchmark(MATMUL_CUBLAS, M, N, K);

    return 0;
}
