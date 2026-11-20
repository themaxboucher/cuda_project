#include <stdio.h>
#include <stdlib.h>
#include "benchmark.cuh"
#include "cuda_utils.cuh"
#include "kernels.cuh"

#define BENCHMARK_ITERATIONS 100

void matmul_benchmark(MatmulType type, int M, int N, int K) {
    size_t bytes_A = (size_t) M * K * sizeof(float);
    size_t bytes_B = (size_t) K * N * sizeof(float);
    size_t bytes_C = (size_t) M * N * sizeof(float);

    // Allocate memory on the host
    float* host_A = (float*) malloc(bytes_A);
    float* host_B = (float*) malloc(bytes_B);

    // Fill A and B with random values in [-1, 1]
    for (int i = 0; i < M * K; i++) host_A[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    for (int i = 0; i < K * N; i++) host_B[i] = 2.0f * rand() / RAND_MAX - 1.0f;

    // Allocate memory on the device
    float* device_A;
    float* device_B;
    float* device_C;
    CUDA_CHECK(cudaMalloc(&device_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&device_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&device_C, bytes_C));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(device_A, host_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, host_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    double flop_count = 2.0 * M * N * K;

    switch (type) {
        case MATMUL_NAIVE:
            TIME_GPU("Naive", BENCHMARK_ITERATIONS, flop_count, matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, M, N, K));
            break;
        case MATMUL_TILED:
            TIME_GPU("Tiled", BENCHMARK_ITERATIONS, flop_count, matmul_tiled<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, M, N, K));
            break;
        case MATMUL_CUBLAS: {
            // Create the handle outside the timed region so only the matmul is measured
            cublasHandle_t handle;
            CUBLAS_CHECK(cublasCreate(&handle));
            TIME_GPU("cuBLAS", BENCHMARK_ITERATIONS, flop_count, matmul_cublas(handle, device_A, device_B, device_C, M, N, K));
            CUBLAS_CHECK(cublasDestroy(handle));
            break;
        }
    }

    // Free memory
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));
    free(host_A);
    free(host_B);
}
