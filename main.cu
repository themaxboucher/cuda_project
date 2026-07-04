#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define TILE_SIZE 16

#define TIME_GPU(label, iterations, ...) \
    do { \
        cudaEvent_t start, stop; \
        cudaEventCreate(&start); \
        cudaEventCreate(&stop); \
        __VA_ARGS__; /* Warmup. The first run will be slower. */ \
        cudaDeviceSynchronize(); \
        cudaEventRecord(start); \
        for (int i = 0; i < iterations; i++) { __VA_ARGS__; } \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        float milliseconds = 0; \
        cudaEventElapsedTime(&milliseconds, start, stop); \
        printf("%s: %f ms (avg over %d iterations)\n", label, milliseconds / iterations, iterations); \
        cudaEventDestroy(start); \
        cudaEventDestroy(stop); \
    } while (0)


__global__ void vector_add_kernel(int* c, const int* a, const int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}


__global__ void matmul_naive_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


__global__ void matmul_tiled_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++) {
        // 1. LOAD
        a_tile[ty][tx] = (row < M && phase * TILE_SIZE + tx < K) ? A[row * K + phase * TILE_SIZE + tx] : 0.0f;
        b_tile[ty][tx] = (col < N && phase * TILE_SIZE + ty < K) ? B[(phase * TILE_SIZE + ty) * N + col] : 0.0f;
        // 2. SYNC
        __syncthreads();
        // 3. COMPUTE
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


void matmul_cublas(float* C, const float* A, const float* B, int M, int N, int K) {
    // Create a CuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // With cublas<t>gemm C = alpha * A * B + beta * C
    // We want to compute C = A * B
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS stores matrices in column-major order
    // This means we can compute B * A and get the same result as A * B
    // https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transposes
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );
    
    // Clean up the handle
    cublasDestroy(handle);
}


void vector_add_example() {
    int size = 1000000;
    int bytes = size * sizeof(int);

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
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    // Copy data from host to device
    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_c, device_a, device_b, size);

    // Copy data from device to host
    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Vector addition result: ");
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(host_a);
    free(host_b);
    free(host_c);
}


void matmul_example(int type = 0) {
    int M = 1024; // Rows of A
    int N = 1024; // Columns of B
    int K = 1024; // Columns of A and rows of B
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate memory on the host
    float* host_A = (float*) malloc(bytes_A);
    float* host_B = (float*) malloc(bytes_B);
    float* host_C = (float*) malloc(bytes_C);

    // Set all the values of A and B to float 1.0
    for (int i = 0; i < M * K; i++) host_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) host_B[i] = 1.0f;
    
    // Allocate memory on the device
    float* device_A;
    float* device_B;
    float* device_C;
    cudaMalloc(&device_A, bytes_A);
    cudaMalloc(&device_B, bytes_B);
    cudaMalloc(&device_C, bytes_C);

    // Copy data from host to device
    cudaMemcpy(device_A, host_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    if (type == 0) {
        TIME_GPU("Naive kernel", 100, matmul_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_C, device_A, device_B, M, N, K));
    } else if (type == 1) {
        TIME_GPU("Tiled kernel", 100, matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_C, device_A, device_B, M, N, K));
    } else {
        TIME_GPU("CuBLAS", 100, matmul_cublas(device_C, device_A, device_B, M, N, K));
    }

    // Copy data from device to host
    cudaMemcpy(host_C, device_C, bytes_C, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Matrix multiplication result: ");
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%f ", host_C[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);
}


int main() {
    vector_add_example();
    matmul_example(0); // Naive kernel
    matmul_example(1); // Tiled kernel
    matmul_example(2); // CuBLAS

    return 0;
}