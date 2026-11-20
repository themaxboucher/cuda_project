// Checks every GPU matmul against a simple CPU version on random inputs.
// The sizes include ones that aren't multiples of TILE_SIZE, so the bounds
// checks and zero padding get exercised too.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.cuh"
#include "kernels.cuh"


void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double) A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = (float) sum;
        }
    }
}


// Returns true when every element of actual is close to expected.
bool all_close(const float* expected, const float* actual, int count, int K) {
    // Float rounding error grows with the length of each dot product
    float tolerance = 1e-5f * K + 1e-5f;
    for (int i = 0; i < count; i++) {
        // Written as !(diff <= tolerance) so NaN counts as a mismatch
        if (!(fabsf(expected[i] - actual[i]) <= tolerance)) {
            printf("    mismatch at index %d: expected %f, got %f\n", i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


// Runs all GPU implementations on one size and returns the number of failures.
int test_size(int M, int N, int K, cublasHandle_t handle) {
    size_t bytes_A = (size_t) M * K * sizeof(float);
    size_t bytes_B = (size_t) K * N * sizeof(float);
    size_t bytes_C = (size_t) M * N * sizeof(float);

    float* host_A = (float*) malloc(bytes_A);
    float* host_B = (float*) malloc(bytes_B);
    float* expected = (float*) malloc(bytes_C);
    float* actual = (float*) malloc(bytes_C);

    for (int i = 0; i < M * K; i++) host_A[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    for (int i = 0; i < K * N; i++) host_B[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    matmul_cpu(host_A, host_B, expected, M, N, K);

    float* device_A;
    float* device_B;
    float* device_C;
    CUDA_CHECK(cudaMalloc(&device_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&device_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&device_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(device_A, host_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_B, host_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    const char* names[] = {"naive", "tiled", "cuBLAS"};
    int failures = 0;

    for (int type = 0; type < 3; type++) {
        // Fill C with garbage so a kernel that skips elements can't pass by accident
        CUDA_CHECK(cudaMemset(device_C, 0xFF, bytes_C));

        if (type == 0) {
            matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, M, N, K);
        } else if (type == 1) {
            matmul_tiled<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, M, N, K);
        } else {
            matmul_cublas(handle, device_A, device_B, device_C, M, N, K);
        }
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(actual, device_C, bytes_C, cudaMemcpyDeviceToHost));

        bool passed = all_close(expected, actual, M * N, K);
        printf("  [%s] %-6s %4d x %4d x %4d\n", passed ? "PASS" : "FAIL", names[type], M, N, K);
        if (!passed) failures++;
    }

    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));
    free(host_A);
    free(host_B);
    free(expected);
    free(actual);
    return failures;
}


int main() {
    srand(42);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // {M, N, K}: square, non-square, and sizes that don't divide evenly into tiles
    int sizes[][3] = {
        {16, 16, 16},
        {64, 64, 64},
        {1, 1, 1},
        {17, 33, 5},
        {100, 70, 130},
        {256, 128, 512},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Matmul correctness tests:\n");
    int failures = 0;
    for (int i = 0; i < num_sizes; i++) {
        failures += test_size(sizes[i][0], sizes[i][1], sizes[i][2], handle);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    if (failures > 0) {
        printf("%d test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    printf("All tests passed\n");
    return EXIT_SUCCESS;
}
