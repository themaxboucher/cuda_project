#include "kernels.cuh"
#include "cuda_utils.cuh"

void matmul_cublas(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K) {
    // With cublas<t>gemm C = alpha * A * B + beta * C
    // We want to compute C = A * B
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS stores matrices in column-major order
    // This means we can compute B * A and get the same result as A * B
    // https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transposes
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    ));
}
