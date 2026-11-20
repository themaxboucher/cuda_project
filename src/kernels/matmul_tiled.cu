#include "kernels.cuh"

// Each block loads TILE_SIZE x TILE_SIZE tiles of A and B into shared memory,
// so every value is read from global memory once per block instead of once
// per thread.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++) {
        // 1. LOAD one element of each tile per thread (0 past the matrix edge)
        a_tile[ty][tx] = (row < M && phase * TILE_SIZE + tx < K) ? A[row * K + phase * TILE_SIZE + tx] : 0.0f;
        b_tile[ty][tx] = (col < N && phase * TILE_SIZE + ty < K) ? B[(phase * TILE_SIZE + ty) * N + col] : 0.0f;
        // 2. SYNC so the whole tile is loaded before anyone reads it
        __syncthreads();
        // 3. COMPUTE from fast shared memory
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += a_tile[ty][i] * b_tile[i][tx];
        }
        // 4. SYNC so nobody overwrites the tile while others still read it
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
