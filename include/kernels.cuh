#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// The tiled kernel must be launched with TILE_SIZE x TILE_SIZE threads per block.
#define TILE_SIZE 16

// All functions take their inputs first and write to the output last.
// Matrices are row-major: A is M x K, B is K x N, C is M x N.

__global__ void vector_add(const int* a, const int* b, int* c, int size);

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K);

// Runs on the host and calls cuBLAS. Create the handle once and reuse it.
void matmul_cublas(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K);
