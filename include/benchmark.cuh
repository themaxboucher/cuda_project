#pragma once

enum MatmulType {
    MATMUL_NAIVE = 0,
    MATMUL_TILED = 1,
    MATMUL_CUBLAS = 2,
};

// Times one matmul implementation on M x K times K x N matrices and prints GFLOPS.
void matmul_benchmark(MatmulType type, int M, int N, int K);
