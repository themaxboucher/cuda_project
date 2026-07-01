import numpy as np

M = 4
N = 4
K = 4

A = np.ones((M, K))
B = np.ones((K, N))

def matmul_naive(A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    # For each element in the output matrix
    for row in range(M):
        for col in range(N):
            # Compute the dot product
            for k in range(K):
                C[row][col] += A[row][k] * B[k][col]
    return C

def matmul_tiled(A, B, TILE_SIZE=2):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    # For each element in the output matrix
    for row in range(M):
        for col in range(N):
            for phase in range(0, K, TILE_SIZE):
                # Compute a partial sum using a small "tile" of the matrices
                for i in range(TILE_SIZE):
                    k = phase + i
                    if k < K: # Boundary check
                        C[row][col] += A[row][k] * B[k][col]
    return C