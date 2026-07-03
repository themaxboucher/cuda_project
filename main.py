import numpy as np

SIZE = 64
M = SIZE
N = SIZE
K = SIZE

A = np.ones((M, K))
B = np.ones((K, N))

def matmul_naive(A, B):
    fetches = 0
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    # For each element in the output matrix
    for row in range(M):
        for col in range(N):
            # Compute the dot product
            for k in range(K):
                C[row][col] += A[row][k] * B[k][col]
                fetches += 2
    print(f"Naive: Number of fetches: {fetches}")
    return C

def matmul_rowwise(A, B):
    fetches = 0
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    # For each element in the output matrix
    for row in range(M):
        shared_row = A[row]
        fetches += len(shared_row)
        for col in range(N):
            # Compute the dot product
            for k in range(K):
                C[row][col] += shared_row[k] * B[k][col]
                fetches += 1
    print(f"Rowwise: Number of fetches: {fetches}")
    return C

def matmul_tiled(A, B, TILE_SIZE=4):
    fetches = 0
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N))

    for i in range(0, M, TILE_SIZE):
        for j in range(0, N, TILE_SIZE):
            for k in range(0, K, TILE_SIZE):
                tile_A = A[i:(i + TILE_SIZE), k:(k + TILE_SIZE)]
                tile_B = B[k:(k + TILE_SIZE), j:(j + TILE_SIZE)]
                fetches += tile_A.size + tile_B.size

                C[i:(i + TILE_SIZE), j:(j + TILE_SIZE)] += tile_A @ tile_B

    print(f"Tiled (size {TILE_SIZE}): Number of fetches: {fetches}")
    return C

answer = np.full((M, N), SIZE)

assert np.array_equal(matmul_naive(A, B), answer)
assert np.array_equal(matmul_rowwise(A, B), answer)
for tile_size in [size for size in range(1, SIZE + 1) if SIZE % size == 0]:
    assert np.array_equal(matmul_tiled(A, B, tile_size), answer)