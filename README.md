# CUDA Matrix Multiplication

Hand-written CUDA matrix multiplication kernels, each one faster than the last, benchmarked against NVIDIA's cuBLAS.

## Results

FP32, 1024 × 1024 × 1024, average of 100 runs on an NVIDIA _\<GPU\>_.

| Kernel                | GFLOPS | % of cuBLAS |
| --------------------- | ------ | ----------- |
| Naive                 | _TBD_  | _TBD_       |
| Tiled (shared memory) | _TBD_  | _TBD_       |
| cuBLAS                | _TBD_  | 100%        |

Performance is measured in GFLOPS (billions of floating-point operations per second). A matrix multiplication of an M × K matrix by a K × N matrix takes `2 · M · N · K` operations: one multiply and one add per step of each dot product.

## Quick start

You need an NVIDIA GPU and the CUDA toolkit. If you don't have one, open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) in Google Colab and use the free T4 GPU runtime.

```bash
make test   # check every kernel against a CPU reference
make run    # run the vector add demo and the matmul benchmark
```

## How it works

### The CUDA hierarchy

Grid → Block → Thread

A kernel is a function that thousands of GPU threads run at the same time. Threads are grouped into blocks, and blocks make up the grid. Each thread works out which piece of data it owns from its block and thread index:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### Vector addition

Vector addition is "embarrassingly parallel": each operation is independent of the others, so each thread adds one pair of elements.

$\begin{bmatrix} a \\ b \end{bmatrix} + \begin{bmatrix} c \\ d \end{bmatrix} = \begin{bmatrix} a + c \\ b + d \end{bmatrix}$

### Naive matrix multiplication

[`matmul_naive.cu`](src/kernels/matmul_naive.cu) uses a 2D grid where each thread computes one element of C. To do it, the thread reads a full row of A and a full column of B from **global memory**. Global memory is large but slow, and neighbouring threads read the same values over and over.

### Tiled matrix multiplication

[`matmul_tiled.cu`](src/kernels/matmul_tiled.cu) uses **shared memory**, a small and fast memory that all threads in a block can see. The block walks through the K dimension one 16 × 16 tile at a time:

1. **Load:** each thread copies one element of A and one of B into shared memory.
2. **Sync:** `__syncthreads()` waits until the whole tile is loaded.
3. **Compute:** each thread does 16 multiply-adds using only shared memory.
4. **Sync** again before the next tile overwrites this one.

Each value now comes from global memory once per block instead of once per thread, which cuts global reads by a factor of `TILE_SIZE`. [`fetch_analysis.py`](python/fetch_analysis.py) shows the same effect by counting fetches in plain Python.

### cuBLAS

[`matmul_cublas.cu`](src/kernels/matmul_cublas.cu) calls NVIDIA's own library as the target to beat.

## Next steps

- Register tiling: each thread computes several outputs
- Vectorized `float4` loads
- Profiling each kernel with Nsight Compute
