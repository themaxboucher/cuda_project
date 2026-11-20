#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Wrap every CUDA runtime call so failures stop the program with a clear
// message instead of silently producing garbage.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int) status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel launches don't return an error code, so check for one right after.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Times the code passed in __VA_ARGS__ on the GPU and prints GFLOPS.
#define TIME_GPU(label, iterations, flop_count, ...) \
    do { \
        cudaEvent_t start, stop; \
        CUDA_CHECK(cudaEventCreate(&start)); \
        CUDA_CHECK(cudaEventCreate(&stop)); \
        __VA_ARGS__; /* Warmup. The first run will be slower. */ \
        CUDA_CHECK_LAUNCH(); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaEventRecord(start)); \
        for (int i = 0; i < iterations; i++) { __VA_ARGS__; } \
        CUDA_CHECK(cudaEventRecord(stop)); \
        CUDA_CHECK(cudaEventSynchronize(stop)); \
        float milliseconds = 0; \
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop)); \
        CUDA_CHECK(cudaEventDestroy(start)); \
        CUDA_CHECK(cudaEventDestroy(stop)); \
        float ms_per_iter = milliseconds / iterations; \
        double gflops = flop_count / (ms_per_iter * 1e-3) / 1e9; \
        printf("%-14s %10.2f GFLOPS | %8.4f ms (avg over %d iterations)\n", label, gflops, ms_per_iter, iterations); \
    } while (0)
