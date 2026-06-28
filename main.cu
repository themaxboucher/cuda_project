#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int* c, const int* a, const int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
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
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(device_c, device_a, device_b, size);

    // Copy data from device to host
    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(host_a);
    free(host_b);
    free(host_c);

    // Print the result
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }

    return 0;
}