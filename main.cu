#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

const int N = 1 << 20; // Around 1 million elements

void add_on_cpu(float *x, float *y, float *z, int n)
{
    for (int i = 0; i < n; i++)
    {
        z[i] = x[i] + y[i];
    }
}

__global__ void add_on_gpu(float *x, float *y, float *z, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        z[index] = x[index] + y[index];
    }
}

int main()
{
    float *x, *y, *z, *d_x, *d_y, *d_z;

    x = new float[N];
    y = new float[N];
    z = new float[N];

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = i;
        y[i] = i;
    }

    // CPU computation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_on_cpu(x, y, z, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << diff_cpu.count() << " s\n";

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU computation
    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    add_on_gpu<<<numBlocks, blockSize>>>(d_x, d_y, d_z, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
    std::cout << "GPU time: " << diff_gpu.count() << " s\n";

    cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] x;
    delete[] y;
    delete[] z;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}