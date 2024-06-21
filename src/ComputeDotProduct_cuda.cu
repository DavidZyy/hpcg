
#include <cstdio>
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

#include <cassert>
#include <cuda_runtime_api.h>
#include "ComputeDotProduct_cuda.hpp"

__global__ void dotProductKernel(const double* x, const double* y, double* partialResults, local_int_t n) {
    extern __shared__ double sdata[];

    // Calculate thread index
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    sdata[tid] = 0.0;

    // Perform partial dot product
    if (i < n) {
        sdata[tid] = x[i] * y[i];
    }
    __syncthreads();

    // Parallel reduction within the block
    // blockDim is 256
    // one block's(256) result is reduced to sdata[0]
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to the partial results array
    if (tid == 0) {
        partialResults[blockIdx.x] = sdata[0];
        // printf("blockIdx %d, partialResults[blockIdx.x] %f\n", blockIdx.x, partialResults[blockIdx.x]);
    }
}

int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {
    assert(x.localLength >= n); // Test vector lengths
    assert(y.localLength >= n);

    // Allocate device memory
    double* d_x;
    double* d_y;
    double* d_partialResults;
    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_y, n * sizeof(double));

    // Copy vectors to device
    cudaMemcpy(d_x, x.values, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.values, n * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256; // You can adjust this value
    // int blockSize = 1024; // You can adjust this value
    // int blockSize = 2048; // You can adjust this value
    int gridSize = (n + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_partialResults, gridSize * sizeof(double));

    // Launch kernel
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_x, d_y, d_partialResults, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy partial results back to host
    double* h_partialResults = new double[gridSize];
    cudaMemcpy(h_partialResults, d_partialResults, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Final reduction on host
    double local_result = 0.0;
    for (int i = 0; i < gridSize; ++i) {
        local_result += h_partialResults[i];
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialResults);
    delete[] h_partialResults;

#ifndef HPCG_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    double t0 = mytimer();
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result = global_result;
    time_allreduce += mytimer() - t0;
#else
    time_allreduce += 0.0;
    result = local_result;
#endif

    return 0;
}

// __global__ void dotProductKernel(const double* x, const double* y, double* partialResults, local_int_t n) {
//     extern __shared__ double sdata[];
// 
//     // Calculate thread index
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
// 
//     // Initialize shared memory
//     sdata[tid] = 0.0;
// 
//     // Perform partial dot product
//     if (i < n) {
//         sdata[tid] = x[i] * y[i];
//     }
//     __syncthreads();
// 
//     // Parallel reduction within the block
//     // blockDim is 256
//     // one block's(256) result is reduced to sdata[0]
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }
// 
//     // Write the result for this block to the partial results array
//     if (tid == 0) {
//         partialResults[blockIdx.x] = sdata[0];
//         // print the address of sdata[0] and blockIdx.x
//         printf("sdata[0] address: %p ", &sdata[0]);
//         // printf("blockIdx.x: %d\n", blockIdx.x);
//         // printf("blockIdx %d, partialResults[blockIdx.x] %f\n", blockIdx.x, partialResults[blockIdx.x]);
//     }
// }
// 
// int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
//     double & result, double & time_allreduce) {
//     assert(x.localLength >= n); // Test vector lengths
//     assert(y.localLength >= n);
// 
//     // Allocate device memory
//     double* d_x;
//     double* d_y;
//     double* d_partialResults;
//     cudaMalloc((void**)&d_x, n * sizeof(double));
//     cudaMalloc((void**)&d_y, n * sizeof(double));
// 
//     // Copy vectors to device
//     cudaMemcpy(d_x, x.values, n * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, y.values, n * sizeof(double), cudaMemcpyHostToDevice);
// 
//     // Calculate grid and block sizes
//     int blockSize = 256; // You can adjust this value
//     // int blockSize = 1024; // You can adjust this value
//     // int blockSize = 2048; // You can adjust this value
//     int gridSize = (n + blockSize - 1) / blockSize;
//     cudaMalloc((void**)&d_partialResults, gridSize * sizeof(double));
// 
//     // Launch kernel
//     dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_x, d_y, d_partialResults, n);
// 
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     }
//     // Copy partial results back to host
//     double* h_partialResults = new double[gridSize];
//     cudaMemcpy(h_partialResults, d_partialResults, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
// 
//     // Final reduction on host
//     double local_result = 0.0;
//     for (int i = 0; i < gridSize; ++i) {
//         local_result += h_partialResults[i];
//     }
// 
//     // Free device memory
//     cudaFree(d_x);
//     cudaFree(d_y);
//     cudaFree(d_partialResults);
//     delete[] h_partialResults;
// 
// #ifndef HPCG_NO_MPI
//     // Use MPI's reduce function to collect all partial sums
//     double t0 = mytimer();
//     double global_result = 0.0;
//     MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//     result = global_result;
//     time_allreduce += mytimer() - t0;
// #else
//     time_allreduce += 0.0;
//     result = local_result;
// #endif
// 
//     return 0;
// }


