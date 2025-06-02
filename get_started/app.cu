#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <random>

#define VECTOR_SIZE (1 << 25) // 32MB
#define N 100 // Number of vector additions per task
#define M 10000  // Number of tasks

// Global memory pointers
float *h_A, *h_B, *h_C;
float *d_A, *d_B, *d_C;

cudaStream_t stream;

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void prepareTask() {
    size_t size = VECTOR_SIZE * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);
}

void runTask(int taskId) {
    // Launch kernel N times
    int threadsPerBlock = 256;
    int blocksPerGrid = (VECTOR_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int i = 0; i < N; ++i) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, VECTOR_SIZE);
    }
    cudaStreamSynchronize(stream);
}

void cleanupTask() {
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(30, 50);

    // Prepare task (memory allocation)
    prepareTask();

    // Run tasks
    for (int i = 0; i < M; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        runTask(i);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Task %d completed in %ld ms\n", i, duration.count());
        
        // Sleep for random interval between tasks
        if (i < M - 1) {
            int sleepTime = dis(gen);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        }
    }

    // Cleanup
    cleanupTask();

    return 0;
}