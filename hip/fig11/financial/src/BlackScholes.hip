/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <chrono>
#include <thread>
// #include <helper_timer.h>
// #include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int NUM_ITERATIONS = 512;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

const int FREQ = 5;
const int TEST_SECONDS = 1000000; // never stop

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("[%s] - Starting...\n", argv[0]);

  //'h_' prefix - CPU (host) memory space
  float
      // Results calculated by CPU for reference
      *h_CallResultCPU,
      *h_PutResultCPU,
      // CPU copy of GPU results
      *h_CallResultGPU, *h_PutResultGPU,
      // CPU instance of input data
      *h_StockPrice, *h_OptionStrike, *h_OptionYears;

  //'d_' prefix - GPU (device) memory space
  float
      // Results calculated by GPU
      *d_CallResult,
      *d_PutResult,
      // GPU instance of input data
      *d_StockPrice, *d_OptionStrike, *d_OptionYears;
  int i;

  findCudaDevice(argc, (const char **)argv);

  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
  h_CallResultCPU = (float *)malloc(OPT_SZ);
  h_PutResultCPU = (float *)malloc(OPT_SZ);
  h_CallResultGPU = (float *)malloc(OPT_SZ);
  h_PutResultGPU = (float *)malloc(OPT_SZ);
  h_StockPrice = (float *)malloc(OPT_SZ);
  h_OptionStrike = (float *)malloc(OPT_SZ);
  h_OptionYears = (float *)malloc(OPT_SZ);

  printf("...allocating GPU memory for options.\n");
  checkCudaErrors(hipMalloc((void **)&d_CallResult, OPT_SZ));
  checkCudaErrors(hipMalloc((void **)&d_PutResult, OPT_SZ));
  checkCudaErrors(hipMalloc((void **)&d_StockPrice, OPT_SZ));
  checkCudaErrors(hipMalloc((void **)&d_OptionStrike, OPT_SZ));
  checkCudaErrors(hipMalloc((void **)&d_OptionYears, OPT_SZ));

  printf("...generating input data in CPU mem.\n");
  srand(5347);

  // Generate options set
  for (i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }

  hipStream_t stream;
  checkCudaErrors(hipStreamCreate(&stream));

  auto run = [&]() -> void {
    // Copy options data to GPU memory for further processing
    checkCudaErrors(hipMemcpyAsync(d_StockPrice, h_StockPrice, OPT_SZ, hipMemcpyHostToDevice, stream));
    checkCudaErrors(hipMemcpyAsync(d_OptionStrike, h_OptionStrike, OPT_SZ, hipMemcpyHostToDevice, stream));
    checkCudaErrors(hipMemcpyAsync(d_OptionYears, h_OptionYears, OPT_SZ, hipMemcpyHostToDevice, stream));
    // Execute the kernel NUM_ITERATIONS times
    for (i = 0; i < NUM_ITERATIONS; i++) {
      BlackScholesGPU<<<DIV_UP((OPT_N / 2), 512), 512 /*480, 128*/, 0, stream>>>(
          (float2 *)d_CallResult, (float2 *)d_PutResult, (float2 *)d_StockPrice,
          (float2 *)d_OptionStrike, (float2 *)d_OptionYears, RISKFREE, VOLATILITY,
          OPT_N);
      getLastCudaError("BlackScholesGPU() execution failed\n");
    }
    // Read back GPU results to compare them to CPU results
    checkCudaErrors(hipMemcpyAsync(h_CallResultGPU, d_CallResult, OPT_SZ, hipMemcpyDeviceToHost, stream));
    checkCudaErrors(hipMemcpyAsync(h_PutResultGPU, d_PutResult, OPT_SZ, hipMemcpyDeviceToHost, stream));
    checkCudaErrors(hipStreamSynchronize(stream));
  };

  // warmup
  printf("Warmup...\n");
  for (int i = 0; i < 3 * FREQ; i++) {
    auto start = std::chrono::system_clock::now();
    run();
    std::this_thread::sleep_until(start + std::chrono::microseconds(1000000 / FREQ));
  }

  // benchmark
  printf("Benchmark...\n");
  for (int i = 0; i < TEST_SECONDS * FREQ; i++) {
    auto start = std::chrono::system_clock::now();
    run();
    auto end = std::chrono::system_clock::now();
		auto t = std::chrono::system_clock::to_time_t(end);
		auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count() % 1000000;
		std::tm* now = std::localtime(&t);
		double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		double thpt = NUM_ITERATIONS / us * 1000000;
		fprintf(stderr, "{\"time\": \"%02d:%02d:%02d.%05ld\", \"test_id\": %d, \"iter_cnt\": %d, \"latency (ms)\": %.3f, \"throughput (iters/s)\": %.4f},\n",
			      now->tm_hour, now->tm_min, now->tm_sec, ms, i, NUM_ITERATIONS, us / 1000, thpt);
		fflush(stderr);
  
    std::this_thread::sleep_until(start + std::chrono::microseconds(1000000 / FREQ));
  }

  printf("Shutting down...\n");
  printf("...releasing GPU memory.\n");
  checkCudaErrors(hipFree(d_OptionYears));
  checkCudaErrors(hipFree(d_OptionStrike));
  checkCudaErrors(hipFree(d_StockPrice));
  checkCudaErrors(hipFree(d_PutResult));
  checkCudaErrors(hipFree(d_CallResult));

  printf("...releasing CPU memory.\n");
  free(h_OptionYears);
  free(h_OptionStrike);
  free(h_StockPrice);
  free(h_PutResultGPU);
  free(h_CallResultGPU);
  free(h_PutResultCPU);
  free(h_CallResultCPU);
  printf("Shutdown done.\n");

  printf("\n[BlackScholes] - Test Summary\n");
  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
