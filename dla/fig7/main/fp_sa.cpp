#include <cmath>
#include <thread>
#include <atomic>
#include <iostream>
#include <cuda_runtime.h>

#include "utils.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      300
#define TEST_CNT        1000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 4) {
        XINFO("usage: %s <engine dir> <thpt> <cdf>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string engine_dir(argv[1]);
    double thpt = atof(argv[2]);
    int64_t interval_ms = std::round(1000.0 / thpt);
    int64_t variance_ms = interval_ms / 2;
    std::string cdf(argv[3]);

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));
	CudlaModel model(engine_dir);

    srand(time(NULL));
    DataProcessor<int64_t> latency;

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Infer(stream);

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now()
                      + std::chrono::milliseconds(
                            interval_ms - variance_ms + rand() % (2 * variance_ms));
            latency.Add(EXEC_TIME(nanoseconds, { model.Infer(stream); }));
            std::this_thread::sleep_until(next);
        }
    });

    XINFO("[RESULT] sa throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF(cdf);
    return 0;
}
