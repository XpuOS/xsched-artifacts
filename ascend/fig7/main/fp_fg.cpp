#include <cmath>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>

#include "utils.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      1000
#define TEST_CNT        1000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 4) {
        XINFO("usage: %s <model dir> <fg thpt> <cdf>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_dir(argv[1]);
    double thpt = atof(argv[2]);
    std::string cdf(argv[3]);

    aclrtStream stream;
    AclModel model(model_dir);
    ACL_ASSERT(aclrtCreateStream(&stream));

    ProcessSync psync;
    psync.Sync(2, "fg build");
    
    srand(time(NULL));
    DataProcessor<int64_t> latency;
    const int64_t interval_ms = std::round(1000.0 / thpt);
    const int64_t variance_ms = interval_ms / 2;

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Execute(stream);
    psync.Sync(4, "fg warmup ready");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now()
                      + std::chrono::milliseconds(
                            interval_ms - variance_ms + rand() % (2 * variance_ms));
            latency.Add(EXEC_TIME(nanoseconds, { model.Execute(stream); }));
            std::this_thread::sleep_until(next);
        }
    });

    psync.Sync(5, "fg");
    XINFO("[RESULT] fg throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    latency.SaveCDF(cdf);

    ACL_ASSERT(aclrtDestroyStream(stream));
}
