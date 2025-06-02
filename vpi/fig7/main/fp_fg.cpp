#include <cmath>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>

#include "sde.h"
#include "mblur.h"
#include "xsched/utils.h"

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 6) {
        XINFO("usage: %s <device> <media> <cmd cnt> <fg thpt> <cdf>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string device(argv[1]);
    const std::string media(argv[2]);
    const size_t cmd_cnt = atol(argv[3]);
    double thpt = atof(argv[4]);
    std::string cdf(argv[5]);

    ProcessSync psync;
    psync.Sync(2, "fg build waiting");

    int64_t warmup_cnt = 1;
    int64_t test_cnt = 1;
    std::unique_ptr<VpiRunner> runner = nullptr;
    if (device == "pva") {
        warmup_cnt = 20;
        test_cnt = 100;
        runner = std::make_unique<MultiPvaBlurRunner>(media, 8);
    } else if (device == "ofa") {
        warmup_cnt = 50;
        test_cnt = 100;
        runner = std::make_unique<OfaSdeRunner>(media);
    } else {
        XERRO("unsupported device: %s", device.c_str());
    }
    runner->Init();
    
    srand(time(NULL));
    DataProcessor<int64_t> latency;
    const int64_t interval_ms = std::round(1000.0 / thpt);
    const int64_t variance_ms = interval_ms / 2;

    // warm up
    for (int i = 0; i < warmup_cnt; ++i) runner->Execute(cmd_cnt, false);
    psync.Sync(4, "fg warmup ready");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < test_cnt; ++i) {
            auto next = std::chrono::system_clock::now()
                      + std::chrono::milliseconds(
                            interval_ms - variance_ms + rand() % (2 * variance_ms));
            latency.Add(EXEC_TIME(nanoseconds, { runner->Execute(cmd_cnt, false); }));
            std::this_thread::sleep_until(next);
        }
    });

    psync.Sync(5, "fg");
    XINFO("[RESULT] fg throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)test_cnt * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    latency.SaveCDF(cdf);

    runner->Final();
}
