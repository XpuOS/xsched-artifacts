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
    if (argc < 4) {
        XINFO("usage: %s <device> <media> <cmd cnt>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string device(argv[1]);
    const std::string media(argv[2]);
    const size_t cmd_cnt = atol(argv[3]);

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

    ProcessSync psync;
    psync.Sync(2, "bg build");

    // warm up
    for (int i = 0; i < warmup_cnt; ++i) runner->Execute(cmd_cnt, false);
    psync.Sync(4, "bg warmup ready");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            runner->Execute(cmd_cnt, false);
            infer_cnt++;
        }
    });
    double bg_thpt = (double)infer_cnt * 1e9 / ns;

    runner->Final();
}
