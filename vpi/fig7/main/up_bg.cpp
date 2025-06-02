#include <fstream>
#include <iostream>

#include "sde.h"
#include "mblur.h"
#include "xsched/utils.h"

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 4) {
        XINFO("usage: %s <device> <media> <cmd cnt> <out>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string device(argv[1]);
    const std::string media(argv[2]);
    const size_t cmd_cnt = atol(argv[3]);
    std::string out(argv[4]);

    int64_t warmup_cnt = 1;
    int64_t test_cnt = 1;
    std::unique_ptr<VpiRunner> runner = nullptr;
    if (device == "pva") {
        warmup_cnt = 40;
        test_cnt = 100;
        runner = std::make_unique<MultiPvaBlurRunner>(media, 8);
    } else if (device == "ofa") {
        warmup_cnt = 50;
        test_cnt = 200;
        runner = std::make_unique<OfaSdeRunner>(media);
    } else {
        XERRO("unsupported device: %s", device.c_str());
    }
    runner->Init();

    ProcessSync psync;
    psync.Sync(2, "Bg build done");

    // warm up
    for (size_t i = 0; i < warmup_cnt; ++i) runner->Execute(cmd_cnt, false);
    double thpt = 0;

    psync.Sync(4, "Bg warmup done");
    int64_t cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            runner->Execute(cmd_cnt, false);
            cnt++;
        }
    });
    psync.Sync(7, "Bg done");

    thpt = (double)cnt * 1e9 / ns;
    std::ofstream file(out, std::ios::app);
    file << thpt << std::endl;
    file.close();
    XINFO("[RESULT] Bg throughput %.2f reqs/s", thpt);

    runner->Final();
}
