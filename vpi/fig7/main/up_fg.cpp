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

    ProcessSync psync;
    psync.Sync(2, "Fg build waiting");

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

    std::this_thread::sleep_for(std::chrono::seconds(20));

    // warm up
    for (size_t i = 0; i < warmup_cnt; ++i) runner->Execute(cmd_cnt, false);
    double thpt = 0;

    psync.Sync(4, "Fg warmup done");
    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < test_cnt; ++i) runner->Execute(cmd_cnt, false);
    });
    psync.Sync(5, "Fg benchmark done");

    thpt = (double)test_cnt * 1e9 / ns;
    std::ofstream file(out);
    file << thpt << std::endl;
    file.close();
    XINFO("[RESULT] Fg throughput %.2f reqs/s", thpt);
    psync.Sync(7, "Fg written");

    runner->Final();
}
