#include <fstream>
#include <iostream>

#include "core.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      100
#define TEST_CNT        500

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("usage: %s <model name> <out>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_name(argv[1]);
    std::string out(argv[2]);

    ProcessSync psync;
    psync.Sync(2, "Bg build start");

    OvInit();
    OvModel model(model_name, "NPU", {{"NPU_USE_NPUW", "YES"}});

    // warm up
    for (size_t i = 0; i < WARMUP_CNT; ++i) model.Infer();
    double thpt = 0;

    psync.Sync(4, "Bg warmup done");
    int64_t cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            model.Infer();
            cnt++;
        }
    });
    psync.Sync(7, "Bg done");

    thpt = (double)cnt * 1e9 / ns;
    std::ofstream file(out, std::ios::app);
    file << thpt << std::endl;
    file.close();
    XINFO("[RESULT] Bg throughput %.2f reqs/s", thpt);

    OvDestroy();
}
