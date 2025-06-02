#include <cmath>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>

#include "core.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      100
#define TEST_CNT        200

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("usage: %s <model name>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_name(argv[1]);

    ProcessSync psync;
    psync.Sync(2, "bg build waiting"); // wait until the engine is built

    OvInit();
    OvModel model(model_name, "NPU", {{"NPU_USE_NPUW", "YES"}});

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Infer();
    psync.Sync(4, "bg warmup ready");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            model.Infer();
            infer_cnt++;
        }
    });
    double bg_thpt = (double)infer_cnt * 1e9 / ns;

    OvDestroy();
}
