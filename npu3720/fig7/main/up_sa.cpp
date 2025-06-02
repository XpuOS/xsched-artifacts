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

    OvInit();
    OvModel model(model_name, "NPU", {{"NPU_USE_NPUW", "YES"}});

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Infer();
    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) { model.Infer(); }
    });

    double thpt = (double)TEST_CNT * 1e9 / ns;
    XINFO("[RESULT] sa throughput %.2f reqs/s", thpt);

    std::ofstream file(out);
    file << thpt << std::endl;

    OvDestroy();
    return 0;
}
