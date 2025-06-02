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
    if (argc < 2) {
        XINFO("usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_dir(argv[1]);

    ProcessSync psync;
    psync.Sync(2, "bg build waiting"); // wait until the engine is built
    aclrtStream stream;
    AclModel model(model_dir);
    ACL_ASSERT(aclrtCreateStream(&stream));

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Execute(stream);
    psync.Sync(4, "bg warmup ready");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            model.Execute(stream);
            infer_cnt++;
        }
    });
    double bg_thpt = (double)infer_cnt * 1e9 / ns;

    ACL_ASSERT(aclrtDestroyStream(stream));
}
