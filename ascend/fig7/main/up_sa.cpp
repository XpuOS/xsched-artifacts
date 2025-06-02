#include <fstream>
#include <iostream>

#include "utils.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      1000
#define TEST_CNT        2000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("usage: %s <model dir> <out>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_dir(argv[1]);
    std::string out(argv[2]);

    aclrtStream stream;
    AclModel model(model_dir);
    ACL_ASSERT(aclrtCreateStream(&stream));

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Execute(stream);
    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) { model.Execute(stream); }
    });

    double thpt = (double)TEST_CNT * 1e9 / ns;
    XINFO("[RESULT] sa throughput %.2f reqs/s", thpt);

    std::ofstream file(out);
    file << thpt << std::endl;
    return 0;
}
