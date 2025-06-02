#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#include "utils.h"
#include "model.h"
#include "xsched/utils.h"

#define WARMUP_CNT      300
#define TEST_CNT        1000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("usage: %s <engine dir> <out>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string engine_dir(argv[1]);
    std::string out(argv[2]);

    ProcessSync psync;
    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));
    CudlaModel model(engine_dir);
    psync.Sync(2, "Fg build done");

    // warm up
    for (size_t i = 0; i < WARMUP_CNT; ++i) model.Infer(stream);
    double thpt = 0;

    psync.Sync(4, "Fg warmup done");
    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) model.Infer(stream);
    });
    psync.Sync(5, "Fg benchmark done");

    thpt = (double)TEST_CNT * 1e9 / ns;
    std::ofstream file(out);
    file << thpt << std::endl;
    file.close();
    XINFO("[RESULT] Fg throughput %.2f reqs/s", thpt);
    psync.Sync(7, "Fg written");

    CUDART_ASSERT(cudaStreamDestroy(stream));
}
