#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#include "model.h"
#include "cuda_assert.h"
#include "xsched/utils.h"

#define WARMUP_CNT      300
#define TEST_CNT        1000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 4) {
        XINFO("usage: %s <model name> <batch size> <out>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_name(argv[1]);
    int batch_size = atoi(argv[2]);
    std::string out(argv[3]);

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));
    TRTModel model(model_name + ".onnx", model_name + ".engine", batch_size);

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Infer(stream);
    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < TEST_CNT; ++i) { model.Infer(stream); }
    });

    double thpt = (double)TEST_CNT * 1e9 / ns;
    XINFO("[RESULT] sa throughput %.2f reqs/s", thpt);

    std::ofstream file(out);
    file << thpt << std::endl;
    return 0;
}
