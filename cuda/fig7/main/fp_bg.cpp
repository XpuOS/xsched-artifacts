#include <cmath>
#include <thread>
#include <atomic>
#include <iostream>
#include <functional>
#include <cuda_runtime.h>

#include "model.h"
#include "cuda_assert.h"
#include "xsched/utils.h"

#define WARMUP_CNT      300
#define TEST_CNT        1000

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("usage: %s <model name> <batch size>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_name(argv[1]);
    int batch_size = atoi(argv[2]);

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));

    ProcessSync psync;
    psync.Sync(2, "bg build waiting"); // wait until the engine is built
    TRTModel model(model_name + ".onnx", model_name + ".engine", batch_size);

    // warm up
    for (int i = 0; i < WARMUP_CNT; ++i) model.Infer(stream);
    psync.Sync(4, "bg warmup ready");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            model.Infer(stream);
            infer_cnt++;
        }
    });
    double bg_thpt = (double)infer_cnt * 1e9 / ns;

    CUDART_ASSERT(cudaStreamDestroy(stream));
}
