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

    ProcessSync psync;
    psync.Sync(2, "Bg build start");

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));
    TRTModel model(model_name + ".onnx", model_name + ".engine", batch_size);

    // warm up
    for (size_t i = 0; i < WARMUP_CNT; ++i) model.Infer(stream);
    double thpt = 0;

    psync.Sync(4, "Bg warmup done");
    int64_t cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 5) {
            model.Infer(stream);
            cnt++;
        }
    });
    psync.Sync(7, "Bg done");

    thpt = (double)cnt * 1e9 / ns;
    std::ofstream file(out, std::ios::app);
    file << thpt << std::endl;
    file.close();
    XINFO("[RESULT] Bg throughput %.2f reqs/s", thpt);

    CUDART_ASSERT(cudaStreamDestroy(stream));
}
