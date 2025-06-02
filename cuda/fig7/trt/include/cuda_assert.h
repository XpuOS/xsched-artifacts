#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "xsched/utils.h"

#define CUDART_ASSERT(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (UNLIKELY(result != cudaSuccess)) { \
            XERRO("cuda runtime error %d", result); \
        } \
    } while (0);
