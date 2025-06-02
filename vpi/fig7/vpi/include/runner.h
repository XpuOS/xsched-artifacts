#pragma once

#include <cstddef>
#include <vpi/Stream.h>

class VpiRunner
{
public:
    VpiRunner() = default;
    virtual ~VpiRunner() = default;

    virtual void Init() = 0;
    virtual void Final() = 0;
    virtual void Execute(size_t cmd_cnt, bool sync) = 0;
};
