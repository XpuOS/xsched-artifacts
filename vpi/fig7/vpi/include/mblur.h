#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "blur.h"

// because the pva has multiple cores, one PvaBlurRunner cannot saturate the cores
// so we need to run multiple PvaBlurRunners at the same time to increase the load.
class MultiPvaBlurRunner : public VpiRunner
{
public:
    MultiPvaBlurRunner(const std::string &video_path_in, size_t num);
    virtual ~MultiPvaBlurRunner() = default;

    virtual void Init() override;
    virtual void Final() override;
    virtual void Execute(size_t cmd_cnt, bool sync) override;

private:
    std::vector<PvaBlurRunner> runners_;
    std::vector<std::thread> threads_;

    std::mutex start_mtx_;
    std::condition_variable start_cv_;
    std::vector<bool> start_;

    std::mutex done_mtx_;
    std::condition_variable done_cv_;
    std::vector<bool> done_;
    
    std::vector<bool> exit_;

    size_t cmd_cnt_;
    bool sync_;
};
