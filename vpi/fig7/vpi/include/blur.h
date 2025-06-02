#pragma once

#include <vector>
#include <string>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <opencv2/videoio.hpp>

#include "utils.h"
#include "runner.h"

class PvaBlurRunner : public VpiRunner
{
public:
    PvaBlurRunner(const std::string &video_path_in);
    virtual ~PvaBlurRunner() = default;

    virtual void Init() override;
    virtual void Final() override;
    virtual void Execute(size_t cmd_cnt, bool sync) override;

    void Enqueue(size_t cmd_cnt, bool sync);
    void Sync();

private:
    VPIStream stream_;
    const std::string video_path_in_;

    int w_;
    int h_;
    double fps_;
    size_t frames_cnt_;
    cv::VideoCapture video_cv_in_;
    // cv::VideoWriter video_cv_out_;
    
    std::vector<VPIImage> frames_vpi_converted_;
    std::vector<VPIImage> frames_vpi_blurred_;
};
