#pragma once

#include <string>

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <opencv2/core/mat.hpp>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/ConvertImageFormat.h>

#include "utils.h"
#include "runner.h"

/// @brief Stereo Disparity Estimation on Optical Flow Accelerator (OFA)
class OfaSdeRunner : public VpiRunner
{
public:
    OfaSdeRunner(const std::string &image_prefix);
    virtual ~OfaSdeRunner() = default;

    virtual void Init() override;
    virtual void Final() override;
    virtual void Execute(size_t cmd_cnt, bool sync) override;

private:
    VPIStream stream_;
    const std::string left_path_in_;
    const std::string right_path_in_;

    int w_;
    int h_;
    size_t frames_cnt_;
    cv::Mat left_cv_in_;
    cv::Mat right_cv_in_;

    VPIImage left_vpi_in_;
    VPIImage right_vpi_in_;

    VPIImage disparity_;
    VPIImage stereo_left_;
    VPIImage stereo_right_;
    
    VPIPayload payload_;
    VPIImage confidence_map_ = nullptr;
    VPIStereoDisparityEstimatorCreationParams stereo_params_;
};
