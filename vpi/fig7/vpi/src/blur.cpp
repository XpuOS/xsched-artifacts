#include <opencv2/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/GaussianFilter.h>
#include <vpi/algo/ConvertImageFormat.h>

#include "blur.h"

PvaBlurRunner::PvaBlurRunner(const std::string &video_path_in): video_path_in_(video_path_in)
{
    XASSERT(video_cv_in_.open(video_path_in_),
            "Cannot open input video '%s'", video_path_in_.c_str());
    
    w_ = 3264; // max supported width for PVA
    h_ = 2448; // max supported height for PVA
    fps_ = video_cv_in_.get(cv::CAP_PROP_FPS);
}

void PvaBlurRunner::Init()
{
    VPI_ASSERT(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_PVA, &stream_));

    cv::Mat frame_cv_in;
    VPIImage frame_vpi_in = nullptr;

    while (video_cv_in_.read(frame_cv_in)) {
        if (frames_vpi_blurred_.size() >= 32) break;
        cv::resize(frame_cv_in, frame_cv_in, cv::Size(w_, h_));
        if (frame_vpi_in == nullptr) {
            // Create a VPIImage that wraps the frame
            VPI_ASSERT(vpiImageCreateWrapperOpenCVMat(frame_cv_in, 0,
                                                      &frame_vpi_in));
        } else {
            // reuse existing VPIImage wrapper to wrap the new frame.
            VPI_ASSERT(vpiImageSetWrappedOpenCVMat(frame_vpi_in,
                                                   frame_cv_in));
        }

        VPIImage frame_vpi_converted;
        VPI_ASSERT(vpiImageCreate(w_, h_,
                                  VPI_IMAGE_FORMAT_U8,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_converted));
        VPI_ASSERT(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_CUDA,
                                               frame_vpi_in,
                                               frame_vpi_converted,
                                               nullptr));
        VPI_ASSERT(vpiStreamSync(stream_));
        frames_vpi_converted_.emplace_back(frame_vpi_converted);
        
        VPIImage frame_vpi_blurred;
        VPI_ASSERT(vpiImageCreate(w_, h_,
                                  VPI_IMAGE_FORMAT_U8,
                                  VPI_EXCLUSIVE_STREAM_ACCESS,
                                  &frame_vpi_blurred));
        frames_vpi_blurred_.emplace_back(frame_vpi_blurred);
    }

    XASSERT(frames_vpi_converted_.size() == frames_vpi_blurred_.size(),
            "frame cnt of converted (%ld) and blurred (%ld) mismatch",
            frames_vpi_converted_.size(), frames_vpi_blurred_.size());
    frames_cnt_ = frames_vpi_converted_.size();
}

void PvaBlurRunner::Final()
{
    // for (size_t i = 0; i < frames_cnt_; ++i) {
    //     VPIImageData data_vpi_out;
    //     VPI_ASSERT(vpiImageLockData(frames_vpi_blurred_[i], VPI_LOCK_READ,
    //                                 VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
    //                                 &data_vpi_out));

    //     // Returned data consists of host-accessible memory buffers
    //     // in pitch-linear layout.
    //     XASSERT(data_vpi_out.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
    //             "Unexpected buffer type: %d", data_vpi_out.bufferType);

    //     VPIImageBufferPitchLinear &pitch_vpi_out = data_vpi_out.buffer.pitch;

    //     cv::Mat frame_cv_out(pitch_vpi_out.planes[0].height,
    //                          pitch_vpi_out.planes[0].width,
    //                          CV_8UC1,
    //                          pitch_vpi_out.planes[0].data,
    //                          pitch_vpi_out.planes[0].pitchBytes);
    //     video_cv_out_ << frame_cv_out;

    //     // Done handling output image, don't forget to unlock it.
    //     VPI_ASSERT(vpiImageUnlock(frames_vpi_blurred_[i]));
    // }

    this->Sync();

    for (auto frame : frames_vpi_converted_) {
        vpiImageDestroy(frame);
    }
    for (auto frame : frames_vpi_blurred_) {
        vpiImageDestroy(frame);
    }

    vpiStreamDestroy(stream_);
}

void PvaBlurRunner::Execute(size_t cmd_cnt, bool sync)
{
    cmd_cnt = std::min(cmd_cnt, frames_cnt_);
    for (size_t i = 0; i < cmd_cnt; ++i) {
        VPI_ASSERT(vpiSubmitGaussianFilter(stream_, VPI_BACKEND_PVA,
                                           frames_vpi_converted_[i],
                                           frames_vpi_blurred_[i],
                                           3, 3, 1, 1,
                                           VPI_BORDER_ZERO));

        if (sync) VPI_ASSERT(vpiStreamSync(stream_));
    }

    if (!sync) VPI_ASSERT(vpiStreamSync(stream_));
}

void PvaBlurRunner::Enqueue(size_t cmd_cnt, bool sync)
{
    cmd_cnt = std::min(cmd_cnt, frames_cnt_);
    for (size_t i = 0; i < cmd_cnt; ++i) {
        VPI_ASSERT(vpiSubmitGaussianFilter(stream_, VPI_BACKEND_PVA,
                                           frames_vpi_converted_[i],
                                           frames_vpi_blurred_[i],
                                           3, 3, 1, 1,
                                           VPI_BORDER_ZERO));
        if (sync) VPI_ASSERT(vpiStreamSync(stream_));
    }
}

void PvaBlurRunner::Sync()
{
    VPI_ASSERT(vpiStreamSync(stream_));
}
