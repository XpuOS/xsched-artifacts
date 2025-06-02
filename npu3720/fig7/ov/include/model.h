#pragma once

#include <string>
#include <openvino/openvino.hpp>

class OvModel
{
public:
    OvModel(const std::string &model_path,
            const std::string &device,
            const ov::AnyMap &properties = {});
    ~OvModel() = default;

    void Infer();
    void InferAsync();
    void Sync();

    void SetInput(void *data, size_t size);
    void GetOutput(void *data, size_t size);

private:
    ov::CompiledModel model_;
    ov::InferRequest request_;
    ov::Tensor input_;
    ov::Tensor output_;
};
