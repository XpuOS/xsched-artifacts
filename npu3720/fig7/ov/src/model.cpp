#include "core.h"
#include "model.h"
#include "xsched/utils.h"

OvModel::OvModel(const std::string &model_path,
                 const std::string &device,
                 const ov::AnyMap &properties)
{
    auto model = g_core->read_model(model_path);
    model_ = g_core->compile_model(model, device, properties);
    request_ = model_.create_infer_request();
    input_ = request_.get_input_tensor();
    output_ = request_.get_output_tensor();

    XINFO("ov model compiled, input: %s%s, output: %s%s",
          input_.get_element_type().c_type_string().c_str(),
          input_.get_shape().to_string().c_str(),
          output_.get_element_type().c_type_string().c_str(),
          output_.get_shape().to_string().c_str());

    char *input_data = (char *)input_.data();
    std::fill(input_data, input_data + input_.get_byte_size(), 0);
}

void OvModel::Infer()
{
    request_.start_async();
    request_.wait();
}

void OvModel::InferAsync()
{
    request_.start_async();
}

void OvModel::Sync()
{
    request_.wait();
}

void OvModel::SetInput(void *data, size_t size)
{
    XASSERT(size < input_.get_byte_size(),
            "copy input overflow, copy size (%zuB) > tensor size (%zuB)",
            size, input_.get_byte_size());

    char *input_data = (char *)input_.data();
    std::copy((char *)data, (char *)data + size, input_data);
}

void OvModel::GetOutput(void *data, size_t size)
{
    XASSERT(size < output_.get_byte_size(),
            "copy output overflow, copy size (%zuB) > tensor size (%zuB)",
            size, output_.get_byte_size());

    char *output_data = (char *)output_.data();
    std::copy(output_data, output_data + size, (char *)data);
}
