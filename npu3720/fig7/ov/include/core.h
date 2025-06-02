#pragma once

#include <memory>
#include <openvino/openvino.hpp>

extern std::unique_ptr<ov::Core> g_core;

void OvInit();
void OvDestroy();
