#include "core.h"

std::unique_ptr<ov::Core> g_core = nullptr;

void OvInit()
{
    g_core = std::make_unique<ov::Core>();
}

void OvDestroy()
{
    g_core = nullptr;
}
