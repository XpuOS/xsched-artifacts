#pragma once

#include <acl/acl.h>

#include "xsched/utils.h"

#define ACL_ASSERT(cmd) \
    do { \
        aclError result = cmd; \
        if (UNLIKELY(result != ACL_SUCCESS)) { \
            XERRO("acl error %d", result); \
        } \
    } while (0);
