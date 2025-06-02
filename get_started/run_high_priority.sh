#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

export XSCHED_POLICY=GBL # means the client will use the global XSched scheduling server
export XSCHED_AUTO_XQUEUE=ON # means the XShim will automatically create XQueues for each task
export XSCHED_AUTO_XQUEUE_PRIORITY=1 # means the auto-created XQueue will be assigned with priority 1
export XSCHED_AUTO_XQUEUE_LEVEL=1 # means the auto-created XQueue will be assigned with level 1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16 # means the auto-created XQueue will be assigned with threshold 16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8 # means the auto-created XQueue will be assigned with command batch size 8
export LD_LIBRARY_PATH=${ROOT}/sys/xsched/output/lib:$LD_LIBRARY_PATH # use XShim to intercept the libcuda.so calls

${TESTCASE_ROOT}/app