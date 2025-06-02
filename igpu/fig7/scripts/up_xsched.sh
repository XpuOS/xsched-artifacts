#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

rm -f /dev/shm/sem.xsched_sem
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/xserver PUP 50000 &
SERVER_PID=$!
sleep 2

export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.0/lib/:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

export XSCHED_POLICY=GBL
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_TIMESLICE=20000
export XSCHED_AUTO_XQUEUE_LEVEL=1

# in levelzero, we treat one command list as one XCommand
# however, one command list may contain 50+ kernels, so the actual threshold is 50+
export XSCHED_AUTO_XQUEUE_THRESHOLD=1
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=1

export XSCHED_AUTO_XQUEUE_UTILIZATION=75
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so python3 ${TESTCASE_ROOT}/python/up_fg.py \
    --thpt-file ${RESULT_DIR}/up_xsched_igpu.thpt &
FG_PID=$!

sleep 2
export XSCHED_AUTO_XQUEUE_UTILIZATION=25
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so python3 ${TESTCASE_ROOT}/python/up_bg.py \
    --thpt-file ${RESULT_DIR}/up_xsched_igpu.thpt &
BG_PID=$!

trap 'kill -9 ${FG_PID} ${BG_PID}; kill -2 ${SERVER_PID}' SIGINT
wait ${FG_PID} ${BG_PID}
kill -2 ${SERVER_PID}
wait ${SERVER_PID}

rm -f /dev/shm/sem.xsched_sem
rm -rf /dev/shm/__IPC_SHM__*
