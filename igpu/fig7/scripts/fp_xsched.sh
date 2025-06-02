#!/bin/bash

THPT=6.5

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

rm -f /dev/shm/sem.xsched_sem
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/xserver HPF 50000 &
SERVER_PID=$!
sleep 2

export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.0/lib/:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

export XSCHED_POLICY=GBL
export XSCHED_AUTO_XQUEUE=ON

export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=64
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=32
export XSCHED_AUTO_XQUEUE_PRIORITY=1
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so python3 ${TESTCASE_ROOT}/python/fp_fg.py \
    --thpt ${THPT} \
    --cdf-file ${RESULT_DIR}/fp_xsched_igpu.cdf &
FG_PID=$!

sleep 2
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_LEVELZERO_SLICE_CNT=1
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so python3 ${TESTCASE_ROOT}/python/fp_bg.py &
BG_PID=$!

trap 'kill -9 ${FG_PID} ${BG_PID}; kill -2 ${SERVER_PID}' SIGINT
wait ${FG_PID} ${BG_PID}
kill -2 ${SERVER_PID}
wait ${SERVER_PID}

rm -f /dev/shm/sem.xsched_sem
rm -rf /dev/shm/__IPC_SHM__*
