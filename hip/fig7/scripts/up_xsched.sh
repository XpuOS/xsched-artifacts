#!/bin/bash

dev_name=mi50

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/xserver PUP 50000 &
SERVER_PID=$!
echo "SERVER_PID: ${SERVER_PID}"
sleep 2

export LD_LIBRARY_PATH=${TESTCASE_ROOT}/output/lib:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

export XSCHED_POLICY=GBL
export XSCHED_AUTO_XQUEUE=ON

export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
export XSCHED_AUTO_XQUEUE_UTILIZATION=75
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimhip.so python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time 30 \
    --tpt-output ${RESULT_DIR}/up_xsched_fg_${dev_name}.thpt &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

export XSCHED_AUTO_XQUEUE_UTILIZATION=25
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimhip.so python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time 30 \
    --tpt-output ${RESULT_DIR}/up_xsched_bg_${dev_name}.thpt &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}; kill -2 ${SERVER_PID}' SIGINT
wait ${FG_PID} ${BG_PID}
kill -2 ${SERVER_PID}
wait ${SERVER_PID}

rm -rf /dev/shm/__IPC_SHM__*
cat ${RESULT_DIR}/up_xsched_fg_${dev_name}.thpt ${RESULT_DIR}/up_xsched_bg_${dev_name}.thpt > ${RESULT_DIR}/up_xsched_${dev_name}.thpt
