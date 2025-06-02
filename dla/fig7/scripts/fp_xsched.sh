#!/bin/bash

fg_thpt=42.7
dev_name=dla

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/xserver HPF 50000 &
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
export XSCHED_AUTO_XQUEUE_PRIORITY=1
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimcudla.so ${TESTCASE_ROOT}/output/bin/fp_fg \
    ${ROOT}/assets/models/dla \
    ${fg_thpt} \
    ${RESULT_DIR}/fp_xsched_${dev_name}.cdf &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

sleep 2
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=2
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=1
export XSCHED_AUTO_XQUEUE_PRIORITY=0
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimcudla.so ${TESTCASE_ROOT}/output/bin/fp_bg \
    ${ROOT}/assets/models/dla &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}; kill -2 ${SERVER_PID}' SIGINT
wait ${FG_PID} ${BG_PID}
kill -2 ${SERVER_PID}
wait ${SERVER_PID}

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
