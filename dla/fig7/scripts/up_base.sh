#!/bin/bash

dev_name=dla

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/up_fg \
    ${ROOT}/assets/models/dla \
    ${RESULT_DIR}/up_base_${dev_name}.thpt &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

sleep 2
${TESTCASE_ROOT}/output/bin/up_bg \
    ${ROOT}/assets/models/dla \
    ${RESULT_DIR}/up_base_${dev_name}.thpt &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}' SIGINT
wait ${FG_PID} ${BG_PID}

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
