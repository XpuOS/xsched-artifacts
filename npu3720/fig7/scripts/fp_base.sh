#!/bin/bash

fg_thpt=8.8
dev_name=npu3720

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

export LD_LIBRARY_PATH=/opt/intel/openvino_2024.4.0/runtime/lib/intel64:/usr/local/lib:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

${TESTCASE_ROOT}/output/bin/fp_fg \
    ${ROOT}/assets/models/resnet152.xml \
    ${fg_thpt} \
    ${RESULT_DIR}/fp_base_${dev_name}.cdf &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

sleep 2
${TESTCASE_ROOT}/output/bin/fp_bg \
    ${ROOT}/assets/models/resnet152.xml &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}' SIGINT
wait ${FG_PID} ${BG_PID}

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
