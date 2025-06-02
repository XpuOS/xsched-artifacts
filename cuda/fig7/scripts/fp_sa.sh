#!/bin/bash

fg_thpt=$1
dev_name=$2
batch_size=1

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/fp_sa \
    ${ROOT}/assets/models/resnet152 \
    ${batch_size} \
    ${fg_thpt} \
    ${RESULT_DIR}/fp_sa_${dev_name}.cdf

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
