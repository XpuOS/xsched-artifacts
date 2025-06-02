#!/bin/bash

device=$1
media=$2
cmd_cnt=$3
fg_thpt=$4

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/fp_sa \
    ${device} \
    ${media} \
    ${cmd_cnt} \
    ${fg_thpt} \
    ${RESULT_DIR}/fp_sa_${device}.cdf

ipcrm --shmem-key 0xbeef
rm -rf /dev/shm/__IPC_SHM__*
