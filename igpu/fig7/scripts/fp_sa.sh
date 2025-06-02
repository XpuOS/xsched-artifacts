#!/bin/bash

THPT=6.5

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

rm -f /dev/shm/sem.xsched_sem
mkdir -p ${RESULT_DIR}

export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.0/lib/:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

python3 ${TESTCASE_ROOT}/python/fp_sa.py \
    --thpt ${THPT} \
    --cdf-file ${RESULT_DIR}/fp_sa_igpu.cdf

rm -f /dev/shm/sem.xsched_sem
