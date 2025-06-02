#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

rm -f /dev/shm/sem.xsched_sem
mkdir -p ${RESULT_DIR}

export LD_LIBRARY_PATH=/opt/intel/oneapi/2025.0/lib/:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

python3 ${TESTCASE_ROOT}/python/up_sa.py \
    --thpt-file ${RESULT_DIR}/up_sa_igpu.thpt

rm -f /dev/shm/sem.xsched_sem
