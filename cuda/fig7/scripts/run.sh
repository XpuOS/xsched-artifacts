#!/bin/bash

fg_thpt=$1
dev_name=$2

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

cd ${TESTCASE_ROOT}
make clean

if [ ${dev_name} == "k40m" ]; then
    make CUDA_GEN_CODE=35
else
    make
fi

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh ${fg_thpt} ${dev_name}
sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_base.sh ${fg_thpt} ${dev_name}
sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh ${fg_thpt} ${dev_name}

sleep 10
bash ${TESTCASE_ROOT}/scripts/up_sa.sh ${dev_name}
sleep 10
bash ${TESTCASE_ROOT}/scripts/up_base.sh ${dev_name}
sleep 10
bash ${TESTCASE_ROOT}/scripts/up_xsched.sh ${dev_name}
