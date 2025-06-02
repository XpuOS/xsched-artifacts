#!/bin/bash

fg_thpt=100
test_time=10
test_cnt=200
dev_name=mi50

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw
mkdir -p ${RESULT_DIR}

python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq ${fg_thpt} \
    --time ${test_time} \
    --latency-cnt ${test_cnt} \
    --latency-output ${RESULT_DIR}/fp_sa_${dev_name}.cdf

