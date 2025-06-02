#!/bin/bash

fg_thpt=7
test_time=40
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
    --latency-output ${RESULT_DIR}/fp_base_${dev_name}.cdf &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time ${test_time} &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}' SIGINT
wait ${FG_PID} ${BG_PID}
