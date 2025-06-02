#!/bin/bash

dev_name=mi50

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw
mkdir -p ${RESULT_DIR}

python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time 30 \
    --tpt-output ${RESULT_DIR}/up_base_fg_${dev_name}.thpt &
FG_PID=$!
echo "FG_PID: ${FG_PID}"

python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time 30 \
    --tpt-output ${RESULT_DIR}/up_base_bg_${dev_name}.thpt &
BG_PID=$!
echo "BG_PID: ${BG_PID}"

trap 'kill -9 ${FG_PID} ${BG_PID}' SIGINT
wait ${FG_PID} ${BG_PID}

cat ${RESULT_DIR}/up_base_fg_${dev_name}.thpt ${RESULT_DIR}/up_base_bg_${dev_name}.thpt > ${RESULT_DIR}/up_base_${dev_name}.thpt
