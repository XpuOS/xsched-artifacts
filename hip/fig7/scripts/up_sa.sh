#!/bin/bash

dev_name=mi50

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw
mkdir -p ${RESULT_DIR}

python ${TESTCASE_ROOT}/resnet_infer.py \
    --freq 10000 \
    --time 30 \
    --tpt-output ${RESULT_DIR}/up_sa_${dev_name}.thpt
