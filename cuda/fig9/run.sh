#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig9/raw

mkdir -p ${RESULT_DIR}

cd ${TESTCASE_ROOT} && make clean && make

${TESTCASE_ROOT}/output/bin/levels 1 > ${RESULT_DIR}/level1.dat
${TESTCASE_ROOT}/output/bin/levels 2 > ${RESULT_DIR}/level2.dat
${TESTCASE_ROOT}/output/bin/levels 3 > ${RESULT_DIR}/level3.dat
