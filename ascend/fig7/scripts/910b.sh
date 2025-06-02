#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
BUILD_PATH=${TESTCASE_ROOT}/build
OUTPUT_PATH=${TESTCASE_ROOT}/output

cd ${TESTCASE_ROOT}
make clean
make

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh
bash ${TESTCASE_ROOT}/scripts/fp_base.sh
bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh

bash ${TESTCASE_ROOT}/scripts/up_sa.sh
bash ${TESTCASE_ROOT}/scripts/up_base.sh
bash ${TESTCASE_ROOT}/scripts/up_xsched.sh
