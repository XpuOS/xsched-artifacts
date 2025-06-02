#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
BUILD_PATH=${TESTCASE_ROOT}/build
OUTPUT_PATH=${TESTCASE_ROOT}/output

bash ${TESTCASE_ROOT}/scripts/build.sh
echo "Build done"

bash ${TESTCASE_ROOT}/scripts/base.sh
echo "Base (Native) test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/xsched.sh
echo "XSched test done"

echo "Plotting..."
python3 ${ROOT}/results/fig12/scripts/plot.py

echo "NPU3720 test done, results: results/fig12/plot/fig12.pdf"
