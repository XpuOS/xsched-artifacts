#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

bash ${TESTCASE_ROOT}/scripts/build.sh
echo "Build done"

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh
echo "Fixed Priority Standalone test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_base.sh
echo "Fixed Priority Base (Native) test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh
echo "Fixed Priority XSched test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/up_sa.sh
echo "Utilization (Bandwidth) Standalone test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/up_base.sh
echo "Utilization (Bandwidth) Base (Native) test done"

sleep 10
bash ${TESTCASE_ROOT}/scripts/up_xsched.sh
echo "Utilization (Bandwidth) XSched test done"

echo "Plotting..."
python3 ${ROOT}/results/fig7/scripts/plot.py

echo "iGPU test done, results: results/fig7/plot/fig7_top_igpu.pdf & results/fig7/plot/fig7_bottom_igpu.pdf"
