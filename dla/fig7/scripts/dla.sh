#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

cd ${TESTCASE_ROOT}
make clean
make

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh
echo "fp_sa done"
sleep 10

bash ${TESTCASE_ROOT}/scripts/fp_base.sh
echo "fp_base done"
sleep 10

bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh
echo "fp_xsched done"
sleep 10

bash ${TESTCASE_ROOT}/scripts/up_sa.sh
echo "up_sa done"
sleep 10

bash ${TESTCASE_ROOT}/scripts/up_base.sh
echo "up_base done"
sleep 10

bash ${TESTCASE_ROOT}/scripts/up_xsched.sh
echo "up_xsched done"
sleep 10
