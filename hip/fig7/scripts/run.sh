#!/bin/bash

sleep_time=10

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

bash ${TESTCASE_ROOT}/scripts/build.sh

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh
sleep ${sleep_time}
bash ${TESTCASE_ROOT}/scripts/fp_base.sh
sleep ${sleep_time}
bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh

sleep ${sleep_time}
bash ${TESTCASE_ROOT}/scripts/up_sa.sh
sleep ${sleep_time}
bash ${TESTCASE_ROOT}/scripts/up_base.sh
sleep ${sleep_time}
bash ${TESTCASE_ROOT}/scripts/up_xsched.sh
