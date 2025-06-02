#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

device=ofa
media=${ROOT}/assets/media/chair_stereo
cmd_cnt=8
fg_thpt=9.0

cd ${TESTCASE_ROOT}
make clean
make

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh ${device} ${media} ${cmd_cnt} ${fg_thpt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_base.sh ${device} ${media} ${cmd_cnt} ${fg_thpt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh ${device} ${media} ${cmd_cnt} ${fg_thpt} 4 2

sleep 10
bash ${TESTCASE_ROOT}/scripts/up_sa.sh ${device} ${media} ${cmd_cnt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/up_base.sh ${device} ${media} ${cmd_cnt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/up_xsched.sh ${device} ${media} ${cmd_cnt} 30000 4 2
