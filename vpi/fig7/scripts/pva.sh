#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig7/raw

device=pva
media=${ROOT}/assets/media/pedestrians.avi
cmd_cnt=8
fg_thpt=2.0

cd ${TESTCASE_ROOT}
make clean
make

bash ${TESTCASE_ROOT}/scripts/fp_sa.sh ${device} ${media} ${cmd_cnt} ${fg_thpt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/fp_base.sh ${device} ${media} ${cmd_cnt} ${fg_thpt}
sleep 10

rm ${RESULT_DIR}/fp_xsched_${device}.cdf
while [ ! -f ${RESULT_DIR}/fp_xsched_${device}.cdf ]; do
    bash ${TESTCASE_ROOT}/scripts/fp_xsched.sh ${device} ${media} ${cmd_cnt} ${fg_thpt} 2 1
    sleep 5
done

bash ${TESTCASE_ROOT}/scripts/up_sa.sh ${device} ${media} ${cmd_cnt}
sleep 10
bash ${TESTCASE_ROOT}/scripts/up_base.sh ${device} ${media} ${cmd_cnt}

out_file=${RESULT_DIR}/up_xsched_${device}.thpt
rm $out_file
while [ ! -f "$out_file" ] || [ "$(wc -l < "${out_file}")" -lt 2 ]; do
    sleep 5
    bash ${TESTCASE_ROOT}/scripts/up_xsched.sh ${device} ${media} ${cmd_cnt} 50000 1 1
done
