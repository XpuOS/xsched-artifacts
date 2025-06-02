#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

# co training workloads
${TESTCASE_ROOT}/scripts/cotrain_sa.sh
echo "cotrain_sa done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_base.sh
echo "cotrain_base done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_vcuda.sh
echo "cotrain_vcuda done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_tgs.sh
echo "cotrain_tgs done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_xsched.sh
echo "cotrain_xsched done"

# scientific & financial workloads
sleep 5
${TESTCASE_ROOT}/scripts/scifin_sa.sh
echo "scifin_sa done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_base.sh
echo "scifin_base done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_vcuda.sh
echo "scifin_vcuda done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_tgs.sh
echo "scifin_tgs done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_xsched.sh
echo "scifin_xsched done"
