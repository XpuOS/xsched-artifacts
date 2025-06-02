#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

# co training workloads
${TESTCASE_ROOT}/scripts/cotrain_sa.sh
echo "cotrain_sa done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_base.sh
echo "cotrain_base done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_xsched.sh
echo "cotrain_xsched done"

sleep 10
${TESTCASE_ROOT}/scripts/cotrain_xsched_wo_prog.sh
echo "cotrain_xsched_wo_prog done"

# scientific & financial workloads
sleep 5
${TESTCASE_ROOT}/scripts/scifin_sa.sh
echo "scifin_sa done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_base.sh
echo "scifin_base done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_xsched.sh
echo "scifin_xsched done"

sleep 5
${TESTCASE_ROOT}/scripts/scifin_xsched_wo_prog.sh
echo "scifin_xsched_wo_prog done"
