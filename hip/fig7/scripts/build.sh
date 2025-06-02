#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
BUILD_PATH=${TESTCASE_ROOT}/build
OUTPUT_PATH=${TESTCASE_ROOT}/output

make -C ${ROOT}/sys/xsched hip BUILD_PATH=${BUILD_PATH} OUTPUT_PATH=${OUTPUT_PATH}
