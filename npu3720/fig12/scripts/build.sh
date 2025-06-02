#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
BUILD_PATH=${TESTCASE_ROOT}/build
OUTPUT_PATH=${TESTCASE_ROOT}/output

rm -rf ${BUILD_PATH} ${OUTPUT_PATH}

make -C ${ROOT}/sys/xsched levelzero BUILD_PATH=${BUILD_PATH}/xsched OUTPUT_PATH=${OUTPUT_PATH}

cmake -B ${BUILD_PATH}/whisper \
      -DWHISPER_OPENVINO=1 \
      -DWHISPER_SDL2=1 \
      -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
      ${TESTCASE_ROOT}/whisper

cmake --build ${BUILD_PATH}/whisper --target install -- -j$(shell nproc)
