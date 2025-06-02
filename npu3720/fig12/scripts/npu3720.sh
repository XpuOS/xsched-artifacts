#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

docker run \
    -it --rm \
    --name xsched-artifacts-npu3720-fig12 \
    --ipc host \
    --pid host \
    --device /dev/accel:/dev/accel \
    --device /dev/dri:/dev/dri \
    --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
    --group-add=$(stat -c "%g" /dev/dri/card* | head -n 1) \
    -v ${ROOT}:/xsched-artifacts \
    shenwhang/xsched-ze:0.5 \
    /bin/bash /xsched-artifacts/npu3720/fig12/scripts/run.sh
