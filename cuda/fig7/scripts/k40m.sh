#!/bin/bash

# for k40m: 7.1 reqs/s (20% load)
fg_thpt=7.1
dev_name=k40m

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)

docker run \
    -it --rm \
    --name xsched-artifacts-cuda \
    --gpus all \
    --ipc host \
    --pid host \
    -v ${ROOT}:/xsched-artifacts \
    shenwhang/xsched-cuda:0.3 \
    /bin/bash /xsched-artifacts/cuda/fig7/scripts/run.sh ${fg_thpt} ${dev_name}
