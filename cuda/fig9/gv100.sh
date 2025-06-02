#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../ && pwd -P)

docker run \
    -it --rm \
    --name xsched-artifacts-cuda \
    --gpus all \
    --ipc host \
    --pid host \
    -v ${ROOT}:/xsched-artifacts \
    shenwhang/xsched-cuda:0.3 \
    /bin/bash /xsched-artifacts/cuda/fig9/run.sh
