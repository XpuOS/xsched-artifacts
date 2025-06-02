#!/bin/bash

TGS_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../ && pwd -P)

docker run --rm --name standalone \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${TGS_ROOT}:/cluster \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd /cluster/workloads/xsched_baseline; python ./shufflenet_tgs.py 1> sa.log 2> sa.err" &

trap 'docker kill standalone' SIGINT
wait
