#!/bin/bash

TGS_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../ && pwd -P)

docker run --rm --name nosched_lp \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${TGS_ROOT}:/cluster \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd /cluster/workloads/xsched_baseline; python ./shufflenet_tgs.py 1> lp.log 2> lp.err" &

sleep 5

docker run --rm --name nosched_lp \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${TGS_ROOT}:/cluster \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd /cluster/workloads/xsched_baseline; python ./shufflenet_tgs.py 1> hp.log 2> hp.err" &

trap 'docker kill nosched_lp nosched_lp' SIGINT
wait
