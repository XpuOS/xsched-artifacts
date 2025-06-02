#!/bin/bash

TGS_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../ && pwd -P)

docker run --rm --name tgs_lp \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${TGS_ROOT}:/cluster \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro \
           -v ${TGS_ROOT}/gsharing:/etc/gsharing \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd /cluster/workloads/xsched_baseline; python ./shufflenet_tgs.py 1> lp.log 2> lp.err" &

sleep 5

docker run --rm --name tgs_hp \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${TGS_ROOT}:/cluster \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro \
           -v ${TGS_ROOT}/gsharing:/etc/gsharing \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd /cluster/workloads/xsched_baseline; python ./shufflenet_tgs.py 1> hp.log 2> hp.err" &

trap 'docker kill tgs_lp tgs_hp' SIGINT
wait
