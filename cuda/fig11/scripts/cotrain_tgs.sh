#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw
TGS_ROOT=${ROOT}/sys/tgs

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
TGS_ROOT_CONTAINER=${ROOT_CONTAINER}/sys/tgs

mkdir -p ${RESULT_DIR}

# build tgs
docker run --rm --name tgs-fig11-build \
           --gpus "device=0" \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TGS_ROOT_CONTAINER}/hijack && ./build.sh;"

cleanup() {
    docker exec tgs-fig11-cotrain-ojob pkill -9 -f 'python ./training/shufflenet.py'
    docker exec tgs-fig11-cotrain-pjob pkill -9 -f 'python ./training/shufflenet.py'
    docker kill tgs-fig11-cotrain-ojob
    docker kill tgs-fig11-cotrain-pjob
}
trap 'cleanup; kill $$' SIGINT

# start low-priority ojob
docker run --rm --name tgs-fig11-cotrain-ojob \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro \
           -v ${TGS_ROOT}/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro \
           -v ${TGS_ROOT}/gsharing:/etc/gsharing \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TESTCASE_ROOT_CONTAINER}; python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_tgs_ojob_gv100.json" &

sleep 5

# start high-priority pjob
docker run --rm --name tgs-fig11-cotrain-pjob \
           --gpus "device=0" \
           --ipc host --network host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro \
           -v ${TGS_ROOT}/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro \
           -v ${TGS_ROOT}/gsharing:/etc/gsharing \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TESTCASE_ROOT_CONTAINER}; python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_tgs_pjob_gv100.json" &

sleep 300
echo "Testcase TGS co-training finished"
cleanup
