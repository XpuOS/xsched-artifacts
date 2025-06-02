#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw
TGS_ROOT=${ROOT}/sys/tgs

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
BUILD_CONTAINER=${TESTCASE_ROOT_CONTAINER}/build/scifin
OUTPUT_CONTAINER=${TESTCASE_ROOT_CONTAINER}/output/scifin
TGS_ROOT_CONTAINER=${ROOT_CONTAINER}/sys/tgs

mkdir -p ${RESULT_DIR}

# build scifin
docker run \
    -it --rm --name xsched-fig11-scifin-build \
    --gpus "device=0" \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER} && \
             make BUILD_PATH=${BUILD_CONTAINER} OUTPUT_PATH=${OUTPUT_CONTAINER}"

# build tgs
docker run --rm --name tgs-fig11-build \
           --gpus "device=0" \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TGS_ROOT_CONTAINER}/hijack && ./build.sh;"

cleanup() {
    docker exec tgs-fig11-scifin-ojob pkill -9 -f 'bin/cuda_euler'
    docker exec tgs-fig11-scifin-pjob pkill -9 -f 'bin/cuda_bs'
    docker kill tgs-fig11-scifin-ojob
    docker kill tgs-fig11-scifin-pjob
}
trap 'cleanup; kill $$' SIGINT

# start low-priority ojob
docker run \
    --rm --name tgs-fig11-scifin-ojob \
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
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             1> ${RESULT_DIR_CONTAINER}/scifin_tgs_ojob_gv100.json" &

sleep 5

# start high-priority pjob
docker run \
    --rm --name tgs-fig11-scifin-pjob \
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
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_bs \
             1> ${RESULT_DIR_CONTAINER}/scifin_tgs_pjob_gv100.json" &

sleep 120
echo "Testcase TGS scientific and financial computing finished"
cleanup
