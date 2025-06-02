#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
BUILD_CONTAINER=${TESTCASE_ROOT_CONTAINER}/build/scifin
OUTPUT_CONTAINER=${TESTCASE_ROOT_CONTAINER}/output/scifin

mkdir -p ${RESULT_DIR}

# build scifin
docker run \
    -it --rm --name xsched-fig11-scifin-build \
    --gpus "device=0" \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER} && \
             make BUILD_PATH=${BUILD_CONTAINER} OUTPUT_PATH=${OUTPUT_CONTAINER}"

cleanup() {
    docker exec base-fig11-sci-ojob pkill -9 -f 'bin/cuda_euler'
    docker exec base-fig11-fin-pjob pkill -9 -f 'bin/cuda_bs'
    docker kill base-fig11-sci-ojob
    docker kill base-fig11-fin-pjob
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name base-fig11-sci-ojob \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             1> ${RESULT_DIR_CONTAINER}/scifin_base_ojob_gv100.json" &
sleep 5

docker run \
    --rm --name base-fig11-fin-pjob \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_bs \
             1> ${RESULT_DIR_CONTAINER}/scifin_base_pjob_gv100.json" &

sleep 120
echo "Testcase base (Native) scientific and financial computing finished"
cleanup
