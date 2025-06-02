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
    docker exec xsched-fig11-sci-sa pkill -9 -f 'bin/cuda_euler'
    docker exec xsched-fig11-fin-sa pkill -9 -f 'bin/cuda_bs'
    docker kill xsched-fig11-sci-sa
    docker kill xsched-fig11-fin-sa
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name xsched-fig11-sci-sa \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             1> ${RESULT_DIR_CONTAINER}/scifin_sa_ojob_gv100.json" &
sleep 120
cleanup
echo "Testcase standalone scientific computing finished, start financial computing"

docker run \
    --rm --name xsched-fig11-fin-sa \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_bs \
             1> ${RESULT_DIR_CONTAINER}/scifin_sa_pjob_gv100.json" &

sleep 120
echo "Testcase standalone scientific and financial computing finished"
cleanup
