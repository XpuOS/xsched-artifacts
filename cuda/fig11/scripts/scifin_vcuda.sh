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
    docker exec vcuda-fig11-scifin-ojob pkill -9 -f 'bin/cuda_euler'
    docker exec vcuda-fig11-scifin-pjob pkill -9 -f 'bin/cuda_bs'
    docker kill vcuda-fig11-scifin-ojob
    docker kill vcuda-fig11-scifin-pjob
}
trap 'cleanup; kill $$' SIGINT

# allocate 40 % to ojob
docker run \
    --rm --name vcuda-fig11-scifin-ojob \
    --gpus "device=0" \
    --ipc host --pid host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/vcuda:0.2 \
    bash -c "echo '40,549755813888' > /etc/vcuda/vcuda.config; \
             cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             1> ${RESULT_DIR_CONTAINER}/scifin_vcuda_ojob_gv100.json" &

sleep 5

# allocate 60 % to pjob
docker run \
    --rm --name vcuda-fig11-scifin-pjob \
    --gpus "device=0" \
    --ipc host --pid host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/vcuda:0.2 \
    bash -c "echo '60,549755813888' > /etc/vcuda/vcuda.config; \
             cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/cuda_bs \
             1> ${RESULT_DIR_CONTAINER}/scifin_vcuda_pjob_gv100.json" &

sleep 120
echo "Testcase vCUDA scientific and financial computing finished"
cleanup
