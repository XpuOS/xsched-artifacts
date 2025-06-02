#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/hip/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
BUILD_CONTAINER=${TESTCASE_ROOT_CONTAINER}/build/scifin
OUTPUT_CONTAINER=${TESTCASE_ROOT_CONTAINER}/output/scifin

mkdir -p ${RESULT_DIR}

# build scifin
docker run \
    -it --rm --name xsched-fig11-scifin-build \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER} && \
             make BUILD_PATH=${BUILD_CONTAINER} OUTPUT_PATH=${OUTPUT_CONTAINER}"

cleanup() {
    docker kill base-fig11-scifin-ojob
    docker kill base-fig11-scifin-pjob
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name base-fig11-scifin-ojob \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/hip_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             2> ${RESULT_DIR_CONTAINER}/scifin_base_ojob_mi50.json" &
sleep 5

docker run \
    --rm --name base-fig11-scifin-pjob \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/hip_bs \
             2> ${RESULT_DIR_CONTAINER}/scifin_base_pjob_mi50.json" &

sleep 120
echo "Testcase base (Native) scientific and financial computing finished"
cleanup
