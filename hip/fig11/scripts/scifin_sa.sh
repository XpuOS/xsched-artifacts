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
    docker kill xsched-fig11-sci-sa
    docker kill xsched-fig11-fin-sa
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name xsched-fig11-sci-sa \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/hip_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             2> ${RESULT_DIR_CONTAINER}/scifin_sa_ojob_mi50.json" &
sleep 120
cleanup
echo "Testcase standalone scientific computing finished, start financial computing"

docker run \
    --rm --name xsched-fig11-fin-sa \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             ${OUTPUT_CONTAINER}/bin/hip_bs \
             2> ${RESULT_DIR_CONTAINER}/scifin_sa_pjob_mi50.json" &

sleep 120
echo "Testcase standalone scientific and financial computing finished"
cleanup
