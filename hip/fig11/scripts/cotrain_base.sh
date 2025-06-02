#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/hip/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw

mkdir -p ${RESULT_DIR}

cleanup() {
    docker kill base-hip-fig11-cotrain-ojob
    docker kill base-hip-fig11-cotrain-pjob
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name base-hip-fig11-cotrain-ojob \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             python ./training/shufflenet.py -n 1; \
             python ./training/shufflenet.py \
             1> ${RESULT_DIR_CONTAINER}/cotrain_base_ojob_mi50.json" &

sleep 5

docker run \
    --rm --name base-hip-fig11-cotrain-pjob \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             python ./training/shufflenet.py -n 1; \
             python ./training/shufflenet.py \
             1> ${RESULT_DIR_CONTAINER}/cotrain_base_pjob_mi50.json" &

sleep 400
echo "Testcase base (Native) co-training finished"
cleanup
