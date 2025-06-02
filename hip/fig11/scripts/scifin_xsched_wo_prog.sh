#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/hip/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
BUILD_CONTAINER=${TESTCASE_ROOT_CONTAINER}/build
OUTPUT_CONTAINER=${TESTCASE_ROOT_CONTAINER}/output
XSCHED_SHIM=${OUTPUT_CONTAINER}/xsched/lib/libshimhip.so

mkdir -p ${RESULT_DIR}
rm -rf /dev/shm/__IPC_SHM__*

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
             make BUILD_PATH=${BUILD_CONTAINER}/scifin OUTPUT_PATH=${OUTPUT_CONTAINER}/scifin"

# build xsched
docker run \
    -it --rm --name xsched-fig11-build \
    --ipc host --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER} && \
             make -C ${ROOT_CONTAINER}/sys/xsched hip \
             BUILD_PATH=${BUILD_CONTAINER}/xsched OUTPUT_PATH=${OUTPUT_CONTAINER}/xsched"

cleanup() {
    docker exec xsched-fig11-xserver rm -rf /dev/shm/__IPC_SHM__*
    docker kill xsched-fig11-scifin-ojob
    docker kill xsched-fig11-scifin-pjob
    docker kill xsched-fig11-xserver
}
trap 'cleanup; kill $$' SIGINT

# start xserver
docker run \
    --rm --name xsched-fig11-xserver \
    --network host --ipc host --pid host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "rm -rf /dev/shm/__IPC_SHM__*; \
             ${OUTPUT_CONTAINER}/xsched/bin/xserver HPF 50000" &

sleep 2

docker run \
    --rm --name xsched-fig11-scifin-ojob \
    --network host --ipc host --pid host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    -e XSCHED_POLICY=GBL \
    -e XSCHED_AUTO_XQUEUE=ON \
    -e XSCHED_AUTO_XQUEUE_THRESHOLD=1 \
    -e XSCHED_AUTO_XQUEUE_BATCH_SIZE=1 \
    -e XSCHED_AUTO_XQUEUE_PRIORITY=0 \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             LD_PRELOAD=${XSCHED_SHIM} ${OUTPUT_CONTAINER}/scifin/bin/hip_euler ${ROOT_CONTAINER}/assets/data/missile.domn.0.2M \
             2> ${RESULT_DIR_CONTAINER}/scifin_xsched_woprog_ojob_mi50.json" &
sleep 5

docker run \
    --rm --name xsched-fig11-scifin-pjob \
    --network host --ipc host --pid host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v ${ROOT}:${ROOT_CONTAINER} \
    -e XSCHED_POLICY=GBL \
    -e XSCHED_AUTO_XQUEUE=ON \
    -e XSCHED_AUTO_XQUEUE_THRESHOLD=1 \
    -e XSCHED_AUTO_XQUEUE_BATCH_SIZE=1 \
    -e XSCHED_AUTO_XQUEUE_PRIORITY=1 \
    rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_1.12.1 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
             LD_PRELOAD=${XSCHED_SHIM} ${OUTPUT_CONTAINER}/scifin/bin/hip_bs \
             2> ${RESULT_DIR_CONTAINER}/scifin_xsched_woprog_pjob_mi50.json" &

sleep 120
echo "Testcase XSched scientific and financial computing (without progressive submission) finished"
cleanup
