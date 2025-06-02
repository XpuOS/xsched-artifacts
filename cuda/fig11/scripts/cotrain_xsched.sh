#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw
BUILD_CONTAINER=${TESTCASE_ROOT_CONTAINER}/build/xsched
OUTPUT_CONTAINER=${TESTCASE_ROOT_CONTAINER}/output/xsched

mkdir -p ${RESULT_DIR}
rm -rf /dev/shm/__IPC_SHM__*

# build xsched
docker run -it --rm --name xsched-fig11-build \
           --gpus "device=0" \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TESTCASE_ROOT_CONTAINER} && \
                    make -C ${ROOT_CONTAINER}/sys/xsched cuda BUILD_PATH=${BUILD_CONTAINER} OUTPUT_PATH=${OUTPUT_CONTAINER}"

cleanup() {
    docker exec xsched-fig11-xserver rm -rf /dev/shm/__IPC_SHM__*
    docker exec xsched-fig11-cotrain-ojob pkill -9 -f 'python ./training/shufflenet.py'
    docker exec xsched-fig11-cotrain-pjob pkill -9 -f 'python ./training/shufflenet.py'
    docker exec xsched-fig11-xserver pkill -9 -f 'bin/xserver'
    docker kill xsched-fig11-cotrain-ojob
    docker kill xsched-fig11-cotrain-pjob
    docker kill xsched-fig11-xserver
}
trap 'cleanup; kill $$' SIGINT

# start xserver
docker run --rm --name xsched-fig11-xserver \
           --gpus "device=0" \
           --network host --ipc host --pid host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/xsched-cuda:0.3 \
           bash -c "rm -rf /dev/shm/__IPC_SHM__*; \
                    ${OUTPUT_CONTAINER}/bin/xserver HPF 50000" &

sleep 2

# start low-priority ojob
docker run --rm --name xsched-fig11-cotrain-ojob \
           --gpus "device=0" \
           --network host --ipc host --pid host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           -e SINGLE_STREAM=ON \
           -e XSCHED_POLICY=GBL \
           -e XSCHED_AUTO_XQUEUE=ON \
           -e XSCHED_AUTO_XQUEUE_LEVEL=2 \
           -e XSCHED_AUTO_XQUEUE_THRESHOLD=64 \
           -e XSCHED_AUTO_XQUEUE_BATCH_SIZE=32 \
           -e XSCHED_AUTO_XQUEUE_PRIORITY=0 \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
                    LD_LIBRARY_PATH=${OUTPUT_CONTAINER}/lib python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_xsched_ojob_gv100.json" &

sleep 5

# start high-priority pjob
docker run --rm --name xsched-fig11-cotrain-pjob \
           --gpus "device=0" \
           --network host --ipc host --pid host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           -e SINGLE_STREAM=ON \
           -e XSCHED_POLICY=GBL \
           -e XSCHED_AUTO_XQUEUE=ON \
           -e XSCHED_AUTO_XQUEUE_LEVEL=1 \
           -e XSCHED_AUTO_XQUEUE_THRESHOLD=512 \
           -e XSCHED_AUTO_XQUEUE_BATCH_SIZE=128 \
           -e XSCHED_AUTO_XQUEUE_PRIORITY=1 \
           shenwhang/xsched-cuda:0.3 \
           bash -c "cd ${TESTCASE_ROOT_CONTAINER}; \
                    LD_LIBRARY_PATH=${OUTPUT_CONTAINER}/lib python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_xsched_pjob_gv100.json" &

sleep 300
echo "Testcase XSched co-training finished"
cleanup
