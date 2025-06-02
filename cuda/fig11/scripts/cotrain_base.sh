#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw

mkdir -p ${RESULT_DIR}

cleanup() {
    docker exec base-fig11-cotrain-ojob pkill -9 -f 'python ./training/shufflenet.py'
    docker exec base-fig11-cotrain-pjob pkill -9 -f 'python ./training/shufflenet.py'
    docker kill base-fig11-cotrain-ojob
    docker kill base-fig11-cotrain-pjob
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name base-fig11-cotrain-pjob \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; python ./training/shufflenet.py \
             1> ${RESULT_DIR_CONTAINER}/cotrain_base_ojob_gv100.json" &

sleep 5
docker run \
    --rm --name base-fig11-cotrain-ojob \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; python ./training/shufflenet.py \
             1> ${RESULT_DIR_CONTAINER}/cotrain_base_pjob_gv100.json" &

sleep 300
echo "Testcase base (Native) co-training finished"
cleanup
