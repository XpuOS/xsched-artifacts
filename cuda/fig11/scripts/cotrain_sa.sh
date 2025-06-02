#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw

mkdir -p ${RESULT_DIR}

cleanup() {
    docker exec xsched-cuda-fig11-cotrain-sa pkill -9 -f 'python ./training/shufflenet.py'
    docker kill xsched-cuda-fig11-cotrain-sa
}
trap 'cleanup; kill $$' SIGINT

docker run \
    --rm --name xsched-cuda-fig11-cotrain-sa \
    --gpus "device=0" \
    --ipc host --network host \
    -v ${ROOT}:${ROOT_CONTAINER} \
    shenwhang/xsched-cuda:0.3 \
    bash -c "cd ${TESTCASE_ROOT_CONTAINER}; python ./training/shufflenet.py \
             1> ${RESULT_DIR_CONTAINER}/cotrain_sa_gv100.json" &

sleep 300
echo "Testcase standalone co-training finished"
cleanup
