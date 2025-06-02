#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig11/raw

ROOT_CONTAINER=/xsched-artifacts
TESTCASE_ROOT_CONTAINER=${ROOT_CONTAINER}/cuda/fig11
RESULT_DIR_CONTAINER=${ROOT_CONTAINER}/results/fig11/raw

mkdir -p ${RESULT_DIR}

cleanup() {
    docker exec vcuda-fig11-cotrain-ojob pkill -9 -f 'python ./training/shufflenet.py'
    docker exec vcuda-fig11-cotrain-pjob pkill -9 -f 'python ./training/shufflenet.py'
    docker kill vcuda-fig11-cotrain-ojob
    docker kill vcuda-fig11-cotrain-pjob
}
trap 'cleanup; kill $$' SIGINT

# allocate 20 % to ojob
docker run --rm --name vcuda-fig11-cotrain-ojob \
           --gpus "device=0" \
           --pid host --ipc host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/vcuda:0.2 \
           bash -c "echo '20,549755813888' > /etc/vcuda/vcuda.config; \
                    cd ${TESTCASE_ROOT_CONTAINER}; \
                    python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_vcuda_ojob_gv100.json" &

sleep 5

# allocate 80 % to pjob
docker run --rm --name vcuda-fig11-cotrain-pjob \
           --gpus "device=0" \
           --pid host --ipc host \
           -v ${ROOT}:${ROOT_CONTAINER} \
           shenwhang/vcuda:0.2 \
           bash -c "echo '80,549755813888' > /etc/vcuda/vcuda.config; \
                    cd ${TESTCASE_ROOT_CONTAINER}; \
                    python ./training/shufflenet.py \
                    1> ${RESULT_DIR_CONTAINER}/cotrain_vcuda_pjob_gv100.json" &

sleep 300
echo "Testcase vCUDA co-training finished"
cleanup
