#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig12/raw

trap 'kill -9 $(jobs -p); exit' INT
rm -rf /dev/shm/__IPC_SHM__*
mkdir -p ${RESULT_DIR}

${TESTCASE_ROOT}/output/bin/xserver LAX 50000 &
SERVER_PID=$!
echo "SERVER_PID: ${SERVER_PID}"
sleep 2

export LD_LIBRARY_PATH=/opt/intel/openvino_2024.4.0/runtime/lib/intel64:/usr/local/lib:${TESTCASE_ROOT}/output/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

export XSCHED_POLICY=GBL
export XSCHED_AUTO_XQUEUE=ON

export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=64
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=32
export XSCHED_AUTO_XQUEUE_PRIORITY=3
export XSCHED_AUTO_XQUEUE_LAXITY=10000 # 10 ms
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so python3 ${TESTCASE_ROOT}/lfbw/lfbw/lfbw_xsched.py \
    --ov-device NPU \
    --threshold 40 --no-ondemand --no-postprocess \
    --width 640 --height 360 --fps 25 \
    --background-image ${TESTCASE_ROOT}/lfbw/background.jpg \
    --foreground-image ${TESTCASE_ROOT}/lfbw/foreground.jpg \
    --foreground-mask-image ${TESTCASE_ROOT}/lfbw/foreground-mask.png \
    --ov-model-xml ${ROOT}/assets/models/selfie_multiclass_256x256.xml \
    --ov-model-bin ${ROOT}/assets/models/selfie_multiclass_256x256.bin \
    --input-video ${ROOT}/assets/media/conference_video.mp4 \
    2> ${RESULT_DIR}/lfbw_xsched.json &
LFBW_PID=$!

sleep 10
export XSCHED_AUTO_XQUEUE_THRESHOLD=1
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=1
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_AUTO_XQUEUE_LAXITY=100000000 # nearly infinite
LD_PRELOAD=${TESTCASE_ROOT}/output/lib/libshimlevelzero.so ${TESTCASE_ROOT}/output/bin/stream \
    -m ${ROOT}/assets/models/ggml-medium.en.bin \
    -oved "NPU" \
    -wf ${ROOT}/assets/media/conference_audio.wav \
    -t 8 --step 3000 --length 3000 \
    2> ${RESULT_DIR}/whisper_xsched.json &
WHISPER_PID=$!

sleep 120
kill -9 $LFBW_PID $WHISPER_PID $SERVER_PID
wait $LFBW_PID $WHISPER_PID $SERVER_PID
