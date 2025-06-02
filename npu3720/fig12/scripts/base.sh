#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig12/raw

trap 'kill -9 $(jobs -p); exit' INT
mkdir -p ${RESULT_DIR}

export LD_LIBRARY_PATH=/opt/intel/openvino_2024.4.0/runtime/lib/intel64:/usr/local/lib:${TESTCASE_ROOT}/output/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

${TESTCASE_ROOT}/output/bin/stream \
    -m ${ROOT}/assets/models/ggml-medium.en.bin \
    -oved "NPU" \
    -wf ${ROOT}/assets/media/conference_audio.wav \
    -t 8 --step 3000 --length 3000 \
    2> ${RESULT_DIR}/whisper_base.json &
WHISPER_PID=$!

sleep 20 # wait for whisper to complete model building
python3 ${TESTCASE_ROOT}/lfbw/lfbw/lfbw_xsched.py \
    --ov-device NPU \
    --threshold 40 --no-ondemand --no-postprocess \
    --width 640 --height 360 --fps 25 \
    --background-image ${TESTCASE_ROOT}/lfbw/background.jpg \
    --foreground-image ${TESTCASE_ROOT}/lfbw/foreground.jpg \
    --foreground-mask-image ${TESTCASE_ROOT}/lfbw/foreground-mask.png \
    --ov-model-xml ${ROOT}/assets/models/selfie_multiclass_256x256.xml \
    --ov-model-bin ${ROOT}/assets/models/selfie_multiclass_256x256.bin \
    --input-video ${ROOT}/assets/media/conference_video.mp4 \
    2> ${RESULT_DIR}/lfbw_base.json &
LFBW_PID=$!

sleep 120
kill -9 $LFBW_PID $WHISPER_PID
wait $LFBW_PID $WHISPER_PID
