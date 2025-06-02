#!/bin/bash

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)

for file in `ls ${ROOT}/assets/models/ascend | grep .onnx`
do
    model=${ROOT}/assets/models/ascend/${file}
    if test -f $model
    then
        slice=${model%.onnx}
        echo "converting $model to $slice.om"
        atc --model=$model --framework=5 --output=$slice --soc_version=$1
    fi
done
