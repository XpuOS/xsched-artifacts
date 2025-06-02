#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function build() {
    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    rm -rf ${ROOT}/build
    mkdir ${ROOT}/build
    cd ${ROOT}/build
    cmake -DCMAKE_BUILD_TYPE=Debug ..
    make
}

build