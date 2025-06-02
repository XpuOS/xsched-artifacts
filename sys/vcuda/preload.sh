#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

function preload() {
    rm -f /etc/ld.so.preload
    touch /etc/ld.so.preload

    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
    cd ${ROOT}/build

    rm -rf /vcuda
    mkdir -p /vcuda

    rm -rf /etc/vcuda
    mkdir -p /etc/vcuda
    echo "100,1099511627776" > /etc/vcuda/vcuda.config # 100% utilization, 1TB memory
    touch /etc/vcuda/pids.config

    cp libcuda-control.so /vcuda/libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 /vcuda/libnvidia-ml.so.1
    cp libcuda-control.so /vcuda/libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so /vcuda/libnvidia-ml.so
    cp libcuda-control.so /vcuda/libcuda.so.1
    patchelf --set-soname libcuda.so.1 /vcuda/libcuda.so.1
    cp libcuda-control.so /vcuda/libcuda.so
    patchelf --set-soname libcuda.so /vcuda/libcuda.so
    cp libcuda-control.so /vcuda/libcontroller.so
    patchelf --set-soname libcontroller.so /vcuda/libcontroller.so
    echo -e "/vcuda/libcontroller.so\n/vcuda/libcuda.so\n/vcuda/libcuda.so.1\n/vcuda/libnvidia-ml.so\n/vcuda/libnvidia-ml.so.1" >/etc/ld.so.preload

    cd ..
}

preload