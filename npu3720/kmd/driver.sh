#!/bin/bash

build() {
	make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
}

install() {
    if [ ! -e "./intel_vpu.ko" ]; then
        echo "error: intel_vpu.ko not found, please build first"
        exit 1
    fi
	rmmod intel_vpu
    insmod ./intel_vpu.ko
}

clean() {
    make -C /lib/modules/$(uname -r)/build M=$(pwd) clean
}

help() {
	echo "usage: ./driver.sh [build|install|clean]"
    exit 1
}

if [ $# -eq 0 ]; then
	echo "lack command"
    help
fi

if [ "$1" = "build" ]; then
    build
    echo "build success"

elif [ "$1" = "install" ]; then
    if [ "$USER" != "root" ]; then
        echo "error: should be run as root"
        exit 1
    fi
    install
    echo "$(dmesg | grep intel_vpu)"
    echo "install success"

elif [ "$1" = "clean" ]; then
	clean
    echo "clean success"

else
	echo "error: unrecognized command \"$1\""
	help
fi
