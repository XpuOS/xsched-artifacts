#!/bin/bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

cd ${ROOT}/sys/xsched
./output/bin/xserver HPF 50000
