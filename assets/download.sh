#!/bin/bash

URL=https://zenodo.org/records/15578003/files/xsched-artifact-assets.zip?download=1
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)

wget -O ${ROOT}/assets/xsched-artifact-assets.zip ${URL}
unzip ${ROOT}/assets/xsched-artifact-assets.zip -d ${ROOT}
