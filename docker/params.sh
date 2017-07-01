#!/bin/bash

REL_PATH_TO_SCRIPT=$(dirname "${BASH_SOURCE[0]}")
cd "${REL_PATH_TO_SCRIPT}"
NAME="intel_cancer"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"

VOLUMES="-v $(pwd)/..:/workdir"