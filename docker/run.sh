#!/bin/bash

source params.sh

nvidia-docker run --rm -it ${NET} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash