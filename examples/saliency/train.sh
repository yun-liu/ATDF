#!/bin/bash

set -x

LOG="logs/vgg16_sal_`date +%Y-%m-%d_%H-%M-%S`.log"
exec &> >(tee -a "$LOG")


../../build/tools/caffe train -solver solver.prototxt -weights 5stage-vgg.caffemodel -gpu 0
