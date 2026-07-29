#!/bin/bash
# Usage: bash AO_QAT/resnet_imagenet/run.sh <network> <n_bit> <quantize_downsample> [imagenet_dir]
#   e.g. bash AO_QAT/resnet_imagenet/run.sh resnet18 2 True /path/to/imagenet
# Runnable from any working directory, including the repo root.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

NETWORK=${1:-resnet18}
N_BIT=${2:-2}
QUAN_DOWNSAMPLE=${3:-True}
DATA=${4:-${IMAGENET_DIR:?set IMAGENET_DIR or pass the ImageNet path as the 4th argument}}
LOG_DIR=log/${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
mkdir -p ${LOG_DIR}
python3 train.py --data=${DATA} --batch_size=256 --learning_rate=1.25e-3 --epochs=300 --weight_decay=0 --momentum=0.9 --student=${NETWORK} --n_bit=${N_BIT} --quantize_downsample=${QUAN_DOWNSAMPLE} | tee -a ${LOG_DIR}/training.txt
