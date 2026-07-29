#!/bin/bash
# Usage: bash AO_QAT/resnet_cifar10/run.sh <network> <n_bit> <quantize_downsample>
#   e.g. bash AO_QAT/resnet_cifar10/run.sh resnet20 2 True
# Runnable from any working directory, including the repo root.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

NETWORK=${1:-resnet20}
N_BIT=${2:-2}
QUAN_DOWNSAMPLE=${3:-True}
LOG_DIR=log/${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
mkdir -p ${LOG_DIR}
python3 train.py --batch_size=256 --learning_rate=1e-3 --epochs=250 --weight_decay=0 --momentum=0.9 --student=${NETWORK} --n_bit=${N_BIT} --quantize_downsample=${QUAN_DOWNSAMPLE} | tee -a ${LOG_DIR}/training.txt
