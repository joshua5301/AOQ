#!/bin/bash
# Usage: bash AO_QAT/resnet_cifar10/run.sh <network> <n_bit> <quantize_downsample>
#   e.g. bash AO_QAT/resnet_cifar10/run.sh resnet20 2 True
# Runnable from any working directory, including the repo root.
#
# Env overrides: LOSS, EPOCHS, BATCH_SIZE, LR, SAVE, OPTIMIZER, WEIGHT_DECAY,
#                MOMENTUM, RESUME, PYTHON
#
#   LOSS=ce bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   OPTIMIZER=sgd LR=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   RESUME=1 SAVE=/content/drive/MyDrive/aoq bash AO_QAT/resnet_cifar10/run.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

NETWORK=${1:-resnet20}
N_BIT=${2:-2}
QUAN_DOWNSAMPLE=${3:-True}
EPOCHS=${EPOCHS:-250}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
SAVE=${SAVE:-./models}
LOSS=${LOSS:-kd}
OPTIMIZER=${OPTIMIZER:-adam}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
MOMENTUM=${MOMENTUM:-0.9}
PYTHON=${PYTHON:-python3}

# mirrors run_dir() in train.py so logs and checkpoints line up
TAG=${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
if [ "${LOSS}" != "kd" ]; then TAG=${TAG}_${LOSS}; fi
if [ "${OPTIMIZER}" != "adam" ]; then TAG=${TAG}_${OPTIMIZER}; fi
RESUME_FLAG=""
if [ -n "${RESUME}" ] && [ "${RESUME}" != "0" ]; then RESUME_FLAG="--resume"; fi
LOG_DIR=log/${TAG}
mkdir -p "${LOG_DIR}"

echo "[run] ${TAG}  epochs=${EPOCHS} loss=${LOSS} optimizer=${OPTIMIZER} lr=${LR}"
"${PYTHON}" train.py ${RESUME_FLAG} \
    --student="${NETWORK}" \
    --n_bit="${N_BIT}" \
    --quantize_downsample="${QUAN_DOWNSAMPLE}" \
    --loss="${LOSS}" \
    --optimizer="${OPTIMIZER}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --learning_rate="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --momentum="${MOMENTUM}" \
    --save="${SAVE}" \
    | tee -a "${LOG_DIR}/training.txt"
