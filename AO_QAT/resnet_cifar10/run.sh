#!/bin/bash
# Usage: bash AO_QAT/resnet_cifar10/run.sh <network> <n_bit> <quantize_downsample>
#   e.g. bash AO_QAT/resnet_cifar10/run.sh resnet20 2 True
# Runnable from any working directory, including the repo root.
#
# Env overrides: LOSS, W_QUANTIZER, QVR_LAMBDA, QVR_MEASURE, QVR_GAIN,
#                QVR_APPLY,
#                OPTIMIZER, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, MOMENTUM,
#                SAVE, RESUME, PYTHON
#
#   LOSS=ce bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   OPTIMIZER=sgd LR=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   RESUME=1 SAVE=/content/drive/MyDrive/aoq bash AO_QAT/resnet_cifar10/run.sh
#
# QVR (Quantization-aware Variance Regularization) penalises L + lam*sqrt(R),
# R = sum p(1-p) gain^2 the rounding-induced loss variance. It needs a uniform
# grid, so W_QUANTIZER=lsq is set automatically whenever QVR_LAMBDA is on --
# AOQ decouples its thresholds from its levels and has no such grid.
#
#   QVR_LAMBDA=10 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   QVR_LAMBDA=10 QVR_MEASURE=cos2 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
#   QVR_LAMBDA=10 QVR_MEASURE=nagel bash .../run.sh resnet20 2   # prior work
#   QVR_LAMBDA=10 QVR_APPLY=coupled QVR_GAIN=const bash .../run.sh resnet20 2
#
# Tune QVR_LAMBDA by the logged force/grad -- the share of the raw gradient the
# penalty contributes. It is exactly linear in QVR_LAMBDA, so one probe epoch
# fixes the scale. Target 0.05, useful band roughly [0.02, 0.5].
#
# Do NOT tune by pull/step: it drifts with the weight distribution as training
# settles (a real run fell 56x between epoch 0 and 91 at fixed lambda), so it
# is a diagnostic, not a knob. It is also undefined under QVR_APPLY=coupled.
#
# QVR_MEASURE picks the V profile: sr peaks at the grid point, cos2 mid-bin,
# nagel at the boundary (the prior-work dampening penalty). Their peak forces
# differ, so compare arms at matched force/grad, not matched QVR_LAMBDA.
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
QVR_LAMBDA=${QVR_LAMBDA:-0}
QVR_MEASURE=${QVR_MEASURE:-sr}
QVR_GAIN=${QVR_GAIN:-sq}
QVR_APPLY=${QVR_APPLY:-decoupled}
PYTHON=${PYTHON:-python3}

# QVR is defined on a uniform grid, so it implies the LSQ quantizer.
if [ "${QVR_LAMBDA}" != "0" ]; then
    W_QUANTIZER=${W_QUANTIZER:-lsq}
    if [ "${W_QUANTIZER}" != "lsq" ]; then
        echo "[run] QVR_LAMBDA needs W_QUANTIZER=lsq, got ${W_QUANTIZER}" >&2
        exit 1
    fi
fi
W_QUANTIZER=${W_QUANTIZER:-aoq}

# mirrors run_dir() in train.py, in the same order, so logs and checkpoints
# line up. "%g" here matches Python's "{:g}" there.
QVR_LAMBDA_G=$(printf "%g" "${QVR_LAMBDA}")
TAG=${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
if [ "${W_QUANTIZER}" != "aoq" ]; then TAG=${TAG}_${W_QUANTIZER}; fi
if [ "${LOSS}" != "kd" ]; then TAG=${TAG}_${LOSS}; fi
if [ "${QVR_LAMBDA}" != "0" ]; then
    TAG=${TAG}_qvr${QVR_LAMBDA_G}_${QVR_MEASURE}
    if [ "${QVR_GAIN}" != "sq" ]; then TAG=${TAG}_${QVR_GAIN}; fi
    if [ "${QVR_APPLY}" != "decoupled" ]; then TAG=${TAG}_${QVR_APPLY}; fi
fi
if [ "${OPTIMIZER}" != "adam" ]; then TAG=${TAG}_${OPTIMIZER}; fi
RESUME_FLAG=""
if [ -n "${RESUME}" ] && [ "${RESUME}" != "0" ]; then RESUME_FLAG="--resume"; fi
LOG_DIR=log/${TAG}
mkdir -p "${LOG_DIR}"

echo "[run] ${TAG}  epochs=${EPOCHS} loss=${LOSS} quantizer=${W_QUANTIZER} optimizer=${OPTIMIZER} lr=${LR}"
"${PYTHON}" train.py ${RESUME_FLAG} \
    --student="${NETWORK}" \
    --n_bit="${N_BIT}" \
    --quantize_downsample="${QUAN_DOWNSAMPLE}" \
    --w_quantizer="${W_QUANTIZER}" \
    --loss="${LOSS}" \
    --qvr_lambda="${QVR_LAMBDA}" \
    --qvr_measure="${QVR_MEASURE}" \
    --qvr_gain="${QVR_GAIN}" \
    --qvr_apply="${QVR_APPLY}" \
    --optimizer="${OPTIMIZER}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --learning_rate="${LR}" \
    --weight_decay="${WEIGHT_DECAY}" \
    --momentum="${MOMENTUM}" \
    --save="${SAVE}" \
    | tee -a "${LOG_DIR}/training.txt"
