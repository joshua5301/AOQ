#!/bin/bash
# Fetch the option A pretrained CIFAR-10 checkpoints used as QAT initialization
# and as the distillation teacher. Source: akamaster/pytorch_resnet_cifar10.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
BASE=https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models

mkdir -p "${DIR}"
for f in resnet20-12fca82f.th resnet32-d509ac18.th resnet44-014dd654.th resnet110-1d1ed7c2.th; do
    if [ -f "${DIR}/${f}" ]; then
        echo "skip ${f} (already present)"
    else
        echo "downloading ${f}"
        curl -fsSL -o "${DIR}/${f}" "${BASE}/${f}"
    fi
done
echo "done -> ${DIR}"
