import os

import torch


def _default_device() -> str:
    """Pick the training device.

    Override with AOQ_DEVICE (e.g. AOQ_DEVICE=cuda:1). Falls back to cuda:0 on a
    single-GPU box such as Colab, and to CPU when no GPU is visible.
    """
    env = os.environ.get("AOQ_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class globalVal:
    device = _default_device()
    loss = 0.0
    epoch = 0.0
