# Running on Colab

The pretrained checkpoints and CIFAR-10 are both committed, so a clone is all
you need — no download step.

## Setup

Runtime → Change runtime type → **GPU**, then:

```python
!git clone https://github.com/joshua5301/AOQ.git
%cd /content/AOQ
```

Colab already ships a CUDA torch, and `requirements.txt` only sets lower bounds
below Colab's versions, so installing it is optional and safe:

```python
!pip install -q -r requirements.txt
```

## Train

```python
!bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

Positional args are `<network> <n_bit> <quantize_downsample>`; networks are
`resnet20`, `resnet32`, `resnet44`, `resnet110`.

## Options

Everything else is an env var, set inline before `bash`:

| var | default | meaning |
|---|---|---|
| `OPTIMIZER` | `adam` | `adam`, `adamw`, or `sgd` |
| `EPOCHS` | `250` | training epochs |
| `BATCH_SIZE` | `256` | |
| `LR` | `1e-3` | initial learning rate |
| `WEIGHT_DECAY` | `0` | used by `adamw` and `sgd` |
| `MOMENTUM` | `0.9` | `sgd` only |
| `SAVE` | `./models` | checkpoint directory |
| `RESUME` | unset | set to `1` to continue from a checkpoint |

```python
!OPTIMIZER=sgd LR=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
!OPTIMIZER=adamw WEIGHT_DECAY=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

`adamw` with `WEIGHT_DECAY=0` is exactly Adam — the only difference between
them is how the decay term is applied, so it does nothing at 0. `sgd` needs its
own learning rate; `1e-3` is tuned for Adam and will be far too small.

## Surviving disconnects

Colab kills long runs, and 250 epochs will not finish in one session. Put
checkpoints on Drive and resume:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!SAVE=/content/drive/MyDrive/aoq bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

Re-running the same command after a disconnect **starts from scratch and
overwrites** — it warns when it does. Add `RESUME=1` to continue instead:

```python
!RESUME=1 SAVE=/content/drive/MyDrive/aoq bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

## Results

| what | where |
|---|---|
| per-run log | `AO_QAT/resnet_cifar10/log/<TAG>/training.txt` |
| all runs, appended | `AO_QAT/resnet_cifar10/log/log.txt` |
| checkpoints | `<SAVE>/<TAG>/checkpoint.pth.tar`, `model_best.pth.tar` |

`<TAG>` is `resnet20_2bit_quantize_downsample_True`, plus `_sgd` / `_adamw`
when the optimizer is not the default — so the arms never overwrite each
other.

```python
!grep "acc@1" AO_QAT/resnet_cifar10/log/resnet20_2bit_quantize_downsample_True/training.txt | tail -5
```

The first `acc@1` line is the full-precision teacher, printed once before
training starts — skip it when reading off the best student accuracy.

## Gotchas

- **`--epochs=1` trains nothing.** The schedule is
  `LambdaLR(1 - step/epochs)` and `scheduler.step()` runs at the top of each
  epoch, so a single-epoch run has a learning rate of exactly 0. Use `EPOCHS=2`
  or more for a smoke test. The last epoch of any run is likewise at lr 0.
- **Keep `EPOCHS=250`.** AOQ's three stages are hardcoded at absolute epochs 50
  and 150, and the alpha cosine has a fixed period of 100. Those encode the
  paper's 1/5, 2/5, 2/5 split only at 250 epochs; at any other budget the
  dampening stage silently changes length.
