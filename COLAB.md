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
| `LOSS` | `kd` | `kd` distills from the teacher, `ce` uses the hard labels |
| `W_QUANTIZER` | `aoq` | weight quantizer; `lsq` is the uniform baseline |
| `QVR_LAMBDA` | `0` | QVR penalty weight (0 = off); implies `W_QUANTIZER=lsq` |
| `QVR_MEASURE` | `sr` | `sr` or `cos2`; neither has a width knob |
| `OPTIMIZER` | `adam` | `adam`, `adamw`, or `sgd` |
| `EPOCHS` | `250` | training epochs |
| `BATCH_SIZE` | `256` | |
| `LR` | `1e-3` | initial learning rate |
| `WEIGHT_DECAY` | `0` | used by `adamw` and `sgd` |
| `MOMENTUM` | `0.9` | `sgd` only |
| `SAVE` | `./models` | checkpoint directory |
| `RESUME` | unset | set to `1` to continue from a checkpoint |

```python
!LOSS=ce bash AO_QAT/resnet_cifar10/run.sh resnet20 2
!OPTIMIZER=sgd LR=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
!OPTIMIZER=adamw WEIGHT_DECAY=0.01 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

The default `kd` loss is pure KL against the teacher and **ignores the labels
entirely** (`utils/KD_loss.py`: "Target is ignored at training time"). `ce`
scores against the labels instead, needs no teacher, and therefore skips its
forward pass each step.

### QVR

Quantization-aware Variance Regularization penalises `L + lam*sqrt(R)`, with
`R = sum p(1-p) gain^2` the rounding-induced loss variance. It needs a uniform
grid, so `W_QUANTIZER=lsq` is set for you when it is on.

```python
!QVR_LAMBDA=10 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
!QVR_LAMBDA=10 QVR_MEASURE=cos2 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

It adds one line per epoch:

```
qvr    epoch   0  lambda 1.0000e+01  std 8.1e-01  force/grad 1.02e-01  pull/step 1.55e-06
```

`QVR_LAMBDA` is QVR's **only** hyperparameter — neither measure has a width
knob. Tune it by `pull/step`, the fraction of a quantization bin the penalty
drags a weight in one step: a 250-epoch run at batch 256 is ~49k steps, so
`pull/step ~ 2e-5` is roughly "one bin over the whole run". `force_ratio` near
0.05 is the other target, and it is exactly linear in `QVR_LAMBDA`, so one
probe epoch is enough to rescale.

The two measures differ in where the force acts: `sr` is maximal at the grid
point (a kink, so settled weights keep jittering), `cos2` is zero there and
peaks mid-bin (settled weights are left alone). Their barrier heights are
identical, but peak force differs by `4/pi`, so **compare arms at matched
`force_ratio`, not matched `QVR_LAMBDA`**.

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

Two lines per epoch, one for training and one for the test set:

```
train  epoch   0  lr 9.960e-04  loss 6.5900e-01  acc@1  79.33  acc@5  98.87  9s
test   epoch   0  loss 5.8000e-01  acc@1  81.01  acc@5  98.95
```

```python
!grep "^.*test   epoch" AO_QAT/resnet_cifar10/log/resnet20_2bit_quantize_downsample_True/training.txt | tail -5
```

To compare arms, `results.sh` pulls the best test accuracy out of every run:

```python
!bash AO_QAT/resnet_cifar10/results.sh
```

```
best@1   epoch  last@1  done  fp@1    run
-------  -----  ------  ----  ------  ---
83.50    11     83.50   12    91.48   ..._lsq_qvr10_cos2
83.04    29     83.04   30    91.48   ..._lsq_qvr10_sr
82.00    29     82.00   30    91.48   resnet20_2bit_quantize_downsample_True
```

Sorted best first. `done` is how many epochs finished — **check it before
comparing**, since a run cut short by a disconnect is not comparable. `fp@1`
is the full-precision teacher, for reference. Takes an optional glob
(`results.sh '*qvr*'`) and reads `$LOG` if the logs live on Drive.

`test epoch -2` is the full-precision teacher, evaluated once before training
starts — skip it when reading off the best student accuracy.

Pass `--print_freq=50` (via `train.py` directly) for the old per-batch progress
lines and the full model dump.

## Gotchas

- **`--epochs=1` trains nothing.** The schedule is
  `LambdaLR(1 - step/epochs)` and `scheduler.step()` runs at the top of each
  epoch, so a single-epoch run has a learning rate of exactly 0. Use `EPOCHS=2`
  or more for a smoke test. The last epoch of any run is likewise at lr 0.
- **Keep `EPOCHS=250`.** AOQ's three stages are hardcoded at absolute epochs 50
  and 150, and the alpha cosine has a fixed period of 100. Those encode the
  paper's 1/5, 2/5, 2/5 split only at 250 epochs; at any other budget the
  dampening stage silently changes length.
