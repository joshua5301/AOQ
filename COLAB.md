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
| `QVR_MEASURE` | `sr` | `sr`, `cos2`, or `nagel`; none has a width knob |
| `QVR_GAIN` | `sq` | sensitivity weight: `sq`, `abs`, or `const` |
| `QVR_APPLY` | `decoupled` | `decoupled` or `coupled` (into `weight.grad`) |
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
`R = sum V(u) * gain^2` the rounding-induced loss variance and `u` the signed
distance to the nearest grid point. It needs a uniform grid, so
`W_QUANTIZER=lsq` is set for you when it is on.

```python
!QVR_LAMBDA=14 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
!QVR_LAMBDA=14 QVR_MEASURE=cos2 bash AO_QAT/resnet_cifar10/run.sh resnet20 2
```

It adds one line per epoch:

```
qvr    epoch   0  lambda 1.4000e+01  std 8.1e-01  force/grad 5.1e-02  pull/step 1.55e-06
```

**Tune by `force/grad`**, the share of the raw gradient the penalty
contributes. It is exactly linear in `QVR_LAMBDA`, so one probe epoch fixes
the scale. Target `0.05`; the useful band is roughly `[0.02, 0.5]`.

Do **not** tune by `pull/step`. It drifts with the weight distribution as
training settles — a real run fell 56x between epoch 0 and epoch 91 at fixed
`QVR_LAMBDA` — so it is a diagnostic, not a knob, and it is undefined under
`QVR_APPLY=coupled`.

#### `QVR_MEASURE` — where the force acts

All three have `V = 0` at the grid point and `V = 1/4` at the boundary, so the
barrier height is identical. They differ in where the force lives:

| `\|dV/du\|` | `u=0` (grid point) | `u=0.25` | `u=0.5` (boundary) |
|---|---|---|---|
| `nagel` | 0.000 | 0.500 | 1.000 |
| `sr` | 1.000 | 0.500 | 0.000 |
| `cos2` | 0.000 | 0.785 | 0.000 |

- `sr` — stochastic rounding, `V = |u|(1-|u|)`. Maximal force at the grid
  point (a cusp), so settled weights keep jittering.
- `cos2` — `V = sin^2(pi u)/4`, the leading Fourier mode of a Gaussian jitter
  and its width-independent limit. Zero force at both ends, peaks mid-bin, so
  settled weights are left alone.
- `nagel` — `V = u^2`, the oscillation-dampening penalty of Nagel et al. 2022.
  Roughly `sr`'s mirror image: zero force at the grid point, maximal at the
  boundary. **This is the prior-work arm.**

`QVR_GAIN=const` is *not* prior work — it is "QVR minus the sensitivity
weight", an ablation isolating what `gain^2` contributes. Use
`QVR_MEASURE=nagel` for the literature comparison. (Even that is not identical:
Nagel's penalty is separable, `sum_i u_i^2`, while QVR's global `sqrt` couples
every coordinate through a shared denominator.)

Peak forces differ across profiles, so **compare arms at matched `force/grad`,
not matched `QVR_LAMBDA`**. Measured starting points for `force/grad = 0.05`
at init: `nagel` ~3.6, `cos2` ~4.4, `sr` ~5.3 — but `force/grad` falls as
training settles, so aim higher (~14) if you want 0.05 at the *end*.

#### `QVR_APPLY` — coupled or decoupled

`decoupled` (default) writes `w -= lr*lam*grad` outside the optimizer, so
`lam` stays monotone under any optimizer — the fix AdamW makes for L2.
`coupled` adds the term to `weight.grad` so the optimizer normalises the sum;
that holds the penalty at a fixed share of the update, at the cost of
saturating once it dominates. Read `force/grad` for both, `pull/step` only for
`decoupled`.

#### `QVR_GAIN` — the sensitivity weight

`sq` (`(g*step)^2`) is the derived one and the only choice for which
`sqrt(R)` is a standard deviation. `abs` and `const` keep the identical
machinery and change the weighting alone. Note `const` needs a much smaller
`QVR_LAMBDA` — the weights are ~60x larger — so re-probe `force/grad`.

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
