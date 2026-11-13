# Train a TopK SAE baseline on GraphCast activations

This trainer reuses `graphcast_interpretability.model.SAE` and the authors'
decoder-gradient projection and normalization helpers. The original SAE class
and hurricane demo are unchanged. GraphCast is fixed: this stage trains the SAE
on saved internal activation vectors, and does not run GraphCast itself.

For a first Colab run, open
[the training notebook](../notebooks/train_topk_colab.ipynb). It installs this
branch, mounts Google Drive, checks the data, trains, plots the loss, and reloads
the best model. You must first extract real GraphCast activations using the
original hurricane notebook. This trainer does not ship an activation dataset.

## 1. Prepare activation files

Use floating-point `.npy` files shaped `[N, 512]` or `[N, 1, 512]` from one
processing step/hook. The default prefix is `layer0008_`, matching the authors'
demo. The hook uses a zero-based processing-step counter; record the exact hook
and prefix when describing your experiment rather than relying on a layer name
alone. Train and compare both SAEs on the same captures.

At extraction, convert values numerically before saving:

```python
arr = np.asarray(x).astype(np.float32)
np.save(output_path, arr)
```

Legacy NumPy `|V2` arrays can contain raw bfloat16 bytes. The trainer rejects
them: `view(np.float16)` does not convert bfloat16 values correctly. Re-extract
as float32, or recover the original dtype with a verified numerical conversion
before training.

The default split keeps whole calendar days together and holds out the latest
20% of dates. Dated filenames must contain `_tYYYY-MM-DDT...` or `_tYYYYMMDDT...`.
At least two different dates are required; several dates spanning different
weather conditions are a better pilot. An epoch means one pass through every
training vector, including each file's final short batch. Repeated epochs reuse
these vectors; `samples_seen` counts presentations, not unique observations.

For event-based experiments, put whole storms/events in separate training and
validation directories. Separate days can still belong to one storm: a date
split alone does not ensure event independence. Keep a separate final test set
for the eventual TopK-versus-KAN comparison and do not pass it to this trainer.

## 2. Install and train

In a terminal inside your checkout:

```bash
python -m pip install -e '.[training]'
python -m graphcast_interpretability.train_topk \
    --data-dir /path/to/graphcast_acts \
    --output-dir runs/topk_layer8 \
    --latent 4096 --k 32 --k-aux 512 \
    --batch-size 256 --epochs 5 --lr 0.0003
```

For independent directories instead of automatic date splitting:

```bash
python -m graphcast_interpretability.train_topk \
    --train-dir /path/to/train_activations \
    --val-dir /path/to/validation_activations \
    --output-dir runs/topk_layer8
```

Directories can be repeated. File discovery is not recursive. Explicit splits
reject duplicate paths and shared dates when timestamps are present. For files
without timestamps, you are responsible for assigning whole dates/events to
the correct directory. Arrays are streamed from disk and checked for finite
values; the entire activation corpus is never loaded into memory.

Python use is equivalent:

```python
from graphcast_interpretability.train_topk import (
    TrainConfig, find_activation_files, split_by_day, train_topk,
)

train_files, val_files = split_by_day(find_activation_files("/path/to/graphcast_acts"))
config = TrainConfig(output_dir="runs/topk_layer8", epochs=5)
result = train_topk(train_files, val_files, config)
print(result["best_checkpoint"])
```

The batch size is a memory-friendly pilot setting; reduce it if needed. SAE
training from cached vectors does not have the same memory requirement as
running the full GraphCast extraction pipeline. A CPU fallback is available.

## 3. What the trainer optimizes

The authors' class centers each raw input across its 512 coordinates, divides
it by its L2 norm (clamped below at `1e-6`), and reconstructs that normalized
vector. The trainer passes **raw input** to `SAE.forward` and uses the matching
normalized vector as the reconstruction target.

The objective is

\[
L = \frac{1}{B}\sum_i\|x_{i,\mathrm{norm}}-\hat x_i\|_2^2
  + \beta\frac{1}{B}\sum_i
    \|\operatorname{stopgrad}(x_{i,\mathrm{norm}}-\hat x_i)-\hat r_i\|_2^2.
\]

Here `hat r` is the class's auxiliary reconstruction using dead features. The
auxiliary term is zero when there are no dead features. Detaching its residual
target keeps that term from directly optimizing the main reconstruction through
the target. No L1 penalty is added: TopK already constrains the feature count.

The trainer uses Adam, projects decoder gradients perpendicular to dictionary
columns, then renormalizes each column after the update. Dead-feature counters
are updated only on training batches. Validation never updates them.

Defaults drawn from Appendix C of
[MacMillan and Ouellette](https://arxiv.org/html/2512.24440v1#A3) are learning rate
`3e-4`, auxiliary weight `1/32`, `k_aux=512`, and a dead window of 5,000,000
vector presentations. The explicit dead window overrides the class's 3,000,000
default to match that description. The paper used batch size 8,192 and billions
of vectors; this smaller batch size and repeated-epoch workflow are pilot
choices, not an exact reproduction of their training run.

For a very small pilot, five million presentations may not elapse. The notebook
explicitly uses `dead_window=100_000` to exercise dead-feature recovery sooner;
record this change and choose the final recipe before running comparisons.

## 4. Read the outputs

| File | Contents |
| --- | --- |
| `config.json` | Architecture, optimizer, and run settings; compatible keys with the original demo |
| `data_manifest.json` | Train/validation paths, shapes, sizes, modification times, SAE source hash, versions, and initial validation metrics |
| `history.csv` | One row per epoch: losses, validation error, FVU, active counts, and feature usage |
| `best.pt` | Model and optimizer state at the lowest validation reconstruction error |
| `last.pt` | Latest completed epoch, including optimizer state, counters, history, and RNG state |

`reconstruction_mse` means mean **squared L2 error per vector** (sum over
coordinates, mean over observations). Divide it by 512 for MSE per coordinate.
Train metrics average pre-update batch losses; validation metrics use the model
at the end of the epoch, so they need not have the same value.

`val_fvu` is reconstruction squared error divided by error from always
predicting the **training** mean of normalized vectors. Below one beats that
baseline. It is undefined/blank when the baseline error is zero. `mean_l0`
counts positive codes and is at most `k`; ReLU can leave fewer than `k` positive
features. `fraction_features_used` reports the fraction used anywhere in the
validation data, while `dead_feature_fraction` uses the training counters.
Better reconstruction alone does not establish better interpretability.

To load a trained model (the `model_state` key also works with the original demo):

```python
import torch
from graphcast_interpretability.model import SAE

checkpoint = torch.load("runs/topk_layer8/best.pt", map_location="cpu", weights_only=True)
cfg = checkpoint["config"]
model = SAE(cfg["d_in"], cfg["latent"], cfg["k"], cfg["k_aux"],
            dead_window=cfg["dead_window"])
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()
# recon, code, auxiliary = model(raw_float32_batch)
```

## 5. Resume after a Colab disconnect

Save the run directory on Google Drive. Reinstall the same code, remount Drive,
and provide the same data and settings. Change `epochs` to the **total** desired
number; resume from the original run directory's `last.pt`:

```python
config.epochs = 10
result = train_topk(train_files, val_files, config,
                    resume=f"{config.output_dir}/last.pt")
```

Or add `--resume runs/topk_layer8/last.pt --epochs 10` to the original CLI call.
Checkpoints are written atomically at epoch boundaries. An interrupted partial
epoch is repeated. Resume checks the data manifest and training settings;
keep file paths and files unchanged. The manifest records metadata, not full
content hashes of large data files. Fresh runs refuse to overwrite an existing
run directory. On the same CPU setup, the tests verify that a resumed run
matches an uninterrupted run exactly; GPU results can vary with hardware and
library versions.

## 6. Verify before spending compute

```bash
python -m unittest discover -s tests -v
```

Tests cover real optimization on synthetic activation vectors, the normalized
target, detached auxiliary residual, complete file coverage, date separation,
bad input rejection, decoder norms, and exact epoch-boundary restart behavior.
Synthetic tests check the training software; evaluating weather features still
requires real GraphCast activations and independent weather-event labels.
