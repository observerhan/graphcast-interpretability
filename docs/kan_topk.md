# KAN + TopK experiment

Run [02c in Colab](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/02c_train_kan_topk_sae.ipynb), then [04](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/04_probe_topk_cyclones.ipynb), then optionally [05](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/05_plot_cyclone_features.ipynb). Use the same Drive account. Existing GraphCast activations, labels, TopK split, and storm caches are reused.

This is an experimental encoder-spline variant of the existing KAN-SAE. It is not a claim of improved cyclone detection or a reproduction of the original KAN-SAE architecture. The previous L1 KAN model and trainer are unchanged, and their saved runs remain usable.

## Model

For the same per-row centered, L2-normalized target `u` as both existing SAEs:

```text
h = W_enc (u - b_pre) + b_enc
s_j = phi_j(h_j)                         # learned cubic spline
I = indices of the k largest abs(s_j)   # independently for each input row
z_j = s_j if j in I else 0              # preserve the selected signs
reconstruction = W_dec z + b_pre
```

The dictionary has 4,096 features, while each input has at most 32 nonzero coefficients. Different inputs can use different subsets. TopK is applied after the splines. The decoder is linear, with unit-length columns. The original spline calibration, fixed grids, and constant extension outside the grid are retained.

The public `encode()` and `forward()` always apply the fixed final k. Probes and plots read these post-gate values, including negative values and zeros. They never substitute the dense pre-gate values.

The model is in [`kan_topk_sae.py`](../src/graphcast_interpretability/kan_topk_sae.py); training is in [`train_kan_topk_sae.py`](../scripts/train_kan_topk_sae.py).

## Training defaults

| Setting | Value |
|---|---|
| Run name | `kan_topk_k32` |
| Output folder | `monthly_2017_2018/kan_topk_runs/kan_topk_k32` |
| Dictionary / active budget | 4,096 / 32 |
| Total training | 20 epochs: 1 dense warm-up + 19 sparse epochs |
| Optimizer | Adam, learning rate `1e-4`, batch size 256, seed 0 |
| Calibration | Up to 50,000 training rows, with saved indices |
| Main objective | Mean per-vector squared reconstruction error |
| L1 penalty | None; the gate directly enforces the activation budget |
| Dead-feature window | 3,000,000 training rows without a selected nonzero activation |
| Auxiliary objective | Reconstruct the detached residual with up to 512 dead features; weight `1/32` |
| Best checkpoint | Lowest gated reconstruction error among sparse-training epochs |

All spline coefficients initially equal zero, as in the existing KAN model. A hard TopK gate at that point would select arbitrary ties and restrict which features receive gradients. The first epoch explicitly trains the dense model so more than k features can begin learning. Subsequent epochs apply the gate in the reconstruction objective.

Validation always applies the final k, even during warm-up. Epoch zero and dense warm-up are never eligible for `best.pt`. During warm-up there is a resumable `last.pt`, but no deployable `best.pt`; notebook 04 requires at least one completed sparse epoch. The loader also rejects a warm-up training checkpoint for probing.

During sparse training, inactivity counts track whether a feature is selected anywhere in a batch. The auxiliary loss selects dead features by absolute magnitude, preserving their signs, and fits the detached reconstruction residual without adding the decoder bias. This auxiliary code does not alter the main 32-feature code. Decoder gradients are projected orthogonal to the unit columns before the optimizer step, and the columns are renormalized afterwards.

## Resume and outputs

Increase only `EPOCHS` to extend a run. Changing k, warm-up length, learning rate, model/trainer source, calibration, data, or other training settings requires a new `RUN_NAME`. For example, try `K_ACTIVE=64` with `RUN_NAME='kan_topk_k64'` as a separate experiment.

`last.pt` contains the current model, optimizer, spline grids, inactivity counts, calibration indices, history, RNG state, and best sparse model. It is written atomically after each complete epoch; resuming repairs derived artifacts from it and repeats only an interrupted partial epoch. CPU tests verify exact uninterrupted/resumed equivalence; identical results across different GPUs or library versions are not guaranteed.

Other outputs are `best.pt`, `history.csv`, `config.json`, `data_split.json`, `calibration.json`, `calibration_indices.npy`, and, after the final cell, `summary.json` and `training_curve.png`.

History records the phase (`dense_warmup` or `sparse`) explicitly. The reconstruction plot excludes dense training error from the sparse training curve. Validation reports:

- `mean_active_features` and `max_active_features`: actual nonzero counts, bounded by k.
- `mean_active_above_threshold`: count above the diagnostic magnitude tolerance, default `1e-6`.
- `validation_l1`: sum of absolute coefficient magnitudes per input, not a feature count.
- `features_used_on_validation`: dictionary entries used anywhere on validation data.
- `validation_clamped_fraction` and `selected_clamped_fraction`: spline boundary saturation across all and selected features respectively.

## Probes and comparison

On this branch notebooks 04 and 05 default to:

```python
SAE_TYPE = 'kan_topk'
SAE_RUN = ''  # resolves to kan_topk_k32
```

The alternatives are `kan_sae` / `kan_baseline` and `topk` / `topk_baseline`. A custom run name must match notebook 02c. All models use the same signed logistic-probe objective and validation cutoff search; the original TopK identity format remains compatible with saved selections. KAN + TopK stores its signed codes as a sparse matrix in RAM, avoiding the dense L1 KAN feature cache.

The SAE uses only the TopK training and reconstruction-validation split from 2017–2018. Cyclone probes fit on 2019, select the feature and cutoff on 2020, and evaluate the locked choice on 2021. Use validation F1 to compare hyperparameters and reserve test results for final reporting. Repeated model selection after inspecting test scores weakens that test as an unbiased estimate; an additional untouched group is preferable for a later final study.

Compare reconstruction, dictionary use, and single-feature F1 together. The historical TopK test F1 is 0.380 and L1 KAN test F1 is approximately 0.108 for the course run. No KAN + TopK weather result is claimed before a Colab run. Lower reconstruction error alone does not establish a more interpretable cyclone feature.

The experiment holds data, normalization, dictionary size, and probing fixed. Signed outputs, spline nonlinearity, warm-up, optimization details, and training budget still differ from the original TopK baseline. A later matched-budget, multi-seed comparison would be needed to attribute an improvement specifically to splines. The original and hybrid KAN training defaults both use 20 epochs; the saved TopK result used its best checkpoint at epoch 8 from a 10-epoch run.

## Validation

Run `python -m unittest discover -s tests -v` in an environment with the test dependencies. Tests cover signed magnitude selection and gradients, tied/zero values, the 512-by-4,096 model dimensions, auxiliary gradients, sparse feature storage, checkpoint identity, dense warm-up exclusion, exact resume across the phase transition, and compatibility with existing TopK and KAN runs. These are synthetic CPU checks, not a replacement for GraphCast evaluation on Colab.
