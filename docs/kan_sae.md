# KAN-SAE model for GraphCast

The model is implemented in [`src/graphcast_interpretability/kan_sae.py`](../src/graphcast_interpretability/kan_sae.py). It follows the encoder and decoder equations in Minjong Cheon's [KAN-SAE paper, arXiv:2605.17493v1](https://arxiv.org/abs/2605.17493v1), Section 4.1 and Algorithm 1. The uploaded paper applies this model to SONNY; this repository adapts it to our saved GraphCast activations.

This branch adds the model, loss calculation, calibration, and tests. Notebooks 01-05 still run the existing TopK workflow and keep their previous code pins. There is not yet a KAN training notebook or KAN checkpoint loader for the cyclone probes.

## What the model calculates

For a row of GraphCast activations, first form the reconstruction target `u`. By default, `u` uses exactly the TopK baseline's preprocessing: subtract the row mean, then divide by its L2 norm with a minimum denominator of `1e-6`.

The KAN-SAE then calculates:

```text
h = W_enc (u - b_pre) + b_enc
z_j = sum_k c[j, k] * B[k, 3](h_j; knots[j])
reconstruction = W_dec z + b_pre
```

`h` is the linear projection; `z` contains the learned features. Every feature has its own spline coefficients and calibrated knot vector. This is a spline activation after a linear encoder, rather than a full KAN with a spline on every input-output connection.

| Quantity | Implementation |
|---|---|
| Input width | Configurable; use `d_in=512` for our GraphCast activations |
| Number of features | Configurable; `latent=4096` matches our TopK dictionary size |
| Spline degree | `degree=3`: cubic, as specified explicitly in Section 5.1 |
| Coefficients | `n_basis=9` trainable values for each feature |
| Encoder parameters | `enc.weight` and `b_enc` |
| Decoder parameters | `dec.weight` and the shared `b_pre` |
| Spline parameters | `spline.coefficients`, shape `(latent, 9)` |
| Fixed spline state | `spline.knots`, shape `(latent, 13)`, stored in checkpoints |
| Model output | `(reconstruction, code)`, with shapes `(batch, d_in)` and `(batch, latent)` |

There is no ReLU, TopK selection, or AuxK loss in this model. Codes may be positive or negative. Sparsity is encouraged by the L1 loss; it does not enforce a fixed count of nonzero features.

The spline is evaluated with the local de Boor recurrence. Only four coefficients per input/feature pair are gathered for a cubic spline, instead of building every basis value or an input-feature connection tensor.

## Initialization and calibration

The paper specifies zero spline coefficients, a training-data mean for `b_pre`, unit-normalized Gaussian decoder columns, and knot ranges based on the 1st and 99th percentiles of a 50,000-token calibration sample.

1. Construct `KANSAE` and move it to the training device.
2. Select a representative calibration sample **only from the SAE training dates**. The caller chooses the sample and its random seed; `initialize_from_data()` does not sample silently.
3. Supply the mean of reconstruction targets over the full training set as `training_mean`. If it is omitted, the model uses the calibration sample's mean as an estimate.
4. Call `initialize_from_data()` before any optimizer steps. It projects the sample through the initial encoder and computes the percentile range separately for each feature. Quantiles are computed in feature chunks to limit memory use.

The coefficients start at zero, so the initial code is zero and the reconstruction is `b_pre`. On the first backward pass, reconstruction gradients update the spline coefficients; gradients into the encoder and decoder weights are initially zero. Once coefficients change, those weights receive gradients too. This behavior follows from the zero initialization and is checked in the tests.

Initialization can only be called once on a fresh model. Loading a checkpoint restores both the knots and the initialization flag; do not recalibrate a resumed model, because moving knots would change its learned functions.

## Explicit choices beyond the paper

The paper does not completely specify spline boundary handling or encoder initialization. Our choices are recorded here so this implementation is not mistaken for recovered author code.

| Detail | Choice in this repository |
|---|---|
| GraphCast input scaling | Match the existing TopK per-row normalization by default. `normalize_inputs=False` instead reconstructs raw inputs. The paper's equations use raw activation vectors. |
| Dictionary size | The class takes `latent` explicitly. The paper used 1,024 features for 384 input dimensions; the GraphCast example uses 4,096 for 512 to match our existing baseline. |
| Range sharing | Calibrate a separate 1st-99th percentile range per feature. The paper does not clearly resolve a shared versus per-feature range. |
| Endpoint knots | Use a clamped uniform grid: repeat each endpoint four times for cubic splines. Nine coefficients then give six intervals and thirteen knots, with five interior knots. |
| Outside the range | Clamp the pre-activation to the nearest endpoint, so the function extends constantly. This is a boundary choice, not an extra ReLU or a constraint on the coefficient signs. Gradients into pre-activations are zero outside the range; endpoint coefficients can still learn. |
| Knot updates | Keep knots fixed after calibration and save them as buffers. No adaptive grid updates or trainable knot positions are added. |
| Constant or very narrow ranges | Widen the interval to at least `min_grid_width=1e-3`, with an additional floating-point resolution safeguard. |
| Encoder initialization | Start from the transpose of the normalized decoder, as in our TopK baseline. The two weight tensors are independently trainable. |
| Encoder bias | Initialize `b_enc` to zero. |

The defaults use float32. Float64 is also covered by the tests; mixed-precision training has not been validated. Calibration requires float32 or float64.

## Model and loss API

Install PyTorch in the environment that runs the SAE. The optional dependency group is available through `python -m pip install -e '.[sae]'`; Colab normally already supplies PyTorch. Keep this separate from the dedicated GraphCast/JAX environment, as in the current notebooks.

In this example, `calibration_rows`, `training_mean`, and `batch` come from the training split. `training_mean` is in the normalized reconstruction-target space, not the raw activation space. Activations passed to the model are still the raw saved rows.

```python
import torch
from graphcast_interpretability.kan_sae import KANSAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KANSAE(d_in=512, latent=4096).to(device)
model.initialize_from_data(calibration_rows, training_mean=training_mean)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch = batch.to(device)
optimizer.zero_grad(set_to_none=True)
reconstruction, code = model(batch)
losses = model.loss(batch, reconstruction, code, l1_weight=5e-5)
losses['loss'].backward()
optimizer.step()
model.renormalize_decoder_()
```

`model.loss()` returns differentiable tensors named `loss`, `reconstruction_loss`, and `l1_loss`. It sums squared reconstruction errors over the input dimensions, sums absolute codes over the features, and then averages both quantities over rows. The total is `reconstruction_loss + l1_weight * l1_loss`. Averaging over all elements separately would change the relative scale of the two terms.

Call `model.renormalize_decoder_()` after every optimizer step, as in the paper. It does not run automatically inside `forward()`. To encode without reconstructing, use `model.encode(batch)`; to reconstruct a chosen code, use `model.decode(code)`.

The paper linearly increases lambda from `5e-5` to `1e-4` across 100 epochs using Adam with learning rate `1e-4`. The epoch schedule belongs in the upcoming trainer. These values are starting references, not tuned settings for our normalized GraphCast data and larger dictionary.

## Checkpoints and feature statistics

Save model identity, constructor settings, and state together:

```python
torch.save({
    'model_type': 'kan_sae',
    'config': model.config(),
    'model_state': model.state_dict(),
}, 'kan_sae.pt')

saved = torch.load('kan_sae.pt', map_location='cpu', weights_only=True)
assert saved['model_type'] == 'kan_sae'
restored = KANSAE(**saved['config'])
restored.load_state_dict(saved['model_state'], strict=True)
restored.eval()
```

This saves the model. A resumable training checkpoint must additionally save optimizer state, epoch, training and validation split identities, calibration-sample provenance, and training history, as the TopK trainer already does. If using float64, construct the restored model with `.double()` before loading.

`model.coefficient_alive_mask(threshold)` implements the paper's `max(abs(c_j)) >= threshold` criterion. The threshold is mandatory: the paper's fitted value `1.4021` should not be transferred automatically to our different activation scale. For activity on data, use an explicitly chosen magnitude tolerance on `abs(code)` rather than `code > 0`.

## Tests and remaining integration

```bash
python -m pip install -e '.[sae,test]'
python -m unittest discover -s tests -v
```

The model tests compare spline values and derivatives with SciPy's independent `BSpline` implementation, including knots, endpoints, constant extension, and multiple degrees. They also check coefficient gradients, per-feature calibration, normalization, signed L1 loss, learning from zero initialization, a small synthetic optimization, exact checkpoint round trips, and forward/backward execution at the intended 512-by-4,096 GraphCast dimensions. These are CPU tests; they do not establish weather-feature quality or benchmark GPU training.

The next integration step is a separate Colab trainer using the existing saved activations and split. The TopK model and training notebook remain the baseline. Before running KAN-SAE through notebooks 04-05, their checkpoint loading and feature handling must also be updated:

- `cyclone_probes.open_experiment()` currently instantiates only the TopK class.
- TopK `forward()` returns three tensors; KAN-SAE returns two and has no auxiliary reconstruction.
- `fit_probes()` rejects negative values, and `best_cutoff()` assumes implicit zero belongs at the beginning of the sorted nonnegative values. Both need signed-feature support.
- KAN plots need to retain the signs of feature values and use an appropriate color scale.

Do not pass a KAN checkpoint into the current TopK notebooks or take absolute values merely to satisfy their checks. The data, splits, labels, and geographic mapping can still be reused once those integrations are made.
