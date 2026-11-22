# Working with the AMS691 repository

## Running the notebooks

Open the workflow notebooks from the Colab links in the README. The KAN + TopK route is 02c, 04, then 05, reusing previously collected data. These three entry points share a new immutable `RELEASE` commit in `/content/ams691_project_<commit>`. The unchanged baseline notebooks 01, 02, 02b and 03 retain their previous pin and recorded outputs. The checkout contains `scripts/`, `src/`, and the bundled cyclone labels.

The notebook adds `scripts/` and `src/` to Python's import path. The collectors also run by their full script paths, so Python can find the adjacent `ams691_extract.py` automatically. No notebook generates this module or reads another notebook's source code.

Notebooks 01, 03, and 05 run `scripts/setup_graphcast.py` with Colab's Python. That script prepares `/content/ams691_graphcast_env` and checks the GraphCast activation hook inside that environment. The collectors then run with that environment's Python. The TopK training and probing steps use Colab's PyTorch environment.

## Following an ERA5 read

1. Find the cloud address in `WEATHER_STORE` in `scripts/ams691_extract.py`.
2. `open_weather(task)` in the same file opens that Zarr dataset and checks its quarter-degree grid.
3. `forecast_window(source, center, pressure_levels)` loads the three times needed for a forecast: initialization minus six hours, initialization, and initialization plus six hours.
4. For the storm maps, `scripts/collect_figure4_storms.py` separately selects the two 10 m wind components at initialization and saves the regional arrays.
5. `scripts/figure4_storms.py` computes the wind magnitude and draws the maps.

The GraphCast commit, checkpoint name, layer, data source, time convention, and saved file formats are the same as in the executed baseline. The setup script keeps the previous package requirements and memory-saving hook patch.

## Reusing existing results

Keep the same Drive account, experiment folder, `COHORT`, and `SAE_RUN` when opening the reorganized notebooks. Existing activation files and checkpoints are still checked against their manifests. No data migration or retraining is needed for this reorganization. Resume guards still reject changed datasets, corrupt activations, or mismatched checkpoint settings.

Saved outputs and execution counts in 04 and 05 have been cleared on the hybrid branch. The new 02c is also unexecuted. Rerun in Colab to record the hybrid experiment. Clearing notebook output does not remove any activation files, checkpoints, or plots from Drive; completed work there will still be reused by the resume checks.

## KAN-SAE training and probing

The KAN-SAE model is implemented separately in `src/graphcast_interpretability/kan_sae.py`; the TopK baseline remains in `src/graphcast_interpretability/model.py`. See [the KAN-SAE guide](kan_sae.md) for the equations, calibration, checkpoint format, and tests. The saved GraphCast activation arrays, training and validation dates, cyclone masks, and storm snapshots can be reused.

Notebook 02 preserves the TopK training calculation. Notebook 02b calls `scripts/train_kan_sae.py`, reads the TopK run's saved split, verifies every activation checksum, and trains KAN-SAE in a separate run folder. Calibration uses only training rows. `last.pt` is the authoritative completed-epoch state, including optimizer and knots; interrupted derived outputs are repaired from it on resume.

Notebooks 04 and 05 use `SAE_TYPE` and `SAE_RUN` to select a checkpoint. `scripts/sae_checkpoint.py` supports legacy TopK, L1 KAN, and KAN + TopK checkpoints. Probe fitting scales by the training maximum absolute feature value and retains signs. Cutoff selection orders negative values, zero, and positive values correctly. Dense L1 KAN fit/validation codes are stored on local disk; signed KAN + TopK uses sparse RAM storage; test evaluation retains only the selected feature. KAN storm plots use a diverging scale centered on zero.

Use the same activation split, cyclone validation/test groups, and evaluation procedure for both models. The KAN defaults are 20 epochs with a 20-epoch L1 ramp; extending `EPOCHS` preserves that ramp. These are course-project settings, not a reproduction of the paper's 100-epoch SONNY experiment. The integration has CPU synthetic tests; real GraphCast KAN training must be run in Colab.

## KAN + TopK extension

Notebook 02c calls `scripts/train_kan_topk_sae.py`. It reuses the original KAN spline implementation, applies signed magnitude TopK after the splines, and saves to a separate `kan_topk_runs` folder. Warm-up and best-checkpoint eligibility are explicit, and the original KAN model/trainer files remain unchanged to preserve resume compatibility. See [the experiment guide](kan_topk.md).

## Local checks and development

The extraction checks require NumPy and xarray, but no GPU, GraphCast weights, or cloud downloads.

```bash
python -m pip install numpy==2.2.6 xarray==2025.1.2
python -m unittest discover -s tests -p test_extraction.py -v
```

The extraction checks use synthetic data for midnight-crossing time windows, latitude ordering, invalid-grid rejection, atomic activation saving, and resuming completed runs. Full GraphCast execution still requires the Colab A100/H100 environment. To run the KAN-SAE tests too, install the `sae,test` extras and use `python -m unittest discover -s tests -v`, as explained in the KAN-SAE guide.

For local script work, add `scripts/` and `src/` to `PYTHONPATH`, or run a collector by its script path. The dependency-light configuration can be inspected without installing GraphCast:

```bash
PYTHONPATH=scripts python -c "from ams691_extract import WEATHER_STORE; print(WEATHER_STORE)"
```

When changing shared code, commit those changes first. Then update `RELEASE` in the affected entry points (02c, 04 and 05 for this extension) to that code commit and commit the notebook changes. Preserve the original baseline pins when their implementation has not changed. This makes the Colab entry points use a reproducible code version. A notebook does not automatically use edits from a different Git branch merely because it was opened from that branch.
