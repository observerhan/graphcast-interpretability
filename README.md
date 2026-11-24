# AMS691 · From GraphCast activations to a tropical-cyclone feature

Run these in **Google Colab in your browser**, in this order. You can keep the defaults and choose **Runtime → Run all** in each notebook.

| Order | Notebook | What it does | GPU |
|---|---|---|---|
| 1 | [Extract activations](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/01_extract_activations_to_drive.ipynb) | Downloads weather data, runs GraphCast, saves layer-8 activations to Drive | A100 or H100 |
| 2 | [Train TopK SAE](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/02_train_topk_sae.ipynb) | Finds those activations, trains the authors' SAE class, saves weights and a loss plot | T4 or larger |
| 2b | [Train KAN-SAE](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/02b_train_kan_sae.ipynb) | Reuse the exact TopK split, calibrate splines, and train a separate resumable KAN-SAE | T4 or larger |
| 2c | [Train KAN + TopK](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/02c_train_kan_topk_sae.ipynb) | Reuse the TopK split, warm up splines, then train with at most 32 signed features per input | T4 or larger |
| 3 | [Prepare cyclone examples](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/03_prepare_cyclone_examples.ipynb) | Run the same frozen GraphCast at exact mask hours and pair each layer-8 row with its mesh label | A100 or H100 |
| 4 | [Probe SAE features](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/04_probe_topk_cyclones.ipynb) | Load the selected TopK, KAN, or KAN + TopK checkpoint, fit signed probes, choose on validation F1, and evaluate on held-out storms | T4 or larger |
| 5 | [Plot Ida and Hagibis](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/kan-topk-training/05_plot_cyclone_features.ipynb) | Read notebook 4’s selected feature and draw the ten storm snapshots from Figure 4B–C with matching ERA5 winds | A100 or H100 |

### Where the code lives

The seven notebooks are the Colab entry points. Shared GraphCast code is stored as ordinary Python files in the repository, so you can read it directly in GitHub or VS Code.

| Location | Responsibility |
|---|---|
| [`scripts/ams691_extract.py`](scripts/ams691_extract.py) | `WEATHER_STORE`, the GraphCast checkpoint, ERA5 loading, forecast windows, activation saving, and monthly extraction |
| [`scripts/setup_graphcast.py`](scripts/setup_graphcast.py) | Install the separate Colab GraphCast environment and verify the activation hook |
| [`src/graphcast_interpretability/model.py`](src/graphcast_interpretability/model.py) | The authors' unchanged TopK SAE class |
| [`src/graphcast_interpretability/kan_sae.py`](src/graphcast_interpretability/kan_sae.py) | KAN-SAE model, spline calibration, L1 loss, and decoder normalization |
| [`scripts/train_kan_sae.py`](scripts/train_kan_sae.py) | Exact baseline split, training-only calibration, KAN optimization, and completed-epoch resume |
| [`src/graphcast_interpretability/kan_topk_sae.py`](src/graphcast_interpretability/kan_topk_sae.py) and [`scripts/train_kan_topk_sae.py`](scripts/train_kan_topk_sae.py) | Signed spline + magnitude TopK model, warm-up, auxiliary loss, and resumable training |
| [`scripts/sae_checkpoint.py`](scripts/sae_checkpoint.py) | Load KAN + TopK, KAN, or legacy TopK checkpoints and preserve their feature values |
| [`02_train_topk_sae.ipynb`](02_train_topk_sae.ipynb) | Training settings, batching, loss, optimizer, checkpoints, and learning curves |
| [`scripts/collect_cyclone_activations.py`](scripts/collect_cyclone_activations.py) | Run the shared extractor at labeled cyclone times |
| [`scripts/cyclone_data.py`](scripts/cyclone_data.py) and [`data/cyclone_2019_2021/`](data/cyclone_2019_2021/) | Label loading, fixed schedules, mesh labels, and data provenance |
| [`scripts/cyclone_probes.py`](scripts/cyclone_probes.py) | Fit, select, and evaluate the per-feature logistic probes |
| [`scripts/collect_figure4_storms.py`](scripts/collect_figure4_storms.py) and [`scripts/figure4_storms.py`](scripts/figure4_storms.py) | Collect the storm examples, encode the selected feature, and draw the maps |

Notebooks **02c, 04, and 05** share a pinned revision containing KAN + TopK. The unchanged notebooks 01, 02, 02b, and 03 retain their previous code pin and executed results. Their activation files and saved models are compatible with the new workflow. Outputs in 04 and 05 are cleared on this branch for the new experiment; the executed KAN notebooks remain on `kan-sae-training`.

**To run the new experiment:** open **02c → 04 → 05** using the links above. Use a T4 or larger for training/probing and the existing plotting GPU setup for 05. The defaults are 4,096 features, `k=32`, and 20 epochs including one dense warm-up. Only sparse-training epochs can become `best.pt`. The run is saved separately in `kan_topk_runs/kan_topk_k32`, and notebooks 04 and 05 default to `SAE_TYPE='kan_topk'`. Reuse the data already saved by 01, 03, and 05. See [the KAN + TopK experiment guide](docs/kan_topk.md) for the training objective, diagnostics, and comparison limits.

See [the workflow guide](docs/workflow.md) for the import paths, local checks, and the parts we can reuse for KAN-SAE.

**To continue with KAN-SAE after running the existing baseline:** open notebook **02b**, then notebook **04**, then optionally notebook **05**. Existing activation files from 01 and 03 can be reused. In notebooks 04 and 05 select `SAE_TYPE='kan_sae'` to use `kan_baseline`, or `topk` to use `topk_baseline`; this branch defaults to `kan_topk`. A blank `SAE_RUN` chooses that model's baseline run name.

KAN training reuses notebook 02's exact data split and file checksums. The default run is 20 epochs, 4,096 features, batch size 256, and Adam at `1e-4`, with L1 increasing from `5e-5` to `1e-4` over a fixed 20-epoch ramp. Results are saved separately under `monthly_2017_2018/kan_runs/kan_baseline`. Increase `EPOCHS` to continue; keep the other settings unchanged or choose a new run name. See [the KAN training and probing guide](docs/kan_sae.md#colab-training-and-probing) for outputs, resume behavior, and signed-feature handling.

Connect the **same Google Drive account** in all notebooks. Notebooks 1 and 2 save their experiment under **My Drive → AMS691_GraphCast → monthly_2017_2018**. No terminal commands, local installation, manually downloaded weather files, or activation-path editing are needed.

The defaults extract **27 snapshots at 12:00 UTC** (about **2.3 GB** of activations). The **24 training snapshots** use the 15th of every month in 2017 and 2018. We add **3 validation snapshots** on August 29, 2017, April 29, 2018, and December 29, 2018. The count is 10% of the number of training dates, rounded up. Notebook 2 checks reconstruction across those three dates after every epoch. Notebooks 3 and 4 use separate, labeled 2019–2021 storms for cyclone F1.

Notebook 1 saves the dates in `activations_layer8/dataset.json`. Notebook 2 reads only the listed snapshots and records the split in `data_split.json` so the same files can be used for KAN-SAE. To expand later, change `DAYS_OF_MONTH` to `[5, 15]` in notebook 1 (48 training dates and 5 validation dates) and choose a new `RUN_NAME` in notebook 2.

Notebook 1 uses the full original 0.25°/37-level GraphCast checkpoint, 40,962 mesh nodes, and the authors' step-8 post-residual hook. It prepares compatible packages automatically without a kernel restart. A standard free T4 has too little memory for that extraction setup; the notebook checks the GPU before installing. Notebook 2 needs only PyTorch and reuses the unchanged author SAE source. Training is visible in Step 5.

If a session disconnects, rerun the notebook. Extraction skips validated completed files; training continues from its last completed epoch. Increase `EPOCHS` to continue training. Use a new `RUN_NAME` after changing the data or model settings.

Results are in `AMS691_GraphCast/monthly_2017_2018/topk_runs/topk_baseline`: `best.pt`, `last.pt`, `history.csv`, `training_curve.png`, and small JSON files recording the settings, data split, and results.

### Labeled cyclone experiment

The `kan-topk-training` branch includes the activation, training, cyclone-probe, and storm-plotting workflows. If the labeled activations do not exist yet, run notebook 3. Before notebook 4, train the hybrid with notebook 02c, select `SAE_TYPE='kan_sae'` for notebook 02b's run, or select `topk` for your existing baseline. Its default selects 12 snapshots from 2019 to fit probes, 8 from 2020 to choose the feature and cutoff, and 8 from 2021 to measure test F1. It saves about 2.35 GB in `AMS691_GraphCast/cyclones_2019_2021` and resumes completed timestamps. Both notebooks also support the 93-snapshot `all_guarded` schedule. Use the same setting in both.

**What was counted.** The cleaned release has 697 timestamped maps in 2019–2021. Taking the union of the two annotators, 452 maps contain at least one cyclone, totaling 769 connected footprints. Counting diagonals and wrapping longitude follows the dataset’s region definition. At the four exact GraphCast hours there are 117 maps, 77 positive maps, and 125 footprints. The mask files have no persistent storm IDs. In an independent check, 671 of the 769 footprints contained an interpolated NOAA IBTrACS main-track center, representing 206 distinct IDs.

The [timestamp inventory](data/cyclone_2019_2021/mask_inventory.csv) gives annotator-specific counts, union counts, matched IDs and positive nodes per timestamp (`-1` for hours outside the model input schedule). The [split and time list](data/cyclone_2019_2021/schedule.csv) records the subset selected before looking at SAE activations. The course sample contains all-background maps as well as positive maps, excludes December and January around split boundaries, and has disjoint active IBTrACS IDs across groups. [Dataset methods](https://doi.org/10.1038/s41597-025-05480-0) · [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

The bundled mesh labels are ready to use. To reproduce the inventory and projection, run `python scripts/prepare_cyclone_labels.py TC_labels_cleaned.zip ibtracs.since1980.list.v04r01.csv OUTPUT_DIR` with the notebook 3 GraphCast environment plus `h5netcdf` and `h5py`. [Input URLs, checksums and projection settings](data/cyclone_2019_2021/provenance.json).

Notebook 4 writes the zero-based feature ID, its inclusive cutoff and validation scores to `selected_feature.json`, then writes held-out results to `test_metrics.json`. Both are under `cyclones_2019_2021/course/probe_runs/` for the default sample.

Notebook 5 produces the Ida and Hagibis panels from Figure 4B–C using the feature saved by notebook 4. The earlier TopK course result selected feature 3288 with validation F1 0.252023 and test F1 0.380000. It collects five Ida snapshots from August 28–30, 2021 at 12-hour intervals, and five Hagibis snapshots from October 5–13, 2019 at 48-hour intervals. The ten timestamps match the figure. Each map uses the initialization-time convention from notebook 3, with ERA5 10 m winds at that same time. The source figure’s exact input-versus-target time convention is ambiguous in the paper, and the saved metadata states our convention explicitly.

Use the same `COHORT`, `SAE_TYPE`, `SAE_RUN` and `SAE_EXPERIMENT` settings as notebook 4. KAN maps preserve signs with symmetric color limits centered on zero. The ten new activation files occupy about 0.84 GB, with completed files reused after a disconnect. The plotting script uses the authors’ containing-triangle mesh-to-grid mapping and shares each row’s color scale across its five dates. PNG and PDF panels plus `figure4_metadata.json` are saved in the selected probe run’s `figure4_storms` folder. The regional windows are 100–70°W, 15–40°N for Ida and 120–165°E, 5–45°N for Hagibis.



