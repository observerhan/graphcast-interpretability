# AMS691 · From GraphCast activations to a tropical-cyclone feature

Run these in **Google Colab in your browser**, in this order. You can keep the defaults and choose **Runtime → Run all** in each notebook.

| Order | Notebook | What it does | GPU |
|---|---|---|---|
| 1 | [Extract activations](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/01_extract_activations_to_drive.ipynb) | Downloads weather data, runs GraphCast, saves layer-8 activations to Drive | A100 or H100 |
| 2 | [Train TopK SAE](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/02_train_topk_sae.ipynb) | Finds those activations, trains the authors' SAE class, saves weights and a loss plot | T4 or larger |
| 3 | [Prepare cyclone examples](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/cyclone-probes/03_prepare_cyclone_examples.ipynb) | Run the same frozen GraphCast at exact mask hours and pair each layer-8 row with its mesh label | A100 or H100 |
| 4 | [Probe TopK features](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/cyclone-probes/04_probe_topk_cyclones.ipynb) | Load notebook 2’s `best.pt`, fit one probe per feature, select on validation F1 and evaluate on held-out storms | T4 or larger |

Connect the **same Google Drive account** in all four notebooks. Notebooks 1 and 2 save their experiment under **My Drive → AMS691_GraphCast → monthly_2017_2018**. No terminal commands, local installation, manually downloaded weather files, or activation-path editing are needed.

The defaults extract **27 snapshots at 12:00 UTC** (about **2.3 GB** of activations). The **24 training snapshots** use the 15th of every month in 2017 and 2018. We add **3 validation snapshots** on August 29, 2017, April 29, 2018, and December 29, 2018. The count is 10% of the number of training dates, rounded up. Notebook 2 checks reconstruction across those three dates after every epoch. Notebooks 3 and 4 use separate, labeled 2019–2021 storms for cyclone F1.

Notebook 1 saves the dates in `activations_layer8/dataset.json`. Notebook 2 reads only the listed snapshots and records the split in `data_split.json` so the same files can be used for KAN-SAE. To expand later, change `DAYS_OF_MONTH` to `[5, 15]` in notebook 1 (48 training dates and 5 validation dates) and choose a new `RUN_NAME` in notebook 2.

Notebook 1 uses the full original 0.25°/37-level GraphCast checkpoint, 40,962 mesh nodes, and the authors' step-8 post-residual hook. It prepares compatible packages automatically without a kernel restart. A standard free T4 has too little memory for that extraction setup; the notebook checks the GPU before installing. Notebook 2 needs only PyTorch and reuses the unchanged author SAE source. Training is visible in Step 5.

If a session disconnects, rerun the notebook. Extraction skips validated completed files; training continues from its last completed epoch. Increase `EPOCHS` to continue training. Use a new `RUN_NAME` after changing the data or model settings.

Results are in `AMS691_GraphCast/monthly_2017_2018/topk_runs/topk_baseline`: `best.pt`, `last.pt`, `history.csv`, `training_curve.png`, and small JSON files recording the settings, data split, and results.

### Labeled cyclone experiment

The `cyclone-probes` branch extends `simple-colab`. If you have already run notebooks 1 and 2, open notebook 3. Its default selects 12 snapshots from 2019 to fit probes, 8 from 2020 to choose the feature and cutoff, and 8 from 2021 to measure test F1. It saves about 2.35 GB in `AMS691_GraphCast/cyclones_2019_2021` and resumes completed timestamps. Both notebooks also support the 93-snapshot `all_guarded` schedule. Use the same setting in both.

**What was counted.** The cleaned release has 697 timestamped maps in 2019–2021. Taking the union of the two annotators, 452 maps contain at least one cyclone, totaling 769 connected footprints. Counting diagonals and wrapping longitude follows the dataset’s region definition. At the four exact GraphCast hours there are 117 maps, 77 positive maps, and 125 footprints. The mask files have no persistent storm IDs. In an independent check, 671 of the 769 footprints contained an interpolated NOAA IBTrACS main-track center, representing 206 distinct IDs.

The [timestamp inventory](data/cyclone_2019_2021/mask_inventory.csv) gives annotator-specific counts, union counts, matched IDs and positive nodes per timestamp (`-1` for hours outside the model input schedule). The [split and time list](data/cyclone_2019_2021/schedule.csv) records the subset selected before looking at SAE activations. The course sample contains all-background maps as well as positive maps, excludes December and January around split boundaries, and has disjoint active IBTrACS IDs across groups. [Dataset methods](https://doi.org/10.1038/s41597-025-05480-0) · [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

The bundled mesh labels are ready to use. To reproduce the inventory and projection, run `python scripts/prepare_cyclone_labels.py TC_labels_cleaned.zip ibtracs.since1980.list.v04r01.csv OUTPUT_DIR` with the notebook 3 GraphCast environment plus `h5netcdf` and `h5py`. [Input URLs, checksums and projection settings](data/cyclone_2019_2021/provenance.json).

Notebook 4 writes the zero-based feature ID, its inclusive cutoff and validation scores to `selected_feature.json`, then writes held-out results to `test_metrics.json`. Both are under `cyclones_2019_2021/course/probe_runs/` for the default sample.

The original four-snapshot Ida run is available in the branch history. Its activation and model folders in Drive are separate from the monthly experiment.

---

## Original project information

# GraphCast Interpretability

This repository contains code and utilities accompanying the paper:

> **MacMillan, T. & Ouellette, N. T. (2025)**  
> [Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features](https://arxiv.org/abs/2512.24440)
> *(submitted / under review)*

The focus of this codebase is **mechanistic interpretability of DeepMind’s GraphCast weather model**, using sparse autoencoders (SAEs) to identify physically meaningful features in intermediate GraphCast representations.

Some fun features to look at include the hurricane tracking feature

![Feature 3243 aligned with a tropical cyclone](figures/hurricane_single.png)

Or the Pineapple express

![Feature 1820 aligned with an atmospheric river](figures/ar_single.png)


The primary runnable artifact at present is a **Google Colab notebook** that reproduces a key hurricane case study from the paper. Other components are utilities and scripts used during large-scale analysis on HPC systems. We are working to publish these as well.

---
## Quick start

Click the Colab link below to reproduce a hurricane example from the paper:

[Demo notebook](https://colab.research.google.com/github/theodoremacmillan/graphcast-interpretability/blob/main/hurricane_feature_demo.ipynb) in Colab 

Due to Colab updating their Jax package version, you'll have to switch the runtime version to 2025.07 instead of 'Latest'. It is recommended to run the Colab on an A100 or H100 as the activation steps consume considerable GPU memory.
