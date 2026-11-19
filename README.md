# AMS691: two notebooks, from GraphCast to TopK SAE

Run these in **Google Colab in your browser**, in this order. You can keep the defaults and choose **Runtime → Run all** in each notebook.

| Order | Notebook | What it does | GPU |
|---|---|---|---|
| 1 | [Extract activations](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/01_extract_activations_to_drive.ipynb) | Downloads weather data, runs GraphCast, saves layer-8 activations to Drive | A100 or H100 |
| 2 | [Train TopK SAE](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/02_train_topk_sae.ipynb) | Finds those activations, trains the authors' SAE class, saves weights and a loss plot | T4 or larger |

Connect the **same Google Drive account** in both notebooks. This experiment is saved under **My Drive → AMS691_GraphCast → monthly_2017_2018**. No terminal commands, local installation, manually downloaded weather files, or activation-path editing are needed.

The defaults extract **27 snapshots at 12:00 UTC** (about **2.3 GB** of activations). The **24 training snapshots** use the 15th of every month in 2017 and 2018. We add **3 validation snapshots** on August 29, 2017, April 29, 2018, and December 29, 2018. The count is 10% of the number of training dates, rounded up. Notebook 2 checks reconstruction across those three dates after every epoch. The separate cyclone F1 evaluation will use labeled 2019–2021 storms.

Notebook 1 saves the dates in `activations_layer8/dataset.json`. Notebook 2 reads only the listed snapshots and records the split in `data_split.json` so the same files can be used for KAN-SAE. To expand later, change `DAYS_OF_MONTH` to `[5, 15]` in notebook 1 (48 training dates and 5 validation dates) and choose a new `RUN_NAME` in notebook 2.

Notebook 1 uses the full original 0.25°/37-level GraphCast checkpoint, 40,962 mesh nodes, and the authors' step-8 post-residual hook. It prepares compatible packages automatically without a kernel restart. A standard free T4 has too little memory for that extraction setup; the notebook checks the GPU before installing. Notebook 2 needs only PyTorch and reuses the unchanged author SAE source. Training is visible in Step 5.

If a session disconnects, rerun the notebook. Extraction skips validated completed files; training continues from its last completed epoch. Increase `EPOCHS` to continue training. Use a new `RUN_NAME` after changing the data or model settings.

Results are in `AMS691_GraphCast/monthly_2017_2018/topk_runs/topk_baseline`: `best.pt`, `last.pt`, `history.csv`, `training_curve.png`, and small JSON files recording the settings, data split, and results.

This branch, **simple-colab**, starts from the original `main` branch. The earlier `topk-trainer` branch is separate. Only these two new notebooks are needed for this workflow; the original files are retained below for reference.

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
