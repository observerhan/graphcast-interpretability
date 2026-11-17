# AMS691: two notebooks, from GraphCast to TopK SAE

Run these in **Google Colab in your browser**, in this order. You can keep the defaults and choose **Runtime → Run all** in each notebook.

| Order | Notebook | What it does | GPU |
|---|---|---|---|
| 1 | [Extract activations](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/01_extract_activations_to_drive.ipynb) | Downloads weather data, runs GraphCast, saves layer-8 activations to Drive | A100 or H100 |
| 2 | [Train TopK SAE](https://colab.research.google.com/github/observerhan/graphcast-interpretability/blob/simple-colab/02_train_topk_sae.ipynb) | Finds those activations, trains the authors' SAE class, saves weights and a loss plot | T4 or larger |

Connect the **same Google Drive account** in both notebooks. Everything is saved under **My Drive → AMS691_GraphCast**. No terminal commands, local installation, manually downloaded weather files, or activation-path editing are needed.

The default first run extracts four noon snapshots (August 27–30, 2021), about **320 MiB** of activations. Training uses the first three dates and validates on the last. This is a small initial experiment; nearby dates are not independent storms, and reconstruction quality alone does not establish interpretability.

Notebook 1 uses the full original 0.25°/37-level GraphCast checkpoint, 40,962 mesh nodes, and the authors' step-8 post-residual hook. It prepares compatible packages automatically without a kernel restart. A standard free T4 has too little memory for that extraction setup; the notebook checks the GPU before installing. Notebook 2 needs only PyTorch and reuses the unchanged author SAE source. Training is visible in Step 5.

If a session disconnects, rerun the notebook. Extraction skips validated completed files; training continues from its last completed epoch. Increase `EPOCHS` to continue training. Use a new `RUN_NAME` after changing the data or model settings.

Results are in `AMS691_GraphCast/topk_runs/topk_baseline`: `best.pt`, `last.pt`, `history.csv`, `training_curve.png`, and small JSON files recording the settings, data split, and results.

This branch, **simple-colab**, starts from the original `main` branch. The earlier `topk-trainer` branch is separate. Only these two new notebooks are needed for this workflow; the original files are retained below for reference.

Validation: notebook structure and local synthetic training/capture checks are run during development. Full pretrained GraphCast extraction on a Colab GPU and the user's Drive still require a real Colab run.

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
