"""Fit one balanced logistic probe per frozen SAE feature and select on held-out storms."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from tqdm.auto import tqdm

from cyclone_data import activation_path, read_labels, read_samples, sha256, write_json

RIDGE = 1e-4  # on the slope after division by the feature's training maximum


def open_experiment(root, data_dir, cohort, checkpoint_path):
    import torch
    from graphcast_interpretability.model import SAE

    manifest_path = root / cohort / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run notebook 03 with cohort={cohort!r} on this Drive first.")
    manifest = json.loads(manifest_path.read_text())
    rows = read_samples(data_dir, cohort)
    if (manifest["samples"] != rows or manifest["time_convention"] != "last_input_utc"
            or set(manifest["completed"]) != {r["center"] for r in rows}):
        raise ValueError("The chosen labeled run is incomplete or differs from the published schedule.")
    label_path = root / "mesh_labels.npz"
    if sha256(label_path) != manifest["label_sha256"] or sha256(data_dir / label_path.name) != manifest["label_sha256"]:
        raise ValueError("Mesh label bundle does not match notebook 03.")
    labels = read_labels(label_path, rows)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Notebook 02's best.pt was expected at {checkpoint_path}. Set SAE_RUN to your run name.")
    training_split = json.loads((checkpoint_path.parent / "data_split.json").read_text())
    if manifest["source"] != training_split["source"]:
        raise ValueError("GraphCast layer, hook, model or source differs from notebook 02.")
    old_days = set(training_split["training_dates"] + training_split["validation_dates"])
    if old_days & {r["center"][:10] for r in rows}:
        raise ValueError("A labeled day was also used to fit or select this SAE.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (saved["data_signature"] != training_split["data_signature"] or
            set(saved["validation_dates"]) != set(training_split["validation_dates"])):
        raise ValueError("The checkpoint and data_split.json are from different training runs.")
    cfg = saved["config"]
    if cfg["d_in"] != manifest["source"]["shape"][1]:
        raise ValueError("SAE input width does not match the GraphCast embedding.")
    model = SAE(d_in=cfg["d_in"], latent=cfg["latent"], k_active=cfg["k"],
                k_aux=cfg["k_aux"], dead_window=cfg["dead_window"])
    model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.requires_grad_(False).eval().to(device)
    identity = dict(checkpoint_sha256=sha256(checkpoint_path),
                    checkpoint_epoch=int(saved["epoch"]), latent=cfg["latent"], k=cfg["k"],
                    cohort=cohort, dataset_manifest_sha256=sha256(manifest_path), ridge=RIDGE)
    return model, device, manifest, labels, identity


def encode_split(role, root, manifest, labels, model, device, batch_size=1024):
    """Keep only nonzero post-TopK coefficients. Row order is time, then mesh node."""
    import torch

    matrices, ys, centers = [], [], []
    for sample in tqdm([r for r in manifest["samples"] if r["split"] == role], desc=f"Encoding {role}"):
        center = sample["center"]
        path = activation_path(root, center)
        if sha256(path) != manifest["completed"][center]:
            raise ValueError(f"Activation checksum differs at {center}. Rerun notebook 03.")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.shape != tuple(manifest["source"]["shape"]) or arr.dtype != np.float32:
            raise ValueError(f"Unexpected activation shape or type at {center}")
        blocks = []
        with torch.inference_mode():
            for start in range(0, len(arr), batch_size):
                x = torch.from_numpy(np.array(arr[start:start + batch_size], copy=True)).to(device)
                _, code, _ = model(x)  # includes notebook 02's per-node centering, L2 norm and TopK
                if not torch.isfinite(code).all():
                    raise ValueError(f"Non-finite feature in {center}")
                i, j = torch.where(code != 0)
                blocks.append(sparse.csr_matrix((code[i, j].cpu().numpy(),
                                                 (i.cpu().numpy(), j.cpu().numpy())),
                                                shape=tuple(code.shape)))
        matrices.append(sparse.vstack(blocks, format="csr"))
        ys.append(labels[center])
        centers.append(center)
    X = sparse.vstack(matrices, format="csc")
    y = np.concatenate(ys)
    if not np.any(y) or np.all(y):
        raise ValueError(f"{role} needs both cyclone and background nodes.")
    return X, y, centers


def column(X, j):
    lo, hi = X.indptr[j:j + 2]
    return X.indices[lo:hi], X.data[lo:hi]


def fit_probes(X, y):
    """Each fit node contributes. Implicit zero inputs are aggregated by class."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    weights = np.zeros((X.shape[1], 2), dtype=np.float64)  # slope per raw unit, intercept
    for j in tqdm(range(X.shape[1]), desc="Fitting single-feature probes"):
        rows, value = column(X, j)
        if not len(value) or (len(value) == len(y) and value.min() == value.max()):
            continue
        scale = float(value.max())
        if scale <= 0 or np.any(value < 0):
            raise ValueError("Expected nonnegative post-ReLU TopK codes.")
        pos = y[rows]
        p = int(pos.sum())
        z = np.r_[value.astype(np.float64) / scale, 0., 0.]
        target = np.r_[pos, True, False].astype(np.float64)
        w = np.r_[np.where(pos, .5 / n1, .5 / n0),
                  .5 * (n1 - p) / n1, .5 * (n0 - len(pos) + p) / n0]

        def objective(params):
            a, b = params
            score = a * z + b
            loss = (w * np.logaddexp(0., (1. - 2. * target) * score)).sum() + .5 * RIDGE * a * a
            residual = w * (expit(score) - target)
            return loss, np.array([(residual * z).sum() + RIDGE * a, residual.sum()])

        opt = minimize(objective, np.zeros(2), method="L-BFGS-B", jac=True,
                       options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 150})
        if not opt.success or not np.all(np.isfinite(opt.x)):
            raise RuntimeError(f"Probe {j} did not converge: {opt.message}")
        weights[j] = [opt.x[0] / scale, opt.x[1]]
    return weights


def best_cutoff(value, pos, n, n1, slope):
    """Sweep tied activation values together, including all implicit zeros."""
    order = np.argsort(value, kind="stable")
    levels, starts, sizes = np.unique(value[order], return_index=True, return_counts=True)
    hits = np.add.reduceat(pos[order].astype(np.int64), starts) if len(value) else np.empty(0, dtype=np.int64)
    if n > len(value):
        levels = np.r_[np.float32(0), levels]
        sizes = np.r_[n - len(value), sizes]
        hits = np.r_[n1 - int(pos.sum()), hits]
    if slope > 0:
        levels, sizes, hits = levels[::-1], sizes[::-1], hits[::-1]
    tp = np.cumsum(hits)
    predicted = np.cumsum(sizes)
    f1 = 2. * tp / (n1 + predicted)
    k = int(np.argmax(f1))  # for tied F1 prefer fewer flagged nodes
    return float(levels[k]), int(tp[k]), int(predicted[k])


def metrics(tp, pred, n1, total):
    fp, fn = pred - tp, n1 - tp
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(total - pred - fn),
                precision=float(tp / pred) if pred else 0., recall=float(tp / n1) if n1 else 0.,
                f1=float(2 * tp / (n1 + pred)) if n1 + pred else 0.)


def select_feature(X, y, weights):
    """Evaluate every feature with a fitted direction on the validation nodes."""
    n1 = int(y.sum())
    ranking = []
    for j, (slope, intercept) in enumerate(tqdm(weights, desc="Choosing feature and cutoff")):
        if slope == 0.:  # no training signal (balanced intercept-only probability is 0.5)
            continue
        ix, v = column(X, j)
        cutoff, tp, pred = best_cutoff(v, y[ix], len(y), n1, slope)
        ranking.append(dict(feature=int(j), slope=float(slope), intercept=float(intercept),
                            rule=">=" if slope > 0 else "<=", activation_cutoff=cutoff,
                            probability_cutoff=float(expit(slope * cutoff + intercept)),
                            **metrics(tp, pred, n1, len(y))))
    if not ranking:
        raise ValueError("No SAE feature had a nonconstant fitted probe.")
    ranking.sort(key=lambda r: (-r["f1"], r["fp"], r["feature"]))
    return ranking


def evaluate_selected(X, y, centers, winner):
    alpha = X.getcol(winner["feature"]).toarray().ravel()
    cutoff = winner["activation_cutoff"]
    pred = alpha >= cutoff if winner["rule"] == ">=" else alpha <= cutoff
    if len(y) % len(centers):
        raise ValueError("Test rows cannot be divided into complete global snapshots.")
    by_time = []
    for t, truth, guessed in zip(centers, np.split(y, len(centers)), np.split(pred, len(centers))):
        by_time.append(dict(center=t, **metrics(int((truth & guessed).sum()), int(guessed.sum()),
                                                int(truth.sum()), len(truth))))
    overall = metrics(int((y & pred).sum()), int(pred.sum()), int(y.sum()), len(y))
    return overall, by_time


def write_table(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_selection(out, identity, weights, ranking):
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "fitted_probes.csv", [dict(feature=j, slope=a, intercept=b)
                                            for j, (a, b) in enumerate(weights)])
    write_table(out / "validation_ranking.csv", ranking)
    write_json(out / "selected_feature.json", dict(experiment=identity, validation=ranking[0]))
