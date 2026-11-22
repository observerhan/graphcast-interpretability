"""Fit one balanced logistic probe per frozen SAE feature and select on held-out storms."""
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from tqdm.auto import tqdm

from cyclone_data import activation_path, read_labels, read_samples, sha256, write_json

RIDGE = 1e-4  # on the slope after division by the training maximum absolute value
PROBE_GRAD_TOL = 1e-8  # checked on the balanced loss in scaled-feature coordinates


class DenseFeatureMatrix:
    """Column-contiguous float32 storage for dense KAN codes on local disk."""
    def __init__(self, shape, cache_dir=None):
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="kan_features_", dir=cache_dir)
        directory = Path(self._directory.name)
        required = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        if shutil.disk_usage(directory).free < required + 64 * 1024**2:
            self._directory.cleanup()
            raise ValueError(f"KAN feature storage needs {required / 1e9:.2f} GB of free local disk.")
        self.array = np.lib.format.open_memmap(directory / "features.npy", mode="w+",
                                             dtype=np.float32, shape=shape, fortran_order=True)
        self.shape = shape

    def close(self):
        if self.array is not None:
            self.array.flush()
            self.array._mmap.close()
            self.array = None
        self._directory.cleanup()


def release_features(X):
    """Release a role's temporary disk cache after fitting or scoring."""
    if isinstance(X, DenseFeatureMatrix):
        X.close()


def open_experiment(root, data_dir, cohort, checkpoint_path):
    import torch
    from sae_checkpoint import load_model

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
        raise FileNotFoundError(f"A trained SAE best.pt was expected at {checkpoint_path}. Check SAE_TYPE and SAE_RUN.")
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
    if saved.get("model_type") in {"kan_sae", "kan_topk"} and saved.get("data_split") != training_split:
        raise ValueError("The KAN checkpoint's recorded split differs from data_split.json.")
    cfg = saved["config"]
    if cfg["d_in"] != manifest["source"]["shape"][1]:
        raise ValueError("SAE input width does not match the GraphCast embedding.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(saved, device)
    model.requires_grad_(False).eval().to(device)
    identity = dict(checkpoint_sha256=sha256(checkpoint_path),
                    checkpoint_epoch=int(saved["epoch"]), latent=cfg["latent"], k=cfg.get("k", cfg.get("k_active")),
                    cohort=cohort, dataset_manifest_sha256=sha256(manifest_path), ridge=RIDGE)
    # Keep old TopK selection identities usable; KAN records its different type.
    if saved.get("model_type") in {"kan_sae", "kan_topk"}:
        identity.update(model_type=saved["model_type"], probe_version=2)
    return model, device, manifest, labels, identity


def encode_split(role, root, manifest, labels, model, device, batch_size=256, cache_dir=None,
                 selected_feature=None):
    """Keep exact signed codes. Dense KAN uses disk; both TopK models use sparse RAM."""
    import torch
    from sae_checkpoint import feature_codes, model_type

    matrices, ys, centers = [], [], []
    samples = [r for r in manifest["samples"] if r["split"] == role]
    if not samples or batch_size < 1:
        raise ValueError("Choose a nonempty split and positive batch size.")
    dense = None
    if model_type(model) == "kan_sae" and selected_feature is None:
        shape = (len(samples) * manifest["source"]["shape"][0], model.latent)
        dense = DenseFeatureMatrix(shape, cache_dir)
        print(f"Encoding signed KAN features to {np.prod(shape) * 4 / 1e9:.2f} GB of local disk.")
    try:
        return _encode_samples(samples, root, manifest, labels, model, device,
                               batch_size, dense, feature_codes, matrices, ys, centers, selected_feature)
    except BaseException:
        if dense is not None:
            dense.close()
        raise


def _encode_samples(samples, root, manifest, labels, model, device, batch_size,
                    dense, feature_codes, matrices, ys, centers, selected_feature):
    import torch
    offset = 0
    for sample in tqdm(samples, desc="Encoding features"):
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
                code = feature_codes(model, x)
                if not torch.isfinite(code).all():
                    raise ValueError(f"Non-finite feature in {center}")
                if selected_feature is not None:
                    if not 0 <= selected_feature < code.shape[1]:
                        raise ValueError("Selected feature is outside the dictionary.")
                    values = code[:, selected_feature]
                    i = torch.where(values != 0)[0]
                    blocks.append(sparse.csr_matrix((values[i].cpu().numpy(),
                                   (i.cpu().numpy(), np.full(len(i), selected_feature))), shape=tuple(code.shape)))
                    continue
                if dense is not None:
                    dense.array[offset + start:offset + start + len(x)] = code.cpu().numpy()
                    continue
                i, j = torch.where(code != 0)
                blocks.append(sparse.csr_matrix((code[i, j].cpu().numpy(),
                                                 (i.cpu().numpy(), j.cpu().numpy())),
                                                shape=tuple(code.shape)))
        if dense is None:
            matrices.append(sparse.vstack(blocks, format="csr"))
        offset += len(arr)
        ys.append(labels[center])
        centers.append(center)
    X = dense if dense is not None else sparse.vstack(matrices, format="csc")
    if dense is not None:
        dense.array.flush()
    y = np.concatenate(ys)
    if not np.any(y) or np.all(y):
        raise ValueError(f"{role} needs both cyclone and background nodes.")
    return X, y, centers


def column(X, j):
    if isinstance(X, DenseFeatureMatrix):
        return None, np.asarray(X.array[:, j])
    lo, hi = X.indptr[j:j + 2]
    return X.indices[lo:hi], X.data[lo:hi]


def _fit_logistic_probe(z, target, w, feature):
    """Solve the same convex two-parameter loss, validating the actual solution.

    L-BFGS-B can report an abnormal line-search termination near the optimum.
    Check the loss and analytic gradient ourselves; if needed, retry with an
    exact-Hessian trust-region solver, which does not use that line search.
    """
    def objective(params):
        a, b = params
        score = a * z + b
        loss = (w * np.logaddexp(0., (1. - 2. * target) * score)).sum() + .5 * RIDGE * a * a
        residual = w * (expit(score) - target)
        return loss, np.array([(residual * z).sum() + RIDGE * a, residual.sum()])

    def hessian(params):
        score = params[0] * z + params[1]
        curvature = w * expit(score) * expit(-score)
        cross = (curvature * z).sum()
        return np.array([[(curvature * z * z).sum() + RIDGE, cross],
                         [cross, curvature.sum()]])

    initial = np.zeros(2)
    initial_loss, _ = objective(initial)

    def checked(params):
        if not np.all(np.isfinite(params)):
            return False, float("inf")
        loss, gradient = objective(params)
        grad_norm = float(np.max(np.abs(gradient)))
        good = (np.isfinite(loss) and np.isfinite(grad_norm)
                and loss <= initial_loss + 1e-12 and grad_norm <= PROBE_GRAD_TOL)
        return good, grad_norm

    opt = minimize(objective, initial, method="L-BFGS-B", jac=True,
                   options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 150, "maxls": 50})
    good, _ = checked(opt.x)
    if good:
        return opt.x, False

    # Start independently so a bad or nonfinite line-search iterate is not reused.
    retry = minimize(objective, initial, method="trust-exact", jac=True, hess=hessian,
                     options={"gtol": 1e-9, "maxiter": 150})
    good, grad_norm = checked(retry.x)
    if not good:
        raise RuntimeError(f"Probe {feature} did not converge after both solvers: "
                           f"L-BFGS-B={opt.message}; trust-exact={retry.message}; "
                           f"gradient norm={grad_norm:.3g} (required <= {PROBE_GRAD_TOL:g}).")
    return retry.x, True


def fit_probes(X, y):
    """Each fit node contributes. Implicit zero inputs are aggregated by class."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        raise ValueError("Probe fitting requires both classes.")
    weights = np.zeros((X.shape[1], 2), dtype=np.float64)  # slope per raw unit, intercept
    retries = 0
    for j in tqdm(range(X.shape[1]), desc="Fitting single-feature probes"):
        rows, value = column(X, j)
        if not np.isfinite(value).all():
            raise ValueError(f"Feature {j} values must be finite.")
        if not len(value) or (len(value) == len(y) and value.min() == value.max()):
            continue
        scale = float(np.abs(value).max())
        if scale == 0:
            continue
        pos = y if rows is None else y[rows]
        p = int(pos.sum())
        z = np.r_[value.astype(np.float64) / scale, 0., 0.]
        target = np.r_[pos, True, False].astype(np.float64)
        w = np.r_[np.where(pos, .5 / n1, .5 / n0),
                  .5 * (n1 - p) / n1, .5 * (n0 - len(pos) + p) / n0]

        params, retried = _fit_logistic_probe(z, target, w, j)
        retries += retried
        weights[j] = [params[0] / scale, params[1]]
    if retries:
        print(f"Validated all probes; {retries} needed the fallback solver.")
    return weights


def best_cutoff(value, pos, n, n1, slope):
    """Sweep tied activation values together, including all implicit zeros."""
    order = np.argsort(value, kind="stable")
    levels, starts, sizes = np.unique(value[order], return_index=True, return_counts=True)
    hits = np.add.reduceat(pos[order].astype(np.int64), starts) if len(value) else np.empty(0, dtype=np.int64)
    if n > len(value):
        zero = np.searchsorted(levels, 0)
        count, positives = n - len(value), n1 - int(pos.sum())
        if zero < len(levels) and levels[zero] == 0:
            sizes[zero] += count
            hits[zero] += positives
        else:
            levels = np.insert(levels, zero, 0)
            sizes = np.insert(sizes, zero, count)
            hits = np.insert(hits, zero, positives)
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
        cutoff, tp, pred = best_cutoff(v, y if ix is None else y[ix], len(y), n1, slope)
        ranking.append(dict(feature=int(j), slope=float(slope), intercept=float(intercept),
                            rule=">=" if slope > 0 else "<=", activation_cutoff=cutoff,
                            probability_cutoff=float(expit(slope * cutoff + intercept)),
                            **metrics(tp, pred, n1, len(y))))
    if not ranking:
        raise ValueError("No SAE feature had a nonconstant fitted probe.")
    ranking.sort(key=lambda r: (-r["f1"], r["fp"], r["feature"]))
    return ranking


def evaluate_selected(X, y, centers, winner):
    alpha = (np.asarray(X.array[:, winner["feature"]]) if isinstance(X, DenseFeatureMatrix)
             else X.getcol(winner["feature"]).toarray().ravel())
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
