"""Resumable KAN-SAE training on the exact activation split saved by TopK.

The Colab notebook supplies paths and settings. Numerical work stays here so
it can be tested on CPU without Drive, GraphCast, or notebook-source parsing.
"""
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm.auto import tqdm

from cyclone_data import sha256, write_json
from graphcast_interpretability.kan_sae import KANSAE


@dataclass
class TrainingConfig:
    latent: int = 4096
    batch_size: int = 256
    learning_rate: float = 1e-4
    seed: int = 0
    calibration_samples: int = 50_000
    l1_start: float = 5e-5
    l1_end: float = 1e-4
    l1_ramp_epochs: int = 20
    activity_threshold: float = 1e-6

    def validate(self):
        for name in ("latent", "batch_size", "calibration_samples", "l1_ramp_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in ("learning_rate", "activity_threshold", "l1_start", "l1_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.learning_rate == 0 or self.activity_threshold == 0 or self.l1_end < self.l1_start:
            raise ValueError("Use positive learning rate/activity threshold and l1_end >= l1_start.")

    def l1_at_epoch(self, epoch):
        fraction = min(max(epoch - 1, 0) / max(self.l1_ramp_epochs - 1, 1), 1.)
        if self.l1_ramp_epochs == 1:
            fraction = 1.
        return self.l1_start + fraction * (self.l1_end - self.l1_start)


@dataclass
class ActivationData:
    train_files: list
    val_files: list
    split: dict


def prepare_data(activation_dir, reference_split, cache_dir):
    """Copy and checksum the files in TopK's data_split.json, without resplitting."""
    activation_dir, reference_split, cache_dir = map(Path, (activation_dir, reference_split, cache_dir))
    if not reference_split.is_file():
        raise FileNotFoundError("Run notebook 02 first: its TopK data_split.json defines this comparison.")
    reference = json.loads(reference_split.read_text())
    manifest = json.loads((activation_dir / "dataset.json").read_text())
    source = reference["source"]
    if source != manifest["source"] or source.get("layer") != 8 or source.get("dtype") != "float32":
        raise ValueError("GraphCast source differs from the TopK reference split.")
    shape = source["shape"]
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Expected a two-dimensional GraphCast activation shape.")
    train_names, val_names = reference["training_files"], reference["validation_files"]
    names = train_names + val_names
    if not train_names or not val_names or len(names) != len(set(names)):
        raise ValueError("Training and validation files must be nonempty and disjoint.")
    training_dates, validation_dates = reference["training_dates"], reference["validation_dates"]
    if set(training_dates) & set(validation_dates):
        raise ValueError("Training and validation dates overlap.")
    requested = {f"layer0008_t{t}.npy" for t in manifest["requested_centers"]}
    for group, dates in ((train_names, training_dates), (val_names, validation_dates)):
        actual_dates = set()
        for name in group:
            if Path(name).name != name or name not in requested:
                raise ValueError(f"Invalid or unlisted activation file: {name}")
            actual_dates.add(datetime.strptime(name, "layer0008_t%Y-%m-%dT%H.npy").date().isoformat())
        if actual_dates != set(dates):
            raise ValueError("The reference dates and files disagree.")
    expected = {r["file"]: r["sha256"] for r in reference["data_signature"]}
    if set(expected) != set(names) or len(expected) != len(reference["data_signature"]):
        raise ValueError("Reference checksums do not cover exactly the selected files.")
    # Never overwrite source activations through a mistakenly configured cache.
    if cache_dir.resolve() == activation_dir.resolve():
        raise ValueError("The local cache must be separate from the activation directory.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in tqdm(sorted(names), desc="Checking TopK reference activations"):
        local = cache_dir / name
        temporary = local.with_suffix(".npy.tmp")
        shutil.copyfile(activation_dir / name, temporary)
        temporary.replace(local)
        array = np.load(local, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != tuple(shape) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid activation array: {name}")
        if sha256(local) != expected[name]:
            raise ValueError(f"{name} differs from the file used for TopK training.")
    keys = ("source", "training_dates", "validation_dates", "training_files", "validation_files", "data_signature")
    split = {key: reference[key] for key in keys}
    split["reference_split_sha256"] = sha256(reference_split)
    return ActivationData([cache_dir / n for n in train_names], [cache_dir / n for n in val_names], split)


def batches(paths, batch_size, device, seed=0, epoch=None):
    rng = np.random.default_rng(seed + (epoch or 0))
    order = list(paths)
    if epoch is not None:
        rng.shuffle(order)
    for path in order:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        indices = rng.permutation(len(array)) if epoch is not None else np.arange(len(array))
        for start in range(0, len(array), batch_size):
            yield torch.from_numpy(np.array(array[indices[start:start + batch_size]], copy=True)).to(device)


def calibration_sample(paths, count, seed):
    """Uniform global training-row sample, without replacement, with saved indices."""
    arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
    ends = np.cumsum([len(a) for a in arrays])
    indices = np.sort(np.random.default_rng(seed).choice(int(ends[-1]), min(count, int(ends[-1])), replace=False))
    sample = np.empty((len(indices), arrays[0].shape[1]), dtype=np.float32)
    begin = 0
    for array, end in zip(arrays, ends):
        lo, hi = np.searchsorted(indices, [begin, end])
        sample[lo:hi] = array[indices[lo:hi] - begin]
        begin = end
    return torch.from_numpy(sample), torch.from_numpy(indices.astype(np.int64))


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class KANTrainer:
    """Save last.pt as the authoritative completed-epoch state, then derive outputs."""
    def __init__(self, data, run_dir, config=None, device=None, code_revision=None):
        self.data, self.run_dir = data, Path(run_dir)
        self.cfg = config or TrainingConfig()
        self.cfg.validate()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.code_revision = code_revision
        torch.manual_seed(self.cfg.seed)
        self.model = KANSAE(d_in=data.split["source"]["shape"][1], latent=self.cfg.latent).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.training_config = {**asdict(self.cfg), "trainer_version": 1,
                                "model_source_sha256": sha256(Path(inspect.getfile(KANSAE))),
                                "trainer_source_sha256": sha256(Path(__file__))}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        last_path = self.run_dir / "last.pt"
        if last_path.exists():
            saved = torch.load(last_path, map_location="cpu", weights_only=True)
            if (saved.get("model_type") != "kan_sae" or saved["config"] != self.model.config()
                    or saved["training_config"] != self.training_config or saved["data_split"] != data.split):
                raise ValueError("Data or training settings changed. Choose a new RUN_NAME.")
            self.model.load_state_dict(saved["model_state"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer_state"])
            self.train_mean = saved["train_mean"].to(self.device)
            self.epoch, self.history = saved["epoch"], saved["history"]
            self.best_epoch, self.best_error = saved["best_epoch"], saved["best_error"]
            self.best_state = saved["best_model_state"]
            self.calibration_indices = saved["calibration_indices"]
            self.calibration = saved["calibration"]
            torch.set_rng_state(saved["torch_rng_state"])
            if self.device.type == "cuda" and saved.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
            self._write_outputs()  # Repair interrupted best.pt / CSV writes from last.pt.
            print(f"Continuing after completed epoch {self.epoch}; calibrated knots restored.")
        else:
            if any(self.run_dir.iterdir()):
                raise ValueError("Run directory contains files but no last.pt. Choose a new RUN_NAME.")
            print("Computing the mean using training dates only...")
            total = torch.zeros(self.model.d_in, device=self.device, dtype=torch.float64)
            count = 0
            with torch.no_grad():
                for x in self._batches(data.train_files):
                    total += self.model.reconstruction_target(x).double().sum(dim=0)
                    count += len(x)
            self.train_mean = (total / count).float()
            print("Calibrating spline grids from training rows...")
            sample, self.calibration_indices = calibration_sample(data.train_files, self.cfg.calibration_samples, self.cfg.seed)
            self.model.initialize_from_data(sample, training_mean=self.train_mean)
            self.calibration = dict(method="uniform training rows without replacement", seed=self.cfg.seed,
                                    count=len(sample), quantiles=[0.01, 0.99],
                                    training_files=data.split["training_files"],
                                    indices_sha256=hashlib.sha256(self.calibration_indices.numpy().tobytes()).hexdigest())
            del sample
            metrics = self.evaluate()
            self.epoch = self.best_epoch = 0
            self.best_error = metrics["validation_error"]
            self.best_state = _cpu_state(self.model)
            self.history = [dict(epoch=0, training_error=None, training_l1=None, training_loss=None,
                                 l1_weight=self.cfg.l1_start, **metrics)]
            self._save_last()
            self._write_outputs()

    def _batches(self, paths, epoch=None):
        return batches(paths, self.cfg.batch_size, self.device, self.cfg.seed, epoch)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        error = baseline = l1 = active = outside = count = 0
        used = torch.zeros(self.cfg.latent, dtype=torch.bool, device=self.device)
        for x in self._batches(self.data.val_files):
            target = self.model.reconstruction_target(x)
            reconstruction, code = self.model(x)
            if not torch.isfinite(reconstruction).all() or not torch.isfinite(code).all():
                raise ValueError("Validation produced nonfinite values.")
            error += (target - reconstruction).square().sum().item()
            baseline += (target - self.train_mean).square().sum().item()
            l1 += code.abs().sum().item()
            active_mask = code.abs() > self.cfg.activity_threshold
            active += active_mask.sum().item()
            used |= active_mask.any(dim=0)
            h = self.model.enc(target - self.model.b_pre) + self.model.b_enc
            knots, degree, n_basis = self.model.spline.knots, self.model.spline.degree, self.model.spline.n_basis
            outside += ((h < knots[:, degree]) | (h > knots[:, n_basis])).sum().item()
            count += len(x)
        return dict(validation_error=error / count, mean_baseline_error=baseline / count,
                    fraction_variance_unexplained=error / baseline if baseline > 0 else None,
                    validation_l1=l1 / count, mean_active_features=active / count,
                    features_used_on_validation=int(used.sum()),
                    validation_clamped_fraction=outside / (count * self.cfg.latent))

    def _common_checkpoint(self):
        return dict(model_type="kan_sae", config=self.model.config(), training_config=self.training_config,
                    data_signature=self.data.split["data_signature"], validation_dates=self.data.split["validation_dates"],
                    data_split=self.data.split, train_mean=self.train_mean.cpu(), calibration=self.calibration,
                    code_revision=self.code_revision, torch_version=str(torch.__version__))

    def _save_last(self):
        payload = dict(self._common_checkpoint(), model_state=_cpu_state(self.model),
                       optimizer_state=self.optimizer.state_dict(), epoch=self.epoch, history=self.history,
                       best_epoch=self.best_epoch, best_error=self.best_error, best_model_state=self.best_state,
                       calibration_indices=self.calibration_indices, torch_rng_state=torch.get_rng_state(),
                       cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
        _save_torch(self.run_dir / "last.pt", payload)

    def _write_outputs(self):
        _save_torch(self.run_dir / "best.pt", dict(self._common_checkpoint(),
                    model_state=self.best_state, epoch=self.best_epoch))
        write_json(self.run_dir / "data_split.json", self.data.split)
        write_json(self.run_dir / "config.json", dict(model=self.model.config(), training=self.training_config,
                                                       model_type="kan_sae", code_revision=self.code_revision))
        write_json(self.run_dir / "calibration.json", self.calibration)
        path = self.run_dir / "calibration_indices.npy"
        with path.with_suffix(".npy.tmp").open("wb") as handle:
            np.save(handle, self.calibration_indices.numpy(), allow_pickle=False)
        path.with_suffix(".npy.tmp").replace(path)
        path = self.run_dir / "history.csv"
        with path.with_suffix(".csv.tmp").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        path.with_suffix(".csv.tmp").replace(path)

    def fit(self, epochs):
        if not isinstance(epochs, int) or epochs < self.epoch or epochs < 1:
            raise ValueError("EPOCHS must be positive and at least the last completed epoch.")
        for epoch in range(self.epoch + 1, epochs + 1):
            self.model.train()
            weight = self.cfg.l1_at_epoch(epoch)
            error = l1 = total_loss = count = 0
            for x in tqdm(self._batches(self.data.train_files, epoch), desc=f"Epoch {epoch}/{epochs}"):
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model.loss(x, *self.model(x), l1_weight=weight)
                if not torch.isfinite(losses["loss"]):
                    raise ValueError("Nonfinite loss; the last completed epoch remains saved.")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"), error_if_nonfinite=True)
                self.optimizer.step()
                self.model.renormalize_decoder_()
                error += losses["reconstruction_loss"].item() * len(x)
                l1 += losses["l1_loss"].item() * len(x)
                total_loss += losses["loss"].item() * len(x)
                count += len(x)
            metrics = self.evaluate()
            self.history.append(dict(epoch=epoch, training_error=error / count, training_l1=l1 / count,
                                     training_loss=total_loss / count, l1_weight=weight, **metrics))
            if metrics["validation_error"] < self.best_error:
                self.best_epoch, self.best_error = epoch, metrics["validation_error"]
                self.best_state = _cpu_state(self.model)
            self.epoch = epoch
            self._save_last()  # All derived artifacts can be rebuilt if interrupted after here.
            self._write_outputs()
            print(f"Epoch {epoch}: validation error {metrics['validation_error']:.6f}; "
                  f"mean |code| sum {metrics['validation_l1']:.4f}; best epoch {self.best_epoch}.")
        return self.history

    def summarize(self):
        """Evaluate the selected checkpoint, leaving the current training weights intact."""
        current = _cpu_state(self.model)
        try:
            self.model.load_state_dict(self.best_state)
            result = dict(best_epoch=self.best_epoch, model_type="kan_sae",
                          validation_dates=self.data.split["validation_dates"],
                          activity_threshold=self.cfg.activity_threshold, **self.evaluate())
        finally:
            self.model.load_state_dict(current)
        write_json(self.run_dir / "summary.json", result)
        return result

    def plot_history(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        trained = [r for r in self.history if r["epoch"] > 0]
        axes[0].plot([r["epoch"] for r in trained], [r["training_error"] for r in trained], label="Training")
        axes[0].plot([r["epoch"] for r in self.history], [r["validation_error"] for r in self.history], label="Validation")
        axes[0].axhline(self.history[0]["mean_baseline_error"], color="gray", ls="--", label="Training mean")
        axes[0].set(xlabel="Epoch", ylabel="Squared error per vector", title="KAN-SAE reconstruction")
        axes[0].legend()
        axes[1].plot([r["epoch"] for r in self.history], [r["validation_l1"] for r in self.history])
        axes[1].set(xlabel="Epoch", ylabel="Mean sum of absolute feature values", title="Validation L1")
        fig.tight_layout()
        fig.savefig(self.run_dir / "training_curve.png", dpi=160)
        return fig
