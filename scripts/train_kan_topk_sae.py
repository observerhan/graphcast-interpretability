"""Train the encoder-spline KAN + TopK experiment on the saved TopK split.

The existing KAN trainer is intentionally not modified: its source checksum
is part of the resume identity of the user's completed L1 experiments.
"""
import csv
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cyclone_data import sha256, write_json
from graphcast_interpretability.kan_sae import KANSAE
from graphcast_interpretability.kan_topk_sae import KANTopKSAE, magnitude_topk
from train_kan_sae import _cpu_state, _save_torch, batches, calibration_sample, prepare_data


@dataclass
class TopKTrainingConfig:
    latent: int = 4096
    k_active: int = 32
    warmup_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-4
    seed: int = 0
    calibration_samples: int = 50_000
    activity_threshold: float = 1e-6
    dead_window: int = 3_000_000
    k_aux: int = 512
    aux_weight: float = 1 / 32

    def validate(self):
        for name in ("latent", "k_active", "warmup_epochs", "batch_size",
                     "calibration_samples", "dead_window", "k_aux"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.k_active > self.latent:
            raise ValueError("k_active must not exceed latent.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in ("learning_rate", "activity_threshold", "aux_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.learning_rate == 0 or self.activity_threshold == 0:
            raise ValueError("Use positive learning rate and activity threshold.")


class KANTopKTrainer:
    """Dense initialization, then fixed-k training with resumable AuxK state.

    Every validation pass uses the final k. Only epochs trained with that gate
    can produce best.pt. Warm-up never competes with sparse checkpoints.
    """
    def __init__(self, data, run_dir, config=None, device=None, code_revision=None):
        self.data, self.run_dir = data, Path(run_dir)
        self.cfg = config or TopKTrainingConfig()
        self.cfg.validate()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.code_revision = code_revision
        torch.manual_seed(self.cfg.seed)
        self.model = KANTopKSAE(data.split["source"]["shape"][1], self.cfg.latent,
                               k_active=self.cfg.k_active).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.miss_counts = torch.zeros(self.cfg.latent, dtype=torch.long, device=self.device)
        self.training_config = dict(asdict(self.cfg), trainer_version=1, l1_weight=0.,
            kan_source_sha256=sha256(Path(inspect.getfile(KANSAE))),
            model_source_sha256=sha256(Path(inspect.getfile(KANTopKSAE))),
            data_helpers_sha256=sha256(Path(inspect.getfile(prepare_data))),
            trainer_source_sha256=sha256(Path(__file__)))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        last = self.run_dir / "last.pt"
        if last.exists():
            saved = torch.load(last, map_location="cpu", weights_only=True)
            if (saved.get("model_type") != "kan_topk" or saved["config"] != self.model.config()
                    or saved["training_config"] != self.training_config or saved["data_split"] != data.split):
                raise ValueError("Data or training settings changed. Choose a new RUN_NAME.")
            self.model.load_state_dict(saved["model_state"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer_state"])
            self.miss_counts.copy_(saved["miss_counts"])
            self.train_mean = saved["train_mean"].to(self.device)
            self.epoch, self.history = saved["epoch"], saved["history"]
            self.best_epoch, self.best_error = saved["best_epoch"], saved["best_error"]
            self.best_state = saved["best_model_state"]
            self.calibration, self.calibration_indices = saved["calibration"], saved["calibration_indices"]
            if self.best_epoch is not None and self.best_epoch <= self.cfg.warmup_epochs:
                raise ValueError("A warm-up checkpoint cannot be selected as the sparse model.")
            torch.set_rng_state(saved["torch_rng_state"])
            if self.device.type == "cuda" and saved.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
            self._write_outputs()
            print(f"Continuing after epoch {self.epoch}; spline grids and feature activity restored.")
        else:
            if any(self.run_dir.iterdir()):
                raise ValueError("Run directory contains files but no last.pt. Choose a new RUN_NAME.")
            total = torch.zeros(self.model.d_in, device=self.device, dtype=torch.float64)
            count = 0
            print("Computing the mean from training dates only...")
            with torch.no_grad():
                for x in self._batches(data.train_files):
                    total += self.model.reconstruction_target(x).double().sum(0)
                    count += len(x)
            self.train_mean = (total / count).float()
            sample, self.calibration_indices = calibration_sample(
                data.train_files, self.cfg.calibration_samples, self.cfg.seed)
            self.model.initialize_from_data(sample, training_mean=self.train_mean)
            self.calibration = dict(method="uniform training rows without replacement", seed=self.cfg.seed,
                count=len(sample), quantiles=[0.01, 0.99], training_files=data.split["training_files"],
                indices_sha256=hashlib.sha256(self.calibration_indices.numpy().tobytes()).hexdigest())
            del sample
            self.epoch, self.best_epoch, self.best_state, self.best_error = 0, None, None, None
            self.history = [dict(epoch=0, phase="initial", training_error=None,
                auxiliary_error=None, training_loss=None, training_mean_active=None,
                training_dead_features=0, **self.evaluate())]
            self._save_last()
            self._write_outputs()

    def _batches(self, paths, epoch=None):
        return batches(paths, self.cfg.batch_size, self.device, self.cfg.seed, epoch)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        error = baseline = l1 = active = effective = outside = selected_outside = count = 0
        max_active = 0
        used = torch.zeros(self.cfg.latent, dtype=torch.bool, device=self.device)
        for x in self._batches(self.data.val_files):
            target = self.model.reconstruction_target(x)
            reconstruction, code = self.model(x)  # Always final k, including warm-up evaluations.
            if not torch.isfinite(reconstruction).all() or not torch.isfinite(code).all():
                raise ValueError("Validation produced nonfinite values.")
            nonzero = code != 0
            max_active = max(max_active, int(nonzero.sum(1).max()))
            if max_active > self.cfg.k_active:
                raise ValueError("The saved inference gate exceeded its activation budget.")
            error += (target - reconstruction).square().sum().item()
            baseline += (target - self.train_mean).square().sum().item()
            l1 += code.abs().sum().item()
            active += nonzero.sum().item()
            effective += (code.abs() > self.cfg.activity_threshold).sum().item()
            used |= nonzero.any(0)
            h = self.model.enc(target - self.model.b_pre) + self.model.b_enc
            spline = self.model.spline
            clamped = (h < spline.knots[:, spline.degree]) | (h > spline.knots[:, spline.n_basis])
            outside += clamped.sum().item()
            selected_outside += (clamped & nonzero).sum().item()
            count += len(x)
        return dict(validation_error=error / count, mean_baseline_error=baseline / count,
            fraction_variance_unexplained=error / baseline if baseline > 0 else None,
            validation_l1=l1 / count, mean_active_features=active / count,
            max_active_features=max_active, mean_active_above_threshold=effective / count,
            features_used_on_validation=int(used.sum()),
            validation_clamped_fraction=outside / (count * self.cfg.latent),
            selected_clamped_fraction=selected_outside / active if active else 0.)

    def _common_checkpoint(self):
        return dict(model_type="kan_topk", config=self.model.config(), training_config=self.training_config,
            data_signature=self.data.split["data_signature"], validation_dates=self.data.split["validation_dates"],
            data_split=self.data.split, train_mean=self.train_mean.cpu(), calibration=self.calibration,
            code_revision=self.code_revision, torch_version=str(torch.__version__))

    def _save_last(self):
        _save_torch(self.run_dir / "last.pt", dict(self._common_checkpoint(),
            model_state=_cpu_state(self.model), optimizer_state=self.optimizer.state_dict(),
            miss_counts=self.miss_counts.cpu(), epoch=self.epoch, history=self.history,
            best_epoch=self.best_epoch, best_error=self.best_error, best_model_state=self.best_state,
            calibration_indices=self.calibration_indices, torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []))

    def _write_outputs(self):
        if self.best_state is not None:
            _save_torch(self.run_dir / "best.pt", dict(self._common_checkpoint(),
                        model_state=self.best_state, epoch=self.best_epoch))
        write_json(self.run_dir / "data_split.json", self.data.split)
        write_json(self.run_dir / "config.json", dict(model_type="kan_topk", model=self.model.config(),
                    training=self.training_config, code_revision=self.code_revision))
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

    def batch_losses(self, x, *, warmup):
        """AuxK has a detached residual and never changes the main sparse code."""
        target = self.model.reconstruction_target(x)
        dense = self.model.encode_dense(x)
        code = dense if warmup else magnitude_topk(dense, self.cfg.k_active)
        reconstruction = self.model.decode(code)
        reconstruction_loss = (target - reconstruction).square().sum(1).mean()
        auxiliary_loss = reconstruction_loss.new_zeros(())
        dead = self.miss_counts >= self.cfg.dead_window
        if not warmup and self.cfg.aux_weight > 0 and bool(dead.any()):
            auxiliary_code = magnitude_topk(dense, min(self.cfg.k_aux, self.cfg.latent), allowed=dead)
            auxiliary_reconstruction = self.model.dec(auxiliary_code)  # Residual has no decoder bias.
            residual = (target - reconstruction).detach()
            auxiliary_loss = (residual - auxiliary_reconstruction).square().sum(1).mean()
        return reconstruction_loss + self.cfg.aux_weight * auxiliary_loss, reconstruction_loss, auxiliary_loss, code

    def fit(self, epochs):
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < max(1, self.epoch):
            raise ValueError("EPOCHS must be positive and at least the last completed epoch.")
        for epoch in range(self.epoch + 1, epochs + 1):
            warmup = epoch <= self.cfg.warmup_epochs
            phase = "dense_warmup" if warmup else "sparse"
            self.model.train()
            error = auxiliary = loss_sum = active = count = 0
            for x in tqdm(self._batches(self.data.train_files, epoch), desc=f"Epoch {epoch}/{epochs} ({phase})"):
                self.optimizer.zero_grad(set_to_none=True)
                loss, main, aux, code = self.batch_losses(x, warmup=warmup)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; the last completed epoch remains saved.")
                loss.backward()
                # Match the baseline's optimization on the unit-column sphere.
                with torch.no_grad():
                    decoder = self.model.dec.weight
                    if decoder.grad is not None:
                        decoder.grad.sub_((decoder.grad * decoder).sum(0, keepdim=True) * decoder)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"), error_if_nonfinite=True)
                self.optimizer.step()
                self.model.renormalize_decoder_()
                with torch.no_grad():
                    if not warmup:
                        self.miss_counts.add_(len(x))
                        self.miss_counts.masked_fill_((code != 0).any(0), 0)
                error += main.item() * len(x)
                auxiliary += aux.item() * len(x)
                loss_sum += loss.item() * len(x)
                active += (code != 0).sum().item()
                count += len(x)
            result = self.evaluate()
            self.history.append(dict(epoch=epoch, phase=phase, training_error=error / count,
                auxiliary_error=auxiliary / count, training_loss=loss_sum / count,
                training_mean_active=active / count,
                training_dead_features=int((self.miss_counts >= self.cfg.dead_window).sum()), **result))
            if not warmup and (self.best_error is None or result["validation_error"] < self.best_error):
                self.best_epoch, self.best_error = epoch, result["validation_error"]
                self.best_state = _cpu_state(self.model)
            self.epoch = epoch
            self._save_last()
            self._write_outputs()
            print(f"Epoch {epoch}: gated validation error {result['validation_error']:.6f}; "
                  f"active features {result['mean_active_features']:.1f}; "
                  f"best sparse epoch {self.best_epoch or 'pending'}.")
        return self.history

    def summarize(self):
        if self.best_state is None:
            raise ValueError("Complete at least one epoch after warm-up before probing this run.")
        current = _cpu_state(self.model)
        try:
            self.model.load_state_dict(self.best_state)
            result = dict(best_epoch=self.best_epoch, model_type="kan_topk", k_active=self.cfg.k_active,
                warmup_epochs=self.cfg.warmup_epochs, validation_dates=self.data.split["validation_dates"],
                activity_threshold=self.cfg.activity_threshold, **self.evaluate())
        finally:
            self.model.load_state_dict(current)
        write_json(self.run_dir / "summary.json", result)
        return result

    def plot_history(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sparse_rows = [r for r in self.history if r["phase"] == "sparse"]
        axes[0].plot([r["epoch"] for r in sparse_rows], [r["training_error"] for r in sparse_rows], label="Sparse training")
        axes[0].plot([r["epoch"] for r in self.history], [r["validation_error"] for r in self.history], label="Gated validation")
        axes[0].axhline(self.history[0]["mean_baseline_error"], color="gray", ls="--", label="Training mean")
        axes[0].set(xlabel="Epoch", ylabel="Squared error per vector", title="KAN + TopK reconstruction")
        axes[0].legend()
        axes[1].plot([r["epoch"] for r in self.history], [r["mean_active_features"] for r in self.history])
        axes[1].axhline(self.cfg.k_active, color="gray", ls="--")
        axes[1].set(xlabel="Epoch", ylabel="Nonzero features per input", title="Validation sparsity")
        axes[2].plot([r["epoch"] for r in self.history], [r["features_used_on_validation"] for r in self.history])
        axes[2].axhline(self.cfg.latent, color="gray", ls="--")
        axes[2].set(xlabel="Epoch", ylabel="Features used across validation", title="Dictionary use")
        for ax in axes:
            ax.axvspan(0, self.cfg.warmup_epochs, color="gray", alpha=0.1)
        fig.tight_layout()
        fig.savefig(self.run_dir / "training_curve.png", dpi=160)
        return fig
