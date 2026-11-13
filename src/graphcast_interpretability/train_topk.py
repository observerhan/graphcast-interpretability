"""Train the existing SAE class on cached GraphCast node activations.

Run ``python -m graphcast_interpretability.train_topk --help`` for the CLI.
Only the SAE is optimized; GraphCast is not imported or run here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .model import SAE, _project_decoder_grads_orthogonal, _renorm_decoder_columns_

PathLike = Union[str, Path]


@dataclass
class TrainConfig:
    output_dir: str
    d_in: int = 512
    latent: int = 4096
    k: int = 32
    k_aux: int = 512
    aux_weight: float = 1 / 32
    dead_window: int = 5_000_000
    batch_size: int = 256
    epochs: int = 5
    lr: float = 3e-4
    seed: int = 0
    device: str = "auto"
    log_every: int = 50

    def validate(self) -> None:
        for name in (
            "d_in",
            "latent",
            "k",
            "k_aux",
            "dead_window",
            "batch_size",
            "epochs",
            "log_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.k > self.latent or self.k_aux > self.latent:
            raise ValueError("k and k_aux must not exceed latent")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.aux_weight) or self.aux_weight < 0:
            raise ValueError("aux_weight must be finite and nonnegative")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def find_activation_files(
    directories: PathLike | Sequence[PathLike], prefix: str = "layer0008_"
) -> list[Path]:
    """Find one layer's .npy files; do not load an entire activation corpus."""
    if isinstance(directories, (str, Path)):
        directories = [directories]
    paths = sorted(
        {p.resolve() for root in directories for p in Path(root).glob(f"{prefix}*.npy")}
    )
    if not paths:
        raise FileNotFoundError(
            f"No {prefix}*.npy activation files found in {directories}"
        )
    return paths


def _day(path: PathLike) -> str | None:
    # ActivationManager filenames include _tYYYY-MM-DDT... or _tYYYYMMDDT....
    match = re.search(r"_t(\d{4})-?(\d{2})-?(\d{2})(?:T|_|$)", Path(path).stem)
    return date(*map(int, match.groups())).isoformat() if match else None


def split_by_day(
    files: Sequence[PathLike], val_fraction: float = 0.2
) -> tuple[list[Path], list[Path]]:
    """Hold out the latest calendar days, keeping all snapshots of a day together.

    For event-level splits or filenames without dates, supply explicit training
    and validation file lists to train_topk instead.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    paths = sorted({Path(p).resolve() for p in files})
    days = {_day(p) for p in paths}
    if None in days:
        raise ValueError(
            "Automatic splitting needs dated filenames (_tYYYY-MM-DDT...); use explicit train/validation directories otherwise"
        )
    if len(days) < 2:
        raise ValueError(
            "At least two different dates are needed for validation; extract more dates or supply an independent validation set"
        )
    n_val = min(len(days) - 1, max(1, math.ceil(len(days) * val_fraction)))
    val_days = set(sorted(days)[-n_val:])
    return (
        [p for p in paths if _day(p) not in val_days],
        [p for p in paths if _day(p) in val_days],
    )


def _load_array(path: PathLike, d_in: int) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != d_in or array.shape[0] == 0:
        raise ValueError(
            f"{path}: expected nonempty [N, {d_in}] or [N, 1, {d_in}], got {array.shape}"
        )
    if array.dtype.kind != "f":
        raise ValueError(
            f"{path}: expected a real floating-point array, got {array.dtype}. "
            "At extraction, save np.asarray(x).astype(np.float32). "
            "Legacy |V2/bfloat16 bytes must not be reinterpreted as float16."
        )
    return array


def _file_metadata(files: Sequence[PathLike], d_in: int) -> list[dict]:
    result = []
    for path in sorted({Path(p).resolve() for p in files}):
        array = _load_array(path, d_in)
        stat = path.stat()
        result.append(
            {
                "path": str(path),
                "rows": int(array.shape[0]),
                "dtype": str(array.dtype),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return result


def _validate_split(
    train_files: Sequence[PathLike], val_files: Sequence[PathLike]
) -> None:
    train = {Path(p).resolve() for p in train_files}
    val = {Path(p).resolve() for p in val_files}
    if not train or not val:
        raise ValueError(
            "Both training and validation need at least one activation file"
        )
    if train & val:
        raise ValueError("Training and validation contain the same file")
    common_days = ({_day(p) for p in train} & {_day(p) for p in val}) - {None}
    if common_days:
        raise ValueError(
            f"Training and validation share dates {sorted(common_days)}; keep entire days/events together"
        )


def iter_batches(
    files: Sequence[PathLike],
    d_in: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Stream float32 CPU batches, including the last partial batch of every file."""
    rng = np.random.default_rng(seed)
    order = list(files)
    if shuffle:
        rng.shuffle(order)
    for path in order:
        array = _load_array(path, d_in)
        indices = rng.permutation(array.shape[0]) if shuffle else None
        for start in range(0, array.shape[0], batch_size):
            selected = (
                indices[start : start + batch_size]
                if shuffle
                else slice(start, start + batch_size)
            )
            with np.errstate(over="ignore", invalid="ignore"):
                batch = np.array(array[selected], dtype=np.float32, copy=True)
            if not np.isfinite(batch).all():
                raise ValueError(
                    f"Nonfinite values in {path}, batch starting at row {start}"
                )
            yield torch.from_numpy(batch)


def normalize_targets(x: torch.Tensor) -> torch.Tensor:
    """Exactly match SAE.forward's per-row centering and L2 normalization."""
    centered = x - x.mean(dim=1, keepdim=True)
    return centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-6)


def compute_losses(
    target: torch.Tensor,
    recon: torch.Tensor,
    aux_recon: torch.Tensor,
    *,
    has_dead: bool,
    aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared L2 error per vector; AuxK fits the detached main residual."""
    reconstruction = (target - recon).square().sum(dim=1).mean()
    auxiliary = (
        ((target - recon).detach() - aux_recon).square().sum(dim=1).mean()
        if has_dead
        else recon.new_zeros(())
    )
    return reconstruction + aux_weight * auxiliary, reconstruction, auxiliary


@torch.no_grad()
def training_mean(files: Sequence[PathLike], config: TrainConfig) -> torch.Tensor:
    total = torch.zeros(config.d_in, dtype=torch.float64)
    count = 0
    for x in iter_batches(files, config.d_in, config.batch_size):
        total += normalize_targets(x).double().sum(dim=0)
        count += len(x)
    return (total / count).float()


@torch.no_grad()
def evaluate(
    model: SAE,
    files: Sequence[PathLike],
    config: TrainConfig,
    mean: torch.Tensor,
    device: torch.device,
) -> dict:
    """Evaluate without updating the dead-feature counters or using validation to fit the mean."""
    model.eval()
    seen = 0
    squared_error = baseline_error = active_count = 0.0
    used = torch.zeros(config.latent, dtype=torch.bool, device=device)
    mean = mean.to(device)
    for x in iter_batches(files, config.d_in, config.batch_size):
        x = x.to(device)
        target = normalize_targets(x)
        recon, code, _ = model(x)  # raw x goes to the authors' class
        if not torch.isfinite(recon).all() or not torch.isfinite(code).all():
            raise FloatingPointError("Nonfinite SAE validation output")
        squared_error += (target - recon).square().sum().item()
        baseline_error += (target - mean).square().sum().item()
        active_count += (code > 0).sum().item()
        used |= (code > 0).any(dim=0)
        seen += len(x)
    return {
        "reconstruction_mse": squared_error / seen,
        "mean_baseline_mse": baseline_error / seen,
        "fvu": squared_error / baseline_error if baseline_error > 1e-12 else None,
        "mean_l0": active_count / seen,
        "fraction_features_used": used.float().mean().item(),
        "samples": seen,
    }


def _json_write(path: Path, content: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _checkpoint_write(path: Path, content: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def _history_write(path: Path, history: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but no CUDA GPU is available; select a GPU runtime or use --device cpu"
        )
    return device


def train_topk(
    train_files: Sequence[PathLike],
    val_files: Sequence[PathLike],
    config: TrainConfig,
    *,
    resume: PathLike | None = None,
) -> dict:
    """Train or resume at an epoch boundary. config.epochs is the total desired epoch count."""
    config.validate()
    train_files = sorted({Path(p).resolve() for p in train_files})
    val_files = sorted({Path(p).resolve() for p in val_files})
    _validate_split(train_files, val_files)
    source_path = Path(inspect.getfile(SAE))
    manifest = {
        "train": _file_metadata(train_files, config.d_in),
        "validation": _file_metadata(val_files, config.d_in),
        "sae_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    device = _device(config.device)
    output = Path(config.output_dir)
    if resume is not None and output.resolve() != Path(resume).resolve().parent:
        raise ValueError(
            "Resume into the original run directory containing last.pt and best.pt"
        )
    if resume is None and any(
        (output / name).exists()
        for name in (
            "config.json",
            "data_manifest.json",
            "last.pt",
            "best.pt",
            "history.csv",
        )
    ):
        raise FileExistsError(
            f"{output} already contains a run; choose a new output directory or use resume"
        )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = SAE(
        config.d_in,
        config.latent,
        config.k,
        config.k_aux,
        unit_norm_decoder=True,
        dead_window=config.dead_window,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_epoch = global_step = samples_seen = 0
    best_mse = float("inf")
    history = []
    loaded = None
    if resume is not None:
        # weights_only works with the tensors and primitive metadata saved here.
        loaded = torch.load(resume, map_location="cpu", weights_only=True)
        if loaded.get("format_version") != 1:
            raise ValueError("Resume requires a checkpoint produced by this trainer")
        saved_config = loaded["config"]
        for name, value in asdict(config).items():
            if (
                name not in ("output_dir", "epochs", "device", "log_every")
                and saved_config[name] != value
            ):
                raise ValueError(f"Resume configuration mismatch: {name}")
        if loaded["data_manifest"] != manifest:
            raise ValueError(
                "Resume data/source mismatch: reuse the same files and SAE implementation"
            )
        model.load_state_dict(loaded["model_state"], strict=True)
        optimizer.load_state_dict(loaded["optimizer_state"])
        start_epoch = loaded["epoch"]
        if config.epochs <= start_epoch:
            raise ValueError(
                f"Checkpoint has completed {start_epoch} epochs; set epochs to a larger total"
            )
        global_step, samples_seen = loaded["global_step"], loaded["samples_seen"]
        best_mse, history = loaded["best_val_mse"], loaded["history"]
        mean = loaded["training_mean"]
        initial_validation = loaded["initial_validation"]
        torch.set_rng_state(loaded["torch_rng_state"])
        if (
            device.type == "cuda"
            and loaded["cuda_rng_states"]
            and len(loaded["cuda_rng_states"]) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(loaded["cuda_rng_states"])
    else:
        mean = training_mean(train_files, config)
        initial_validation = evaluate(model, val_files, config, mean, device)

    output.mkdir(parents=True, exist_ok=True)
    _json_write(output / "config.json", asdict(config))
    _json_write(
        output / "data_manifest.json",
        {
            **manifest,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "normalization": "per-row center then L2 norm (epsilon 1e-6)",
            "initial_validation": initial_validation,
        },
    )
    print(
        f"Training on {device}; {sum(m['rows'] for m in manifest['train']):,} training vectors, {sum(m['rows'] for m in manifest['validation']):,} validation vectors",
        flush=True,
    )
    print(
        f"Initial validation reconstruction MSE per vector: {initial_validation['reconstruction_mse']:.6f}",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_samples = 0
        loss_sum = reconstruction_sum = auxiliary_sum = 0.0
        for x in iter_batches(
            train_files,
            config.d_in,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            x = x.to(device)
            target = normalize_targets(x)
            optimizer.zero_grad(set_to_none=True)
            recon, code, aux_recon = model(x)
            loss, reconstruction, auxiliary = compute_losses(
                target,
                recon,
                aux_recon,
                has_dead=bool(model.dead_mask.any()),
                aux_weight=config.aux_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite loss at training step {global_step}"
                )
            loss.backward()
            _project_decoder_grads_orthogonal(model)
            # Detect nonfinite gradients without changing their magnitudes.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            _renorm_decoder_columns_(model)
            model.update_dead_mask(code.detach(), len(x))
            epoch_samples += len(x)
            samples_seen += len(x)
            global_step += 1
            loss_sum += loss.item() * len(x)
            reconstruction_sum += reconstruction.item() * len(x)
            auxiliary_sum += auxiliary.item() * len(x)
            if global_step % config.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} samples={samples_seen:,} recon={reconstruction.item():.6f} aux={auxiliary.item():.6f}",
                    flush=True,
                )

        validation = evaluate(model, val_files, config, mean, device)
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "train_loss": loss_sum / epoch_samples,
            "train_reconstruction_mse": reconstruction_sum / epoch_samples,
            "train_auxiliary_mse": auxiliary_sum / epoch_samples,
            **{f"val_{key}": value for key, value in validation.items()},
            "dead_feature_fraction": model.dead_mask.float().mean().item(),
        }
        history.append(row)
        improved = validation["reconstruction_mse"] < best_mse
        best_mse = min(best_mse, validation["reconstruction_mse"])
        checkpoint = {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "data_manifest": manifest,
            "epoch": epoch + 1,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_val_mse": best_mse,
            "training_mean": mean.cpu(),
            "initial_validation": initial_validation,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else [],
        }
        if improved:
            _checkpoint_write(output / "best.pt", checkpoint)
        _checkpoint_write(output / "last.pt", checkpoint)
        _history_write(output / "history.csv", history)
        print(
            f"Epoch {epoch + 1}: val MSE={validation['reconstruction_mse']:.6f}; mean active features={validation['mean_l0']:.2f}; best={best_mse:.6f}",
            flush=True,
        )

    return {
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "history": history,
        "initial_validation": initial_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir",
        action="append",
        help="Dated activation directory; repeat for multiple directories",
    )
    source.add_argument(
        "--train-dir",
        action="append",
        help="Explicit training directory; also supply --val-dir",
    )
    parser.add_argument(
        "--val-dir", action="append", help="Explicit independent validation directory"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of latest calendar days to hold out",
    )
    parser.add_argument("--prefix", default="layer0008_")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume", help="Resume last.pt; --epochs is the total desired epoch count"
    )
    defaults = TrainConfig(output_dir="unused")
    for name in (
        "d_in",
        "latent",
        "k",
        "k_aux",
        "dead_window",
        "batch_size",
        "epochs",
        "seed",
        "log_every",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in ("lr", "aux_weight"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.data_dir:
        if args.val_dir:
            parser.error(
                "Use either --data-dir with a date split, or --train-dir and --val-dir"
            )
        train_files, val_files = split_by_day(
            find_activation_files(args.data_dir, args.prefix), args.val_fraction
        )
    else:
        if not args.val_dir:
            parser.error("--train-dir requires --val-dir")
        train_files = find_activation_files(args.train_dir, args.prefix)
        val_files = find_activation_files(args.val_dir, args.prefix)
    config = TrainConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    result = train_topk(train_files, val_files, config, resume=args.resume)
    print(f"Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
