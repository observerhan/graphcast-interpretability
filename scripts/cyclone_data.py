"""Shared file and sample conventions for notebooks 03 and 04."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_samples(data_dir, cohort):
    if cohort not in {"course", "all_guarded"}:
        raise ValueError("Choose 'course' or 'all_guarded' in both notebooks.")
    with (data_dir / "schedule.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if cohort == "course":
        rows = [r for r in rows if r["course_sample"] == "True"]
    result = [dict(center=r["center"], split=r["split"],
                   positive_nodes=int(r["mesh_positive_nodes"])) for r in rows]
    centers = [r["center"] for r in result]
    if len(centers) != len(set(centers)) or {r["split"] for r in rows} != {"fit", "validation", "test"}:
        raise ValueError("The sample list must have distinct UTC times and all three splits.")
    ids = {role: set() for role in ("fit", "validation", "test")}
    for r in rows:
        ids[r["split"]].update(r["active_track_sids"].split())
    if ids["fit"] & ids["validation"] or ids["fit"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Active IBTrACS storm IDs overlap across splits.")
    return result


def read_labels(path, samples):
    with np.load(path, allow_pickle=False) as data:
        centers, labels = data["centers"].tolist(), data["labels"]
        if (labels.shape != (len(centers), 40962) or labels.dtype != np.bool_
                or len(centers) != len(set(centers))):
            raise ValueError("Unexpected label time axis, shape or dtype")
    by_time = dict(zip(centers, labels))
    for sample in samples:
        if int(by_time[sample["center"]].sum()) != sample["positive_nodes"]:
            raise ValueError(f"Label count differs at {sample['center']}")
    return by_time


def activation_path(root, center):
    return root / "activations_layer8" / f"layer0008_t{center}.npy"
