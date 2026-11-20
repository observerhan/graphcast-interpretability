"""Run notebook 01's GraphCast configuration at the exact labeled UTC hours."""
import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import xarray as xr
from google.cloud import storage
from graphcast import checkpoint, data_utils, graphcast, icosahedral_mesh
from graphcast.deep_typed_graph_net import get_activation_manager

# Notebook 03 exports the original %writefile cell of the pinned notebook 01
# to this module in /content. Its functions preserve the original hook and numerics.
from ams691_extract import (GRAPHCAST_COMMIT, MODEL_NAME, WEATHER_STORE, check_array,
                            forecast_window, install_writer, make_forward)
from cyclone_data import activation_path, read_labels, read_samples, sha256, write_json


def load_model():
    bucket = storage.Client.create_anonymous_client().bucket("dm_graphcast")
    cache = Path("/content/ams691_graphcast_downloads")
    cache.mkdir(parents=True, exist_ok=True)

    def download(name):
        path = cache / Path(name).name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            bucket.blob("graphcast/" + name).download_to_filename(str(tmp))
            tmp.replace(path)
        return path

    with download("params/" + MODEL_NAME).open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    stats = {name: xr.load_dataset(download("stats/" + name + ".nc"))
             for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}
    return ckpt, stats


def open_weather(task):
    ds = xr.open_zarr(WEATHER_STORE, consolidated=True, storage_options={"token": "anon"})
    ds = ds.rename({x: y for x, y in [("latitude", "lat"), ("longitude", "lon")] if x in ds.coords})
    ds = ds.sortby("lat") if ds.lat.values[0] > ds.lat.values[-1] else ds
    if not (np.array_equal(ds.lat.values, np.arange(-90, 90.25, 0.25))
            and np.array_equal(ds.lon.values, np.arange(0, 360, 0.25))):
        raise ValueError("ERA5 does not have the expected global quarter-degree grid.")
    needed = set(task.input_variables + task.target_variables + task.forcing_variables)
    return ds[sorted(needed - set(graphcast.GENERATED_FORCING_VARS))]


def collect(root, data_dir, cohort):
    samples = read_samples(data_dir, cohort)
    provenance = json.loads((data_dir / "provenance.json").read_text())
    if GRAPHCAST_COMMIT != provenance["graphcast_commit"]:
        raise ValueError("GraphCast source differs from the source used to prepare the mesh.")
    labels = data_dir / "mesh_labels.npz"
    read_labels(labels, samples)
    directory = root / "activations_layer8"
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(graphcast_commit=GRAPHCAST_COMMIT, checkpoint=MODEL_NAME, weather_store=WEATHER_STORE,
                    layer=8, node_set="mesh_nodes", mode="post_res", dtype="float32", shape=[40962, 512])
    label_hash = sha256(labels)
    convention = dict(source=identity, time_convention="last_input_utc", label_sha256=label_hash)
    marker = directory / "source.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != convention:
            raise ValueError("These existing activations have a different source or label set.")
    elif any(directory.glob("*.npy")):
        raise ValueError("This folder already has snapshots with unknown provenance.")
    else:
        write_json(marker, convention)
    saved_labels = root / labels.name
    if saved_labels.exists() and sha256(saved_labels) != label_hash:
        raise ValueError("The saved mesh label bundle differs from the pinned release.")
    if not saved_labels.exists():
        tmp = saved_labels.with_suffix(".tmp")
        shutil.copyfile(labels, tmp)
        tmp.replace(saved_labels)

    run_dir = root / cohort
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "dataset.json"
    manifest = dict(**convention, cohort=cohort, samples=samples, completed={})
    known = {}
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        if any(prev[k] != manifest[k] for k in manifest if k != "completed"):
            raise ValueError("The selected cohort or model has changed since the last run.")
        known = prev["completed"]
    pending = []
    for row in samples:
        t = row["center"]
        if np.datetime64(t, "h").astype(str) != t or int(t[-2:]) not in (0, 6, 12, 18):
            raise ValueError(f"Expected an exact synoptic hour, got {t}")
        path = activation_path(root, t)
        if path.exists():
            check_array(path)
            digest = sha256(path)
            if t in known and known[t] != digest:
                raise ValueError(f"The saved activation changed at {t}.")
            manifest["completed"][t] = digest
        else:
            pending.append(t)
    write_json(manifest_path, manifest)
    if not pending:
        print(f"All {len(samples)} labeled snapshots are ready in {run_dir}.", flush=True)
        return
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Full GraphCast extraction requires the A100/H100 Colab setup from notebook 01.")

    ckpt, stats = load_model()
    cfg = ckpt.model_config
    if ((cfg.mesh_size, cfg.latent_size) != (6, 512)
            or not np.isclose(cfg.radius_query_fraction_edge_length, provenance["radius_fraction"],
                              rtol=0, atol=1e-8)):
        raise ValueError("The checkpoint geometry differs from the label projection.")
    vertices = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=cfg.mesh_size).vertices
    with np.load(saved_labels, allow_pickle=False) as z:
        np.testing.assert_allclose(vertices, z["vertices"], rtol=0, atol=1e-6)
    manager = get_activation_manager()
    install_writer(manager, directory)
    predict = jax.jit(make_forward(cfg, ckpt.task_config, stats).apply)
    with open_weather(ckpt.task_config) as source:
        for n, center in enumerate(pending, 1):
            print(f"Forecast {n}/{len(pending)} at {center} UTC; first call compiles.", flush=True)
            window = forecast_window(source, center, ckpt.task_config.pressure_levels)
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                window, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(ckpt.task_config))
            # For this nowcasting probe the label is at initialization, the last input.
            # Only a target template and forcings for the +6 h step enter the forward pass.
            manager.set_time(center)
            result = predict(ckpt.params, {}, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
            jax.block_until_ready(result)
            jax.effects_barrier()
            path = activation_path(root, center)
            check_array(path)
            manifest["completed"][center] = sha256(path)
            write_json(manifest_path, manifest)
            del result, window, inputs, targets, forcings
            gc.collect()
    print(f"Ready for notebook 04: {len(samples)} paired snapshots in {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("cohort", choices=["course", "all_guarded"])
    args = parser.parse_args()
    collect(args.root, args.data_dir, args.cohort)
