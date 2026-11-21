"""Collect ten independent GraphCast runs and matching regional ERA5 winds."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from cyclone_data import activation_path, sha256, write_json
from figure4_storms import STORMS, TIMES, save_npz

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def collect(label_root, data_dir, cache):
    import dataclasses
    import jax
    from graphcast import data_utils, icosahedral_mesh
    from graphcast.deep_typed_graph_net import get_activation_manager
    from graphcast_interpretability.interpolate_utils import GridMeshMapper
    from collect_cyclone_activations import load_model, open_weather
    from ams691_extract import (GRAPHCAST_COMMIT, MODEL_NAME, WEATHER_STORE, check_array,
                                forecast_window, install_writer, make_forward)

    provenance = json.loads((data_dir / "provenance.json").read_text())
    source = dict(graphcast_commit=GRAPHCAST_COMMIT, checkpoint=MODEL_NAME,
                  weather_store=WEATHER_STORE, layer=8, node_set="mesh_nodes",
                  mode="post_res", dtype="float32", shape=[40962, 512])
    if GRAPHCAST_COMMIT != provenance["graphcast_commit"]:
        raise ValueError("GraphCast differs from the code used for the cyclone mesh labels.")
    original_source = json.loads((label_root / "activations_layer8" / "source.json").read_text())
    if original_source["source"] != source or original_source["time_convention"] != "last_input_utc":
        raise ValueError("Notebook 03 used a different GraphCast configuration or time convention.")
    manifest_path = cache / "dataset.json"
    manifest = dict(source=source, time_convention="last_input_utc", storms=list(STORMS),
                    completed={}, mappings={})
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        for key in ("source", "time_convention", "storms"):
            if previous[key] != manifest[key]:
                raise ValueError(f"The existing storm cache has different {key}. Choose a new cache folder.")
        manifest = previous
    directory = cache / "activations_layer8"
    directory.mkdir(parents=True, exist_ok=True)

    # Reuse the authors' containing-triangle interpolation on each regional grid.
    for storm in STORMS:
        path = cache / f"plot_grid_{storm['key']}.npz"
        if storm["key"] in manifest["mappings"]:
            if sha256(path) != manifest["mappings"][storm["key"]]["sha256"]:
                raise ValueError(f"The plotting grid changed for {storm['name']}.")
            continue
        west, east, south, north = storm["extent"]
        lat = np.arange(south, north + 0.125, 0.25)
        lon = np.arange(west, east + 0.125, 0.25)
        mapper = GridMeshMapper(lat, lon % 360, splits=6, cache_dir=cache / "mesh_mappings")
        mapping = mapper.precompute_mesh_to_grid()
        expected = np.repeat(np.arange(len(lat) * len(lon)), 3)
        np.testing.assert_array_equal(mapping["grid_idx"], expected)
        save_npz(path, lat=lat, lon=lon, mesh_triplets=mapping["mesh_idx"].reshape(-1, 3))
        manifest["mappings"][storm["key"]] = dict(path=str(path), sha256=sha256(path))
        write_json(manifest_path, manifest)

    pending = []
    for time in TIMES:
        record = manifest["completed"].get(time)
        if record is not None:
            if (sha256(record["activation_path"]) != record["activation_sha256"]
                    or sha256(record["wind_path"]) != record["wind_sha256"]):
                raise ValueError(f"A completed storm file changed at {time}. Restore it or use a new cache folder.")
            print(f"Already collected {time} UTC.", flush=True)
        else:
            pending.append(time)
    if not pending:
        print("All ten storm snapshots are ready.", flush=True)
        return
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Choose the A100/H100 runtime used for notebook 03.")
    ckpt, stats = load_model()
    cfg = ckpt.model_config
    if ((cfg.mesh_size, cfg.latent_size) != (6, 512)
            or not np.isclose(cfg.radius_query_fraction_edge_length, provenance["radius_fraction"],
                              rtol=0, atol=1e-8)):
        raise ValueError("The downloaded GraphCast model uses a different mesh.")
    vertices = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=6).vertices
    with np.load(data_dir / "mesh_labels.npz", allow_pickle=False) as labels:
        np.testing.assert_allclose(vertices, labels["vertices"], rtol=0, atol=1e-6)
    manager = get_activation_manager()
    install_writer(manager, directory)
    predict = jax.jit(make_forward(cfg, ckpt.task_config, stats).apply)
    with open_weather(ckpt.task_config) as weather:
        for number, time in enumerate(pending, 1):
            print(f"Storm snapshot {number}/{len(pending)} at {time} UTC.", flush=True)
            path = activation_path(cache, time)
            old_path = activation_path(label_root, time)
            if old_path.is_file():
                path = old_path
            if not path.is_file():
                window = forecast_window(weather, time, ckpt.task_config.pressure_levels)
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    window, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(ckpt.task_config))
                manager.set_time(time)
                result = predict(ckpt.params, {}, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
                jax.block_until_ready(result)
                jax.effects_barrier()
                del result, window, inputs, targets, forcings
                gc.collect()
            check_array(path)
            storm = next(item for item in STORMS if time in item["times"])
            with np.load(manifest["mappings"][storm["key"]]["path"], allow_pickle=False) as mapping:
                lat, lon = mapping["lat"], mapping["lon"]
            regional = weather[["10m_u_component_of_wind", "10m_v_component_of_wind"]].sel(
                time=np.datetime64(time), lat=lat, lon=lon % 360).load()
            u10 = regional["10m_u_component_of_wind"].transpose("lat", "lon").values.astype(np.float32)
            v10 = regional["10m_v_component_of_wind"].transpose("lat", "lon").values.astype(np.float32)
            if not np.all(np.isfinite(u10)) or not np.all(np.isfinite(v10)):
                raise ValueError(f"Non-finite ERA5 winds at {time}.")
            wind_path = cache / f"wind_{time}.npz"
            save_npz(wind_path, lat=lat, lon=lon, u10=u10, v10=v10, time=time)
            manifest["completed"][time] = dict(
                activation_path=str(path), activation_sha256=sha256(path),
                wind_path=str(wind_path), wind_sha256=sha256(wind_path))
            write_json(manifest_path, manifest)
            del regional, u10, v10
    print("All ten storm snapshots are ready for SAE encoding and plotting.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_root", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    # A local lock avoids relying on Google Drive's support for advisory locks.
    lock_key = hashlib.sha256(str(args.cache.resolve()).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"ams691_figure4_{lock_key}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another collector in this runtime is using the same storm cache.") from error
        collect(args.label_root, args.data_dir, args.cache)
