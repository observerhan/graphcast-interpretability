"""Shared GraphCast configuration, ERA5 loading, and layer-8 activation extraction.

Notebooks 01, 03, and 05 import this version-controlled module.
Run monthly extraction with: python scripts/ams691_extract.py SETTINGS.json
"""
import dataclasses
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


GRAPHCAST_COMMIT = "39d1de436148c5658726b5a092bd2d7ef8701f2a"
MODEL_NAME = (
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 "
    "- mesh 2to6 - precipitation input and output.npz"
)
WEATHER_STORE = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr"


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_array(path, expected_shape=(40962, 512)):
    import numpy as np
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError(f"{path.name}: expected float32 {expected_shape}, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name} contains invalid numbers. Remove this file and rerun extraction.")
    return array.shape


def forecast_window(source, center, pressure_levels):
    import numpy as np
    # Read the two input times and the next six-hour target/forcing time.
    # This also works when the window crosses midnight.
    times = np.datetime64(center, "ns") + np.array([-6, 0, 6], dtype="timedelta64[h]")
    window = source.sel(time=times, level=list(pressure_levels)).load()
    for name in window.data_vars:
        if "time" in window[name].dims:
            window[name] = window[name].expand_dims(batch=[0])
    return window.assign_coords(
        time=times - times[0], datetime=(("batch", "time"), times[None, :])
    )


def install_writer(manager, directory, expected_shape=(40962, 512)):
    import numpy as np
    manager.__init__(enabled=True, save_dir=str(directory), save_steps=[8],
                     save_node_sets=["mesh_nodes"], mode="post_res")

    def save_float32(tag, x, *, step_idx=None, node_set=None, time_str=None):
        if not manager._should_save(tag, step_idx, node_set):
            return
        timestamp = time_str or manager.current_time_str
        if timestamp is None:
            raise ValueError("The forecast time has not been set.")
        # Numerically convert bfloat16; do not reinterpret its bytes as float16.
        array = np.asarray(x).astype(np.float32)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(f"Unexpected activation values or shape: {array.shape}")
        path = directory / f"layer0008_t{timestamp}.npy"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing snapshot: {path}")
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
        print(f"Saved to Drive: {path.name}  {array.shape}", flush=True)

    manager.save = save_float32


def make_forward(model_config, task_config, stats):
    import haiku as hk
    from graphcast import autoregressive, casting, graphcast, normalization

    @hk.transform_with_state
    def forward(inputs, targets_template, forcings):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)
    return forward


def load_model():
    """Load the original GraphCast checkpoint and normalization statistics."""
    import xarray as xr
    from google.cloud import storage
    from graphcast import checkpoint, graphcast

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
    if ckpt.model_config.latent_size != 512 or ckpt.model_config.mesh_size != 6:
        raise ValueError("Downloaded model does not match the intended GraphCast configuration.")
    stats = {name: xr.load_dataset(download("stats/" + name + ".nc"))
             for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}
    return ckpt, stats


def open_weather(task):
    """Open public ERA5 lazily and standardize its coordinates for GraphCast."""
    import numpy as np
    import xarray as xr
    from graphcast import graphcast

    ds = xr.open_zarr(WEATHER_STORE, consolidated=True, storage_options={"token": "anon"})
    ds = ds.rename({x: y for x, y in [("latitude", "lat"), ("longitude", "lon")] if x in ds.coords})
    ds = ds.sortby("lat") if ds.lat.values[0] > ds.lat.values[-1] else ds
    if not (np.array_equal(ds.lat.values, np.arange(-90, 90.25, 0.25))
            and np.array_equal(ds.lon.values, np.arange(0, 360, 0.25))):
        raise ValueError("ERA5 does not have the expected global quarter-degree grid.")
    needed = set(task.input_variables + task.target_variables + task.forcing_variables)
    needed -= set(graphcast.GENERATED_FORCING_VARS)
    missing = needed - set(ds.data_vars)
    if missing:
        raise ValueError(f"Weather dataset is missing {sorted(missing)}")
    return ds[sorted(needed)]


def main(config_path):
    import numpy as np
    import jax
    from graphcast import data_utils
    from graphcast.deep_typed_graph_net import get_activation_manager

    config = json.loads(Path(config_path).read_text())
    directory = Path(config["activation_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    centers = [f"{day}T12" for day in config["dates"]]
    if len(set(config["dates"])) < 2 or len(set(centers)) != len(centers):
        raise ValueError("Choose at least two different dates, with no duplicates.")
    validation_dates = config.get("validation_dates", [max(config["dates"])])
    if not validation_dates or not set(validation_dates) < set(config["dates"]):
        raise ValueError("Choose validation dates from the schedule and keep at least one date for training.")
    for center in centers:
        if not np.datetime64("1959-01-01") < np.datetime64(center) < np.datetime64("2023-01-01"):
            raise ValueError("Choose dates inside the public weather dataset (1959–2022).")

    identity = dict(graphcast_commit=GRAPHCAST_COMMIT, checkpoint=MODEL_NAME,
                    weather_store=WEATHER_STORE, layer=8, node_set="mesh_nodes",
                    mode="post_res", dtype="float32", shape=[40962, 512])
    manifest_path = directory / "dataset.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("source") != identity:
            raise ValueError("This folder contains a different dataset. Choose a new PROJECT_FOLDER.")
    elif list(directory.glob("layer0008_*.npy")):
        raise ValueError("This folder already contains activations of unknown origin. Choose a new PROJECT_FOLDER.")
    manifest = dict(source=identity, requested_centers=centers,
                    validation_dates=validation_dates, completed_centers=[])
    pending = []
    for center in centers:
        path = directory / f"layer0008_t{center}.npy"
        if path.exists():
            check_array(path)
            manifest["completed_centers"].append(center)
            print(f"Already saved; skipping {center}", flush=True)
        else:
            pending.append(center)
    write_json(manifest_path, manifest)
    if not pending:
        print("All requested activations are already in Drive. Open notebook 2.", flush=True)
        return

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GraphCast cannot see a GPU. Select an A100 or H100 runtime and rerun this notebook.")
    print("GPU ready:", jax.devices(), flush=True)

    ckpt, stats = load_model()
    print("Opening the public weather dataset...", flush=True)
    source = open_weather(ckpt.task_config)

    manager = get_activation_manager()
    install_writer(manager, directory)
    forward = make_forward(ckpt.model_config, ckpt.task_config, stats)
    # Passing weights as arguments avoids embedding a large constant in the executable.
    predict = jax.jit(forward.apply)

    for number, center in enumerate(pending, 1):
        print(f"\nSnapshot {number}/{len(pending)}: {center} UTC", flush=True)
        print("Reading three weather times (this can take several minutes)...", flush=True)
        window = forecast_window(source, center, ckpt.task_config.pressure_levels)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            window, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(ckpt.task_config)
        )
        manager.set_time(center)
        print("Running GraphCast; the first forecast also compiles the model...", flush=True)
        result = predict(ckpt.params, {}, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
        jax.block_until_ready(result)
        # Finish every saving callback before changing its timestamp.
        jax.effects_barrier()
        check_array(directory / f"layer0008_t{center}.npy")
        manifest["completed_centers"].append(center)
        write_json(manifest_path, manifest)
        del result, window, inputs, targets, forcings
        gc.collect()

    source.close()
    print(f"\nFinished: {len(centers)} snapshots are saved in {directory}", flush=True)
    print("You can disconnect this runtime and open notebook 2.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])