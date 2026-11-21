"""Encode selected TopK or signed KAN features and draw Figure 4 storm sequences."""
import json
import tempfile
from pathlib import Path

import numpy as np

from cyclone_data import sha256, write_json


STORMS = (
    dict(key="ida", name="Hurricane Ida", panel="B",
         extent=[-100, -70, 15, 40],
         times=["2021-08-28T00", "2021-08-28T12", "2021-08-29T00",
                "2021-08-29T12", "2021-08-30T00"]),
    dict(key="hagibis", name="Typhoon Hagibis", panel="C",
         extent=[120, 165, 5, 45],
         times=["2019-10-05T00", "2019-10-07T00", "2019-10-09T00",
                "2019-10-11T00", "2019-10-13T00"]),
)
TIMES = [time for storm in STORMS for time in storm["times"]]


def save_npz(path, **arrays):
    """Replace a complete NumPy archive atomically, including on Google Drive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def saved_selection(label_root, cohort, checkpoint):
    """Find notebook 04's saved choice using the actual checkpoint checksum."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"The selected SAE checkpoint is missing: {checkpoint}")
    digest = sha256(checkpoint)
    probe_dir = Path(label_root) / cohort / "probe_runs" / (checkpoint.parent.name + "_" + digest[:12])
    selection_path = probe_dir / "selected_feature.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Run notebook 04 with this checkpoint and cohort first: {selection_path}")
    selection = json.loads(selection_path.read_text())
    experiment = selection["experiment"]
    manifest_path = Path(label_root) / cohort / "dataset.json"
    if (experiment["checkpoint_sha256"] != digest or experiment["cohort"] != cohort
            or experiment["dataset_manifest_sha256"] != sha256(manifest_path)):
        raise ValueError("The saved probe choice belongs to a different experiment.")
    return selection, probe_dir


def encode_storms(label_root, data_dir, cohort, checkpoint, cache, out):
    """Use the saved model's own normalization and feature activation function."""
    import torch
    from cyclone_probes import open_experiment
    from sae_checkpoint import feature_codes, model_type

    selection, _ = saved_selection(label_root, cohort, checkpoint)
    model, device, manifest, _, identity = open_experiment(
        Path(label_root), Path(data_dir), cohort, Path(checkpoint))
    signed = model_type(model) == "kan_sae"
    if selection["experiment"] != identity:
        raise ValueError("Notebook 04's saved selection does not match this SAE run.")
    collected = json.loads((Path(cache) / "dataset.json").read_text())
    if (collected["source"] != manifest["source"]
            or collected["time_convention"] != "last_input_utc"
            or collected["storms"] != list(STORMS)
            or set(collected["completed"]) != set(TIMES)):
        raise ValueError("The storm collection is incomplete or uses a different GraphCast configuration.")
    feature = int(selection["validation"]["feature"])
    if not 0 <= feature < identity["latent"]:
        raise ValueError("The selected feature is outside this SAE's dictionary.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoded = {}
    for time in TIMES:
        record = collected["completed"][time]
        target = out / f"feature_{feature}_{time}.npz"
        if target.is_file():
            with np.load(target, allow_pickle=False) as saved:
                reusable = (saved["checkpoint_sha256"].item() == identity["checkpoint_sha256"]
                            and saved["activation_sha256"].item() == record["activation_sha256"]
                            and saved["feature_id"].item() == feature
                            and saved["values"].shape == (manifest["source"]["shape"][0],)
                            and np.all(np.isfinite(saved["values"]))
                            and (signed or np.all(saved["values"] >= 0)))
            if reusable:
                encoded[time] = dict(path=str(target), sha256=sha256(target))
                print(f"Feature already encoded for {time} UTC.", flush=True)
                continue
        activation = Path(record["activation_path"])
        if sha256(activation) != record["activation_sha256"]:
            raise ValueError(f"The saved GraphCast activation changed at {time}.")
        array = np.load(activation, mmap_mode="r", allow_pickle=False)
        if array.shape != tuple(manifest["source"]["shape"]) or array.dtype != np.float32:
            raise ValueError(f"Unexpected GraphCast activation shape or dtype at {time}.")
        values = np.empty(len(array), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(array), 256):
                batch = torch.from_numpy(np.array(array[start:start + 256], copy=True)).to(device)
                code = feature_codes(model, batch)
                values[start:start + len(batch)] = code[:, feature].cpu().numpy()
        if not np.all(np.isfinite(values)) or (not signed and np.any(values < 0)):
            raise ValueError(f"The SAE produced invalid feature values at {time}.")
        save_npz(target, values=values, feature_id=feature,
                 checkpoint_sha256=identity["checkpoint_sha256"],
                 activation_sha256=record["activation_sha256"])
        encoded[time] = dict(path=str(target), sha256=sha256(target))
        print(f"Encoded feature {feature} at {time} UTC. Maximum {values.max():.6g}.", flush=True)
    metadata = dict(experiment=identity, selection=selection["validation"],
                    time_convention="last_input_utc", wind_source="ERA5 at initialization",
                    storms=list(STORMS), encoded=encoded,
                    collection_manifest_sha256=sha256(Path(cache) / "dataset.json"))
    write_json(out / "figure4_metadata.json", metadata)
    return metadata


def feature_color_limits(features, kind):
    """One scale across each storm's five dates; signed KAN maps center on zero."""
    if kind == "kan_sae":
        limit = max(max(float(np.max(np.abs(field))) for field in features), 1e-6)
        return -limit, limit
    return 0., max(max(float(np.max(field)) for field in features), 1e-6)


def draw_storms(cache, out):
    """Plot feature values and ERA5 wind speed with shared scales per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    import cartopy.crs as ccrs

    cache, out = Path(cache), Path(out)
    metadata = json.loads((out / "figure4_metadata.json").read_text())
    if sha256(cache / "dataset.json") != metadata["collection_manifest_sha256"]:
        raise ValueError("The storm collection changed after feature encoding.")
    collected = json.loads((cache / "dataset.json").read_text())
    feature = int(metadata["selection"]["feature"])
    kind = metadata["experiment"].get("model_type", "topk")
    model_label = "KAN-SAE" if kind == "kan_sae" else "TopK SAE"
    colors = plt.get_cmap("YlGnBu")(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    feature_cmap = plt.get_cmap("RdBu_r") if kind == "kan_sae" else ListedColormap(colors)
    outputs, scales = [], {}
    for storm in STORMS:
        mapping_record = collected["mappings"][storm["key"]]
        if sha256(mapping_record["path"]) != mapping_record["sha256"]:
            raise ValueError(f"The plotting mesh mapping changed for {storm['name']}.")
        with np.load(mapping_record["path"], allow_pickle=False) as mapping:
            lat, lon, triplets = mapping["lat"], mapping["lon"], mapping["mesh_triplets"]
        features, winds = [], []
        for time in storm["times"]:
            record = metadata["encoded"][time]
            if sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"The encoded feature changed at {time}.")
            with np.load(record["path"], allow_pickle=False) as saved:
                grid = saved["values"][triplets].mean(axis=1).reshape(len(lat), len(lon))
                features.append(grid)
            wind = collected["completed"][time]
            if sha256(wind["wind_path"]) != wind["wind_sha256"]:
                raise ValueError(f"The wind data changed at {time}.")
            with np.load(wind["wind_path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["lat"], lat)
                np.testing.assert_array_equal(saved["lon"], lon)
                winds.append(np.hypot(saved["u10"], saved["v10"]))
        feature_max = max(float(field.max()) for field in features)
        wind_max = max(float(field.max()) for field in winds)
        feature_norm = Normalize(*feature_color_limits(features, kind))
        wind_norm = Normalize(0, max(5., 5. * np.ceil(wind_max / 5.)))
        fig, axes = plt.subplots(2, 5, figsize=(18, 6.6), constrained_layout=True,
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        for index, time in enumerate(storm["times"]):
            for row, field, cmap, norm in ((0, features[index], feature_cmap, feature_norm),
                                            (1, winds[index], "Greens", wind_norm)):
                ax = axes[row, index]
                ax.set_extent(storm["extent"], crs=ccrs.PlateCarree())
                mesh = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                                     cmap=cmap, norm=norm, shading="auto", rasterized=True)
                ax.coastlines(resolution="50m", linewidth=0.6)
                ax.set_title(time.replace("T", " ") + ":00", fontsize=10)
                if index == 0:
                    colorbar = fig.colorbar(mesh, ax=axes[row].tolist(), shrink=0.86,
                                           fraction=0.022, pad=0.015)
                    label = f"Feature {feature} activation"
                    if row == 1:
                        label = "ERA5 10 m wind speed (m/s)"
                    colorbar.set_label(label, fontsize=10)
        fig.suptitle(f"{storm['panel']}   {storm['name']} — {model_label} feature {feature}\n"
                     "UTC initialization times and ERA5 winds at the same times", fontsize=14)
        stem = out / f"figure4{storm['panel'].lower()}_{storm['key']}"
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
        scales[storm["key"]] = dict(feature_vmin=float(feature_norm.vmin), feature_vmax=float(feature_norm.vmax),
                                    feature_cmap=feature_cmap.name,
                                    wind_vmin=0., wind_vmax=float(wind_norm.vmax))
        print(f"Saved {storm['name']} panels. Feature maximum {feature_max:.6g}.", flush=True)
    metadata["plot_scales"] = scales
    metadata["mesh_to_grid"] = "mean of the containing triangle's three vertex values"
    write_json(out / "figure4_metadata.json", metadata)
    return outputs
