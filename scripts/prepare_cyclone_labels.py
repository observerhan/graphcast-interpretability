"""Rebuild the released 2019–2021 inventory, mesh labels and Colab schedule.

Use the sae-hooks GraphCast commit and dependencies given in notebook 03, plus
h5netcdf and pandas. Pass the two published archives as local files. Only the
small derived data products are versioned with this project.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import xarray as xr
from graphcast import grid_mesh_connectivity, icosahedral_mesh
from scipy import ndimage

TC_ARCHIVE = "https://dataverse.harvard.edu/api/access/datafile/11212451"
TC_MD5 = "83cc92c57da54ec1ec12e7d6fde08141"
TRACK_CSV = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
             "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")
GRAPHCAST_COMMIT = "39d1de436148c5658726b5a092bd2d7ef8701f2a"
RADIUS_FRACTION = 0.5999912857713345  # full 0.25-degree / 37-level checkpoint
YEARS = (2019, 2020, 2021)
ROLES = {2019: "fit", 2020: "validation", 2021: "test"}
# (TC-positive snapshots, all-background snapshots) in the course-sized run.
BUDGET = {"fit": (8, 4), "validation": (5, 3), "test": (5, 3)}


def components(mask):
    """8-connected regions on a latitude/longitude grid with a periodic seam."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count < 2 or not np.any(labels[:, 0]) or not np.any(labels[:, -1]):
        return labels, count
    parent = list(range(count + 1))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    left, right = labels[:, 0], labels[:, -1]
    for a, b in ((left, right), (left[1:], right[:-1]), (left[:-1], right[1:])):
        common = (a != 0) & (b != 0) & (a != b)
        for i, j in zip(a[common], b[common]):
            parent[root(i)] = root(j)
    roots = np.array([root(i) for i in range(count + 1)])
    _, dense = np.unique(roots, return_inverse=True)
    return dense[labels], int(dense.max())


def mesh_mapping(lat, lon):
    # This is the finest vertex order used by GraphCast._run_mesh_gnn.
    mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(splits=6)
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    length = np.linalg.norm(mesh.vertices[senders] - mesh.vertices[receivers], axis=1)
    radius = RADIUS_FRACTION * length.max()
    grid, node = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=lat, grid_longitude=lon, mesh=mesh, radius=radius)
    degree = np.bincount(node, minlength=len(mesh.vertices))
    if len(degree) != 40962 or np.any(degree == 0):
        raise ValueError("Unexpected connectivity on the full GraphCast mesh")
    return mesh.vertices, grid, node, degree


def load_tracks(path):
    cols = ["SID", "TRACK_TYPE", "ISO_TIME", "LAT", "LON"]
    tracks = pd.read_csv(path, usecols=cols, skiprows=[1])
    # Keep enough surrounding months to retain complete cross-year lifetimes.
    tracks = tracks[(tracks.TRACK_TYPE == "main") &
                    tracks.ISO_TIME.between("2018-10-01", "2022-04-01")].copy()
    tracks["time"] = pd.to_datetime(tracks.ISO_TIME, utc=True)
    return [(sid, group.sort_values("time")) for sid, group in tracks.groupby("SID")]


def track_centers(tracks, stamp):
    """List active SIDs and positions bracketed by observations <= 6 hours apart."""
    active, points = [], []
    for sid, group in tracks:
        times = group.time
        if not times.iloc[0] <= stamp <= times.iloc[-1]:
            continue
        active.append(sid)
        pos = times.searchsorted(stamp)
        if times.iloc[pos] == stamp:
            lat, lon = group.iloc[pos][["LAT", "LON"]]
        else:
            gap = times.iloc[pos] - times.iloc[pos - 1]
            if gap > pd.Timedelta(hours=6):
                continue
            f = (stamp - times.iloc[pos - 1]) / gap
            lat0, lon0 = group.iloc[pos - 1][["LAT", "LON"]]
            lat1, lon1 = group.iloc[pos][["LAT", "LON"]]
            lat = lat0 + f * (lat1 - lat0)
            lon = lon0 + f * ((lon1 - lon0 + 180) % 360 - 180)
        if np.isfinite(lat) and np.isfinite(lon):
            points.append((sid, lat, lon % 360))
    return active, points


def select_schedule(inventory):
    # Whole snapshots, with a two-month buffer around each year boundary.
    # Choose evenly spaced ranks within each stratum before observing SAE codes.
    schedule = inventory[inventory.six_hourly & inventory.month.between(2, 11)].copy()
    schedule["split"] = schedule.year.map(ROLES)
    schedule["course_sample"] = False
    active_by_split = {}
    for role, group in schedule.groupby("split"):
        active_by_split[role] = {sid for ids in group.active_track_sids for sid in ids.split()}
        # True empty maps are useful for measuring false alarms. A region that
        # disappears on projection is eligible for the full run but not this quota.
        for candidates, limit in zip((group[group.mesh_positive_nodes > 0],
                                      group[group.union_regions == 0]), BUDGET[role]):
            if len(candidates) < limit:
                raise ValueError(f"Too few eligible maps for {role}: {len(candidates)} < {limit}")
            indices = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
            schedule.loc[candidates.iloc[indices].index, "course_sample"] = True
    for i, role in enumerate(ROLES.values()):
        for other in list(ROLES.values())[i + 1:]:
            shared = active_by_split[role] & active_by_split[other]
            if shared:
                raise ValueError(f"Overlapping active IBTrACS IDs in {role} and {other}: {shared}")
    return schedule


def build(archive, track_file, out):
    with open(archive, "rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != TC_MD5:
            raise ValueError("Expected the published cleaned TC ZIP from Dataverse V2")
    out.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(track_file)
    rows, centers, mesh_masks = [], [], []
    grid = None
    with ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.startswith("TC_labels_cleaned/")
                         and n.endswith(".nc") and Path(n).name[:4] in map(str, YEARS))
        for member in members:
            stamp = pd.Timestamp(Path(member).stem, tz="UTC")
            key = stamp.strftime("%Y-%m-%dT%H")
            with xr.open_dataset(io.BytesIO(z.read(member)), engine="h5netcdf") as ds:
                if (pd.Timestamp(ds.time.values, tz="UTC") != stamp or
                        str(ds["class"].item()) != "Tropical Cyclone (TC)" or
                        not np.array_equal(ds.annotator.values, [0, 1])):
                    raise ValueError(f"Unexpected metadata in {member}")
                lat, lon = ds.latitude.values, ds.longitude.values
                if not (np.array_equal(lat, np.arange(90, -90.25, -0.25)) and
                        np.array_equal(lon, np.arange(0, 360, 0.25))):
                    raise ValueError(f"Unexpected grid in {member}")
                arr = ds.label.transpose("annotator", "latitude", "longitude").values
                if not np.all((arr == 0) | (arr == 1)):
                    raise ValueError(f"Non-binary or missing labels in {member}")
            union = (arr[0] | arr[1]).astype(bool)
            regions, n_union = components(union)
            active, points = track_centers(tracks, stamp)
            matched, occupied = [], set()
            for sid, latitude, longitude in points:
                # Half-grid ties go to the larger array index. Longitude wraps.
                iy = int(np.floor((90 - latitude) * 4 + 0.5))
                ix = int(np.floor(longitude * 4 + 0.5)) % len(lon)
                region = int(regions[iy, ix])
                if region:
                    matched.append(sid)
                    occupied.add(region)
            eligible = stamp.hour % 6 == 0
            mesh_count = -1  # Only exact six-hourly masks are projected.
            if eligible:
                if grid is None:
                    vertices, grid, node, degree = mesh_mapping(lat, lon)
                hits = np.bincount(node[union.ravel()[grid]], minlength=len(degree))
                mask = hits == degree  # mixed cells at the footprint boundary become 0
                mesh_count = int(mask.sum())
                centers.append(key)
                mesh_masks.append(mask)
            rows.append(dict(center=key, year=stamp.year, month=stamp.month,
                             six_hourly=eligible,
                             annotator_0_regions=components(arr[0])[1],
                             annotator_1_regions=components(arr[1])[1],
                             union_regions=n_union, center_matched_regions=len(occupied),
                             matched_track_sids=" ".join(matched), active_track_sids=" ".join(active),
                             mesh_positive_nodes=mesh_count))
    inv = pd.DataFrame(rows).sort_values("center", ignore_index=True)
    if inv.empty or not inv.center.is_unique:
        raise ValueError("Missing or repeated archive timestamps")
    schedule = select_schedule(inv)
    inv.to_csv(out / "mask_inventory.csv", index=False)
    schedule[["center", "split", "course_sample", "union_regions", "mesh_positive_nodes",
              "active_track_sids"]].to_csv(out / "schedule.csv", index=False)
    np.savez_compressed(out / "mesh_labels.npz", centers=np.array(centers),
                        labels=np.stack(mesh_masks), vertices=vertices)
    with open(track_file, "rb") as handle:
        track_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = dict(tc_archive=TC_ARCHIVE, tc_md5=TC_MD5,
                      ibtracs=TRACK_CSV, ibtracs_sha256=track_sha,
                      graphcast_commit=GRAPHCAST_COMMIT, mesh_splits=6, radius_fraction=RADIUS_FRACTION,
                      time_convention="last input (initialization), UTC",
                      grid_mask="union of the two cleaned annotator masks",
                      mesh_mask="1 iff all radius-linked grid cells are positive",
                      regions="8-connected, periodic longitude, per timestamp",
                      ibtracs_match="main track center falls inside a union region, nearest 0.25 degree cell; interpolation up to 6 h",
                      split="Feb–Nov in separate calendar years; all active main-track SIDs disjoint")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(inv.groupby("year").agg(snapshots=("center", "count"),
                                  positive=("union_regions", lambda v: (v > 0).sum()),
                                  footprints=("union_regions", "sum")))
    print(schedule[schedule.course_sample].groupby("split").mesh_positive_nodes.agg(["count", "sum"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_zip", type=Path, help=TC_ARCHIVE)
    parser.add_argument("track_csv", type=Path, help=TRACK_CSV)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.cleaned_zip, args.track_csv, args.output)
