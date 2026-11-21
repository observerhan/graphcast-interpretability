"""CPU checks for ERA5 time alignment, grid loading, saving, and resuming.

Run: python -m unittest discover -s tests -v
These use synthetic arrays and mock the remote dataset and GraphCast runtime.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ams691_extract as extraction
import setup_graphcast


def graphcast_modules():
    package = ModuleType("graphcast")
    package.graphcast = SimpleNamespace(GENERATED_FORCING_VARS=("generated_forcing",))
    package.data_utils = SimpleNamespace()
    hooks = ModuleType("graphcast.deep_typed_graph_net")
    hooks.get_activation_manager = lambda: None
    return {"graphcast": package, "graphcast.deep_typed_graph_net": hooks,
            "jax": ModuleType("jax")}


class ExtractionTests(unittest.TestCase):
    def test_hook_patch_is_repeatable_and_rejects_unknown_source(self):
        modules = graphcast_modules()
        modules["jax"].__version__ = "0.4.38"
        modules["jax"].devices = lambda: [SimpleNamespace(platform="gpu")]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "deep_typed_graph_net.py"
            source.write_text('            tag = f"{self.name}/post_res/nodes/{nset}"\n'
                              '            jdebug.callback(\n')
            spec = SimpleNamespace(submodule_search_locations=[tmp])
            with patch.dict(sys.modules, modules), patch("importlib.util.find_spec", return_value=spec):
                setup_graphcast.patch_hooks_and_check_gpu()
                first = source.read_text()
                self.assertIn("if not am._should_save(tag, step_idx, nset):", first)
                setup_graphcast.patch_hooks_and_check_gpu()
                self.assertEqual(source.read_text(), first)
                source.write_text("unexpected upstream implementation\n")
                with self.assertRaisesRegex(RuntimeError, "Unexpected GraphCast hook"):
                    setup_graphcast.patch_hooks_and_check_gpu()

    def test_window_crosses_midnight_and_keeps_last_input_time(self):
        times = np.array(["2021-08-27T18", "2021-08-28T00", "2021-08-28T06"],
                         dtype="datetime64[ns]")
        source = xr.Dataset(
            {"temperature": (("time", "level"), np.arange(6).reshape(3, 2)),
             "land_mask": ((), 1.0)},
            coords={"time": times, "level": [500, 1000]},
        )
        actual = extraction.forecast_window(source, "2021-08-28T00", [1000])
        np.testing.assert_array_equal(actual.datetime.values, times[None, :])
        np.testing.assert_array_equal(actual.time.values,
                                      np.array([0, 6, 12], dtype="timedelta64[h]"))
        np.testing.assert_array_equal(actual.temperature.values, [[[1], [3], [5]]])
        self.assertEqual(actual.temperature.dims, ("batch", "time", "level"))
        self.assertEqual(actual.land_mask.dims, ())

    def test_era5_loader_orders_latitude_and_selects_required_variables(self):
        lat = np.arange(90, -90.25, -0.25)
        lon = np.arange(0, 360, 0.25)
        ds = xr.Dataset(
            {"10m_u_component_of_wind": (("latitude", "longitude"),
                 np.broadcast_to(lat[:, None], (len(lat), len(lon)))),
             "unused": ((), 1.0)},
            coords={"latitude": lat, "longitude": lon},
        )
        task = SimpleNamespace(input_variables=("10m_u_component_of_wind",),
                               target_variables=(), forcing_variables=("generated_forcing",))
        with patch.dict(sys.modules, graphcast_modules()), patch.object(xr, "open_zarr", return_value=ds):
            actual = extraction.open_weather(task)
        self.assertEqual(set(actual.data_vars), {"10m_u_component_of_wind"})
        np.testing.assert_array_equal(actual.lat.values, lat[::-1])
        self.assertEqual(float(actual["10m_u_component_of_wind"].sel(lat=-90, lon=270)), -90)
        np.testing.assert_array_equal(actual.lon.values, lon)

    def test_era5_loader_rejects_an_unexpected_grid(self):
        ds = xr.Dataset(coords={"latitude": [-90, 0, 90], "longitude": [0, 180]})
        task = SimpleNamespace(input_variables=(), target_variables=(), forcing_variables=())
        with patch.dict(sys.modules, graphcast_modules()), patch.object(xr, "open_zarr", return_value=ds):
            with self.assertRaisesRegex(ValueError, "quarter-degree"):
                extraction.open_weather(task)

    def test_writer_converts_values_and_never_overwrites_a_snapshot(self):
        class Manager:
            def __init__(self, **settings):
                self.current_time_str = "2021-08-28T00"

            def _should_save(self, tag, step, node_set):
                return step == 8 and node_set == "mesh_nodes"

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manager = Manager()
            extraction.install_writer(manager, directory, expected_shape=(2, 3))
            array = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float16)
            manager.save("post_res", array, step_idx=7, node_set="mesh_nodes")
            self.assertEqual(list(directory.iterdir()), [])
            manager.save("post_res", array, step_idx=8, node_set="mesh_nodes")
            path = directory / "layer0008_t2021-08-28T00.npy"
            saved = np.load(path, allow_pickle=False)
            self.assertEqual(saved.dtype, np.float32)
            np.testing.assert_array_equal(saved, array[:, 0, :])
            self.assertEqual(extraction.check_array(path, (2, 3)), (2, 3))
            with self.assertRaises(FileExistsError):
                manager.save("post_res", array, step_idx=8, node_set="mesh_nodes")
            self.assertEqual(list(directory.iterdir()), [path])

    def test_completed_monthly_run_skips_model_and_weather_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "activations"
            out.mkdir()
            dates = ["2017-01-15", "2017-01-29"]
            for day in dates:
                np.save(out / f"layer0008_t{day}T12.npy", np.zeros((2, 3), dtype=np.float32))
            config = root / "settings.json"
            config.write_text(json.dumps(dict(activation_dir=str(out), dates=dates,
                                             validation_dates=[dates[-1]])))
            check_array = extraction.check_array
            with patch.dict(sys.modules, graphcast_modules()), \
                 patch.object(extraction, "check_array", side_effect=lambda p: check_array(p, (2, 3))), \
                 patch.object(extraction, "load_model") as load_model, \
                 patch.object(extraction, "open_weather") as open_weather:
                with self.assertRaisesRegex(ValueError, "unknown origin"):
                    extraction.main(config)
                source = dict(graphcast_commit=extraction.GRAPHCAST_COMMIT,
                              checkpoint=extraction.MODEL_NAME,
                              weather_store=extraction.WEATHER_STORE,
                              layer=8, node_set="mesh_nodes", mode="post_res",
                              dtype="float32", shape=[40962, 512])
                (out / "dataset.json").write_text(json.dumps({"source": source}))
                extraction.main(config)
                extraction.main(config)
            load_model.assert_not_called()
            open_weather.assert_not_called()
            manifest = json.loads((out / "dataset.json").read_text())
            self.assertEqual(manifest["completed_centers"], [day + "T12" for day in dates])
            self.assertEqual(manifest["validation_dates"], [dates[-1]])
            self.assertEqual(manifest["source"]["weather_store"], extraction.WEATHER_STORE)


if __name__ == "__main__":
    unittest.main()
