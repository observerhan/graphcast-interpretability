"""CPU integration checks for KAN training/resume and signed cyclone probing."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from cyclone_data import sha256
from cyclone_probes import (DenseFeatureMatrix, best_cutoff, encode_split, evaluate_selected,
                            fit_probes, open_experiment, release_features, select_feature)
from figure4_storms import feature_color_limits
from graphcast_interpretability.kan_sae import KANSAE
from sae_checkpoint import feature_codes, load_model
from train_kan_sae import KANTrainer, TrainingConfig, prepare_data


def training_fixture(root):
    activations = root / "activations"
    activations.mkdir()
    dates = ["2017-01-15", "2017-02-15", "2017-02-28"]
    names = [f"layer0008_t{date}T12.npy" for date in dates]
    rng = np.random.default_rng(691)
    for i, name in enumerate(names):
        np.save(activations / name, (rng.normal(size=(24, 4)) + i).astype(np.float32))
    source = dict(layer=8, shape=[24, 4], dtype="float32", graphcast_commit="test", checkpoint="test")
    manifest = dict(source=source, requested_centers=[d + "T12" for d in dates], validation_dates=dates[-1:])
    (activations / "dataset.json").write_text(json.dumps(manifest))
    split = dict(source=source, training_files=names[:2], validation_files=names[2:],
                 training_dates=dates[:2], validation_dates=dates[2:],
                 data_signature=[dict(file=n, sha256=sha256(activations / n)) for n in names])
    reference = root / "topk_split.json"
    reference.write_text(json.dumps(split))
    return activations, reference


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_training_resume_matches_uninterrupted_and_repairs_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activations, reference = training_fixture(root)
            data = prepare_data(activations, reference, root / "cache")
            cfg = TrainingConfig(latent=8, batch_size=12, calibration_samples=13,
                                 learning_rate=0.01, l1_ramp_epochs=3)
            full = KANTrainer(data, root / "full", cfg, device="cpu")
            full.fit(3)
            partial = KANTrainer(data, root / "resume", cfg, device="cpu")
            partial.fit(1)
            # A disconnect after last.pt can leave derived outputs missing/stale.
            (root / "resume" / "best.pt").unlink()
            (root / "resume" / "history.csv").write_text("incomplete")
            with patch.object(KANSAE, "initialize_from_data", side_effect=AssertionError("must not recalibrate")):
                resumed = KANTrainer(data, root / "resume", cfg, device="cpu")
            self.assertTrue((root / "resume" / "best.pt").is_file())
            self.assertIn("epoch", (root / "resume" / "history.csv").read_text())
            resumed.fit(3)
            self.assertEqual(resumed.history, full.history)
            for key, value in full.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value, atol=0, rtol=0)
            self.assertEqual(resumed.best_epoch, full.best_epoch)
            self.assertEqual(resumed.calibration, full.calibration)
            self.assertTrue((resumed.calibration_indices < 48).all())  # training only
            self.assertEqual(len(torch.unique(resumed.calibration_indices)), 13)
            train = torch.from_numpy(np.concatenate([np.load(p) for p in data.train_files]))
            torch.testing.assert_close(resumed.train_mean, resumed.model.reconstruction_target(train).mean(0))
            summary = resumed.summarize()
            self.assertEqual(summary["best_epoch"], resumed.best_epoch)
            best = torch.load(root / "resume" / "best.pt", weights_only=True)
            model = load_model(best)
            self.assertTrue(model.grid_initialized)
            self.assertEqual(best["data_signature"], data.split["data_signature"])
            self.assertEqual(cfg.l1_at_epoch(3), cfg.l1_at_epoch(10))
            with self.assertRaisesRegex(ValueError, "settings changed"):
                KANTrainer(data, root / "resume", replace(cfg, learning_rate=0.02), device="cpu")

    def test_reference_data_corruption_and_split_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activations, reference = training_fixture(root)
            original = json.loads(reference.read_text())
            changed = dict(original, validation_files=original["training_files"][:1])
            reference.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                prepare_data(activations, reference, root / "cache")
            reference.write_text(json.dumps(original))
            path = activations / original["training_files"][0]
            array = np.load(path)
            array[0, 0] += 1
            np.save(path, array)
            with self.assertRaisesRegex(ValueError, "differs from the file used for TopK"):
                prepare_data(activations, reference, root / "cache")

    def test_signed_cutoffs_match_exhaustive_dense_reference(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            value = rng.integers(-3, 4, size=40).astype(np.float32)
            truth = rng.random(40) > 0.7
            truth[0], truth[1] = True, False
            for slope in [-1, 1]:
                candidates = []
                for cutoff in np.unique(value):
                    guessed = value >= cutoff if slope > 0 else value <= cutoff
                    tp, pred = int((guessed & truth).sum()), int(guessed.sum())
                    candidates.append((2 * tp / (truth.sum() + pred), -pred, float(cutoff), tp, pred))
                expected = max(candidates)
                for keep in [np.ones(len(value), dtype=bool), value != 0]:
                    cutoff, tp, pred = best_cutoff(value[keep], truth[keep], len(value), int(truth.sum()), slope)
                    self.assertEqual((cutoff, tp, pred), expected[2:])
        self.assertEqual(best_cutoff(np.array([], np.float32), np.array([], bool), 10, 3, -1), (0., 3, 10))

    def test_signed_sparse_and_disk_backed_probes_agree(self):
        rng = np.random.default_rng(9)
        y = rng.random(100) > 0.7
        values = rng.integers(-3, 4, size=(100, 4)).astype(np.float32)
        values[:, 0] = np.where(y, -2, 0)  # Perfect negative-valued cyclone feature.
        values[:, 1] = np.where(y, 2, 0)
        values[:, 3] = 0
        sparse_matrix = sparse.csc_matrix(values)
        with tempfile.TemporaryDirectory() as temporary:
            dense = DenseFeatureMatrix(values.shape, temporary)
            dense.array[:] = values
            weights = fit_probes(sparse_matrix, y)
            dense_weights = fit_probes(dense, y)
            np.testing.assert_allclose(weights, dense_weights, atol=1e-7, rtol=1e-5)
            ranking = select_feature(sparse_matrix, y, weights)
            dense_ranking = select_feature(dense, y, dense_weights)
            self.assertEqual(ranking[0]["feature"], 0)  # Feature-index tie break.
            self.assertEqual(ranking[0]["rule"], "<=")
            self.assertEqual(ranking[0]["activation_cutoff"], -2.)
            self.assertEqual(ranking[0]["f1"], 1.)
            for a, b in zip(ranking, dense_ranking):
                self.assertEqual((a["feature"], a["f1"], a["activation_cutoff"]),
                                 (b["feature"], b["f1"], b["activation_cutoff"]))
            a = evaluate_selected(sparse_matrix, y, ["a", "b"], ranking[0])
            b = evaluate_selected(dense, y, ["a", "b"], dense_ranking[0])
            self.assertEqual(a, b)
            flipped = fit_probes(sparse.csc_matrix(-values), y)
            np.testing.assert_allclose(flipped[:, 0], -weights[:, 0], atol=1e-8)
            np.testing.assert_allclose(flipped[:, 1], weights[:, 1], atol=1e-8)
            release_features(dense)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_kan_encoding_preserves_negative_values_on_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "activations_layer8").mkdir()
            x = np.random.default_rng(3).normal(size=(8, 4)).astype(np.float32)
            path = root / "activations_layer8" / "layer0008_t2019-02-15T12.npy"
            np.save(path, x)
            center = "2019-02-15T12"
            manifest = dict(source=dict(shape=[8, 4]), samples=[dict(center=center, split="fit")],
                            completed={center: sha256(path)})
            labels = {center: np.array([0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)}
            model = KANSAE(4, 6)
            model.initialize_from_data(torch.from_numpy(x))
            with torch.no_grad():
                model.spline.coefficients.normal_(-1, 0.1)
            X, y, centers = encode_split("fit", root, manifest, labels, model, "cpu", batch_size=3,
                                         cache_dir=root / "cache")
            self.assertIsInstance(X, DenseFeatureMatrix)
            np.testing.assert_allclose(np.array(X.array), model.encode(torch.from_numpy(x)).detach().numpy())
            self.assertTrue((X.array < 0).all())
            self.assertEqual(centers, [center])
            np.testing.assert_array_equal(y, labels[center])
            release_features(X)
            self.assertEqual(list((root / "cache").iterdir()), [])
            selected, _, _ = encode_split("fit", root, manifest, labels, model, "cpu", selected_feature=2)
            self.assertTrue(sparse.issparse(selected))
            self.assertEqual(selected.nnz, 8)
            np.testing.assert_allclose(selected.getcol(2).toarray().ravel(),
                                       model.encode(torch.from_numpy(x)).detach().numpy()[:, 2])

    def test_open_experiment_preserves_legacy_identity_and_validates_kan_split(self):
        from graphcast_interpretability.model import SAE
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (root / "course").mkdir()
            run = root / "kan_baseline"
            run.mkdir()
            source = dict(shape=[8, 4], layer=8)
            samples = [dict(center=f"{year}-02-15T12", split=role)
                       for year, role in zip([2019, 2020, 2021], ["fit", "validation", "test"])]
            labels = {r["center"]: np.array([0, 1] * 4, bool) for r in samples}
            for folder in [root, data]:
                (folder / "mesh_labels.npz").write_bytes(b"label-checksum-fixture")
            manifest = dict(source=source, samples=samples, time_convention="last_input_utc",
                            completed={r["center"]: "checksum" for r in samples},
                            label_sha256=sha256(root / "mesh_labels.npz"))
            (root / "course" / "dataset.json").write_text(json.dumps(manifest))
            split = dict(source=source, training_dates=["2017-01-15"], validation_dates=["2018-01-15"],
                         data_signature=[dict(file="training.npy", sha256="test")])
            (run / "data_split.json").write_text(json.dumps(split))
            cfg = dict(d_in=4, latent=6, k=2, k_aux=3, dead_window=100)
            old = SAE(d_in=4, latent=6, k_active=2, k_aux=3, dead_window=100)
            saved = dict(config=cfg, model_state=old.state_dict(), epoch=2,
                         data_signature=split["data_signature"], validation_dates=split["validation_dates"])
            checkpoint = run / "best.pt"
            with patch("cyclone_probes.read_samples", return_value=samples), \
                 patch("cyclone_probes.read_labels", return_value=labels):
                torch.save(saved, checkpoint)
                loaded, _, _, _, identity = open_experiment(root, data, "course", checkpoint)
                self.assertEqual(set(identity), {"checkpoint_sha256", "checkpoint_epoch", "latent", "k",
                                                  "cohort", "dataset_manifest_sha256", "ridge"})
                self.assertEqual(identity["k"], 2)
                kan = KANSAE(4, 6)
                kan.initialize_from_data(torch.randn(10, 4))
                saved.update(model_type="kan_sae", config=kan.config(), model_state=kan.state_dict(), data_split=split)
                torch.save(saved, checkpoint)
                loaded, _, _, _, identity = open_experiment(root, data, "course", checkpoint)
                self.assertEqual(identity["model_type"], "kan_sae")
                self.assertIsNone(identity["k"])
                self.assertTrue(loaded.grid_initialized)
                saved["data_split"] = dict(split, training_dates=["2016-01-01"])
                torch.save(saved, checkpoint)
                with self.assertRaisesRegex(ValueError, "recorded split"):
                    open_experiment(root, data, "course", checkpoint)

    def test_negative_storm_features_are_encoded_and_reused(self):
        import figure4_storms as storms
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, out = root / "cache", root / "out"
            cache.mkdir()
            x = torch.randn(8, 4)
            model = KANSAE(4, 6)
            model.initialize_from_data(x)
            with torch.no_grad():
                model.spline.coefficients.fill_(-2.)
            source = dict(shape=[8, 4])
            identity = dict(model_type="kan_sae", checkpoint_sha256="fixture", latent=6)
            winner = dict(feature=2)
            times = storms.TIMES[:2]
            completed = {}
            for time in times:
                path = cache / (time + ".npy")
                np.save(path, x.numpy())
                completed[time] = dict(activation_path=str(path), activation_sha256=sha256(path))
            (cache / "dataset.json").write_text(json.dumps(dict(source=source, time_convention="last_input_utc",
                storms=list(storms.STORMS), completed=completed)))
            with patch.object(storms, "TIMES", times), \
                 patch.object(storms, "saved_selection", return_value=(dict(experiment=identity, validation=winner), root)), \
                 patch("cyclone_probes.open_experiment", return_value=(model, "cpu", dict(source=source), {}, identity)):
                metadata = storms.encode_storms(root, root, "course", root / "best.pt", cache, out)
                with np.load(metadata["encoded"][times[0]]["path"]) as saved:
                    np.testing.assert_allclose(saved["values"], -2.)
                with patch("sae_checkpoint.feature_codes", side_effect=AssertionError("cache should be reused")):
                    storms.encode_storms(root, root, "course", root / "best.pt", cache, out)

    def test_legacy_topk_and_kan_checkpoint_dispatch(self):
        from graphcast_interpretability.model import SAE
        cfg = dict(d_in=4, latent=6, k=2, k_aux=3, dead_window=100)
        old = SAE(d_in=4, latent=6, k_active=2, k_aux=3, dead_window=100)
        saved = dict(config=cfg, model_state=old.state_dict())
        loaded = load_model(saved)
        x = torch.randn(7, 4)
        torch.testing.assert_close(feature_codes(loaded, x), old(x)[1])
        self.assertTrue((feature_codes(loaded, x) >= 0).all())
        kan = KANSAE(4, 6)
        with self.assertRaisesRegex(ValueError, "no calibrated"):
            load_model(dict(model_type="kan_sae", config=kan.config(), model_state=kan.state_dict()))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_model(dict(saved, model_type="unknown"))

    def test_plot_limits_preserve_both_signs_and_topk_scale(self):
        fields = [np.array([[-3., 1.]]), np.array([[2., 0.]])]
        self.assertEqual(feature_color_limits(fields, "kan_sae"), (-3., 3.))
        self.assertEqual(feature_color_limits([np.abs(f) for f in fields], "topk"), (0., 3.))


if __name__ == "__main__":
    unittest.main()
