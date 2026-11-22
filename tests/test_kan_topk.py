"""Behavioral checks for signed sparsity, warm-up, recovery, resume and probing."""
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
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]
from cyclone_data import sha256
from cyclone_probes import encode_split, open_experiment
from figure4_storms import feature_color_limits
from graphcast_interpretability.kan_topk_sae import KANTopKSAE, magnitude_topk
from sae_checkpoint import feature_codes, load_model, model_type
from train_kan_topk_sae import KANTopKTrainer, TopKTrainingConfig, prepare_data
from test_kan_workflow import training_fixture


class KANTopKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_signed_magnitude_selection_and_gradients(self):
        values = torch.tensor([[-9., 5., -3., 0.], [1., -8., 7., -2.]], requires_grad=True)
        output = magnitude_topk(values, 2)
        expected = torch.tensor([[-9., 5., 0., 0.], [0., -8., 7., 0.]])
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(values.grad, (expected != 0).float())
        for k in [1, 2, 4]:
            tied = magnitude_topk(torch.ones(3, 4), k)
            self.assertTrue(((tied != 0).sum(1) == k).all())
        self.assertEqual(int(torch.count_nonzero(magnitude_topk(torch.zeros(3, 4), 2))), 0)
        for bad in [0, 5, True]:
            with self.assertRaises(ValueError):
                magnitude_topk(values, bad)

    def test_auxiliary_mask_never_leaks_forbidden_values_or_gradients(self):
        for mask in [torch.tensor([False, False, True, False]), torch.zeros(4, dtype=torch.bool)]:
            values = torch.tensor([[100., -90., -2., 70.]], requires_grad=True)
            result = magnitude_topk(values, 3, allowed=mask)
            expected = torch.where(mask, values.detach(), 0.)
            torch.testing.assert_close(result, expected)
            result.sum().backward()
            torch.testing.assert_close(values.grad, mask[None].float())

    def test_graphcast_dimensions_and_checkpoint_round_trip(self):
        torch.manual_seed(691)
        x = torch.randn(6, 512)
        model = KANTopKSAE(512, 4096, k_active=32)
        model.initialize_from_data(x)
        with torch.no_grad():
            model.spline.coefficients.normal_(std=0.01)
        reconstruction, code = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertTrue(((code != 0).sum(1) == 32).all())
        self.assertTrue((code < 0).any())
        self.assertTrue((code > 0).any())
        reconstruction.square().sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        saved = dict(model_type="kan_topk", config=model.config(), model_state=model.state_dict())
        restored = load_model(saved)
        self.assertEqual(model_type(restored), "kan_topk")
        torch.testing.assert_close(feature_codes(restored, x), code)
        self.assertEqual(feature_color_limits([np.array([-2., 1.])], "kan_topk"), (-2., 2.))

    def test_warmup_is_not_exported_and_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activations, reference = training_fixture(root)
            data = prepare_data(activations, reference, root / "cache")
            cfg = TopKTrainingConfig(latent=12, k_active=2, batch_size=12,
                calibration_samples=13, learning_rate=0.01, dead_window=25, k_aux=3)
            full = KANTopKTrainer(data, root / "full", cfg, device="cpu")
            full.fit(5)
            partial = KANTopKTrainer(data, root / "resume", cfg, device="cpu")
            partial.fit(1)
            self.assertIsNone(partial.best_epoch)
            self.assertFalse((root / "resume" / "best.pt").exists())
            with self.assertRaisesRegex(ValueError, "after warm-up"):
                partial.summarize()
            with self.assertRaisesRegex(ValueError, "warm-up"):
                load_model(torch.load(root / "resume" / "last.pt", weights_only=True))
            # The explicit dense warm-up trains more than the eventual k features.
            learned = partial.model.spline.coefficients.detach().abs().amax(1) > 0
            self.assertGreater(int(learned.sum()), cfg.k_active)
            self.assertTrue((partial.miss_counts == 0).all())
            with patch.object(KANTopKSAE, "initialize_from_data", side_effect=AssertionError("no recalibration")):
                resumed = KANTopKTrainer(data, root / "resume", cfg, device="cpu")
            resumed.fit(3)
            (root / "resume" / "best.pt").unlink()
            (root / "resume" / "history.csv").write_text("interrupted")
            resumed = KANTopKTrainer(data, root / "resume", cfg, device="cpu")
            self.assertTrue((root / "resume" / "best.pt").exists())
            self.assertIn("phase", (root / "resume" / "history.csv").read_text())
            resumed.fit(5)
            self.assertEqual(resumed.history, full.history)
            self.assertEqual(resumed.best_epoch, full.best_epoch)
            self.assertGreater(resumed.best_epoch, cfg.warmup_epochs)
            self.assertLess(full.best_error, full.history[0]["validation_error"])
            self.assertTrue(any(r["auxiliary_error"] > 0 for r in full.history if r["phase"] == "sparse"))
            torch.testing.assert_close(resumed.miss_counts, full.miss_counts, rtol=0, atol=0)
            for key, value in full.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value, atol=0, rtol=0)
            self.assertTrue(all(row["max_active_features"] <= cfg.k_active for row in full.history))
            self.assertTrue((resumed.calibration_indices < 48).all())
            self.assertEqual(len(torch.unique(resumed.calibration_indices)), 13)
            summary = resumed.summarize()
            self.assertEqual(summary["model_type"], "kan_topk")
            self.assertLessEqual(summary["max_active_features"], cfg.k_active)
            resumed.plot_history()
            self.assertTrue((root / "resume" / "training_curve.png").is_file())
            saved = torch.load(root / "resume" / "best.pt", weights_only=True)
            restored = load_model(saved)
            self.assertEqual(restored.k_active, 2)
            for changes in [dict(k_active=3), dict(warmup_epochs=2), dict(learning_rate=.02)]:
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    KANTopKTrainer(data, root / "resume", replace(cfg, **changes), device="cpu")
            with self.assertRaisesRegex(ValueError, "warmup_epochs"):
                replace(cfg, warmup_epochs=0).validate()

    def test_auxiliary_loss_gives_gradients_to_only_allowed_spline_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activations, reference = training_fixture(root)
            data = prepare_data(activations, reference, root / "cache")
            cfg = TopKTrainingConfig(latent=6, k_active=2, k_aux=4, dead_window=10,
                                     calibration_samples=10, batch_size=12)
            trainer = KANTopKTrainer(data, root / "run", cfg, device="cpu")
            with torch.no_grad():
                trainer.model.spline.coefficients.copy_(torch.tensor([5., -4., 3., -2., 1., -.5])[:, None])
                trainer.miss_counts[4:] = 10
            x = torch.from_numpy(np.load(data.train_files[0])[:12])
            _, _, auxiliary, code = trainer.batch_losses(x, warmup=False)
            self.assertTrue((code[:, 2:] == 0).all())
            self.assertGreater(float(auxiliary.detach()), 0)
            auxiliary.backward()
            gradient = trainer.model.spline.coefficients.grad.abs().sum(1)
            self.assertTrue((gradient[:4] == 0).all())
            self.assertTrue((gradient[4:] > 0).all())

    def test_signed_sparse_encoding_and_experiment_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activations, reference = training_fixture(root)
            data = prepare_data(activations, reference, root / "cache")
            cfg = TopKTrainingConfig(latent=6, k_active=2, batch_size=12,
                                     calibration_samples=12, learning_rate=0.01)
            trainer = KANTopKTrainer(data, root / "run", cfg, device="cpu")
            trainer.fit(2)
            # Export a controlled signed model through the real checkpoint loader.
            with torch.no_grad():
                trainer.model.spline.coefficients.copy_(torch.tensor([1., -2., 3., -4., 5., -6.])[:, None])
            saved = torch.load(root / "run" / "best.pt", weights_only=True)
            saved["model_state"] = trainer.model.state_dict()
            checkpoint = root / "run" / "best.pt"
            torch.save(saved, checkpoint)
            label_root, label_data = root / "labels", root / "label_data"
            (label_root / "course" / "activations_layer8").mkdir(parents=True)
            label_data.mkdir()
            labels = np.arange(24) % 3 == 0
            np.savez(label_data / "mesh_labels.npz", labels=labels)
            (label_root / "mesh_labels.npz").write_bytes((label_data / "mesh_labels.npz").read_bytes())
            sample = dict(center="2019-08-01T00", split="fit")
            # cyclone_data.activation_path uses the run root's activations_layer8.
            from cyclone_data import activation_path
            path = activation_path(label_root, sample["center"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.load(data.train_files[0]))
            manifest = dict(source=data.split["source"], samples=[sample], time_convention="last_input_utc",
                completed={sample["center"]: sha256(path)}, label_sha256=sha256(label_root / "mesh_labels.npz"))
            (label_root / "course" / "dataset.json").write_text(json.dumps(manifest))
            with patch("cyclone_probes.read_samples", return_value=[sample]), \
                 patch("cyclone_probes.read_labels", return_value={sample["center"]: labels}):
                model, device, manifest, by_time, identity = open_experiment(
                    label_root, label_data, "course", checkpoint)
            self.assertEqual(identity["model_type"], "kan_topk")
            self.assertEqual(identity["k"], 2)
            with patch("cyclone_probes.DenseFeatureMatrix", side_effect=AssertionError("must stay sparse")):
                X, y, centers = encode_split("fit", label_root, manifest, by_time, model, device)
            self.assertTrue(sparse.isspmatrix_csc(X))
            self.assertEqual(X.nnz, 24 * 2)
            self.assertTrue((X.getcol(5).toarray() < 0).all())
            np.testing.assert_allclose(X.toarray(), feature_codes(model, torch.from_numpy(np.load(path)).to(device)).detach().cpu().numpy())
            np.testing.assert_array_equal(y, labels)
            self.assertEqual(centers, [sample["center"]])


if __name__ == "__main__":
    unittest.main()
