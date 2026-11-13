"""Numerical and restart contracts for the TopK training workflow.

Run with python -m unittest discover -s tests -v (after installing .[training]).
"""

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from graphcast_interpretability.model import SAE
from graphcast_interpretability.train_topk import (
    TrainConfig,
    _validate_split,
    compute_losses,
    evaluate,
    find_activation_files,
    iter_batches,
    normalize_targets,
    split_by_day,
    train_topk,
)


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def array(self, name, data):
        path = self.root / name
        np.save(path, data)
        return path

    def corpus(self):
        rng = np.random.default_rng(7)
        basis = rng.normal(size=(3, 8)).astype(np.float32)

        def samples(n):
            return (
                rng.normal(size=(n, 3)) @ basis + 0.01 * rng.normal(size=(n, 8))
            ).astype(np.float32)

        train = self.array("layer0008_mesh_t2021-01-01T00.npy", samples(129))
        val = self.array("layer0008_mesh_t2021-01-03T00.npy", samples(65)[:, None, :])
        return [train], [val]

    def config(self, directory="run", epochs=5):
        return TrainConfig(
            output_dir=str(self.root / directory),
            d_in=8,
            latent=16,
            k=3,
            k_aux=4,
            batch_size=32,
            epochs=epochs,
            lr=0.01,
            dead_window=64,
            device="cpu",
            seed=11,
            log_every=1000,
        )

    def run_training(self, train, val, config, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_topk(train, val, config, **kwargs)

    def test_date_split_keeps_snapshots_together(self):
        for day in range(1, 6):
            for hour in ("00", "06"):
                self.array(
                    f"layer0008_mesh_t2021-01-{day:02d}T{hour}.npy",
                    np.ones((2, 8), np.float32),
                )
        train, val = split_by_day(find_activation_files(self.root), 0.2)
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertTrue(all("2021-01-05" in p.name for p in val))
        with self.assertRaisesRegex(ValueError, "two different dates"):
            split_by_day(val)
        with self.assertRaisesRegex(ValueError, "share dates"):
            _validate_split(val[:1], val[1:])
        with self.assertRaisesRegex(ValueError, "same file"):
            _validate_split(train, train)

    def test_stream_includes_every_partial_batch(self):
        a = self.array("a.npy", np.arange(40, dtype=np.float32).reshape(5, 8))
        b = self.array("b.npy", np.arange(56, dtype=np.float32).reshape(7, 1, 8))
        batches = list(iter_batches([a, b], 8, 4))
        self.assertEqual([len(x) for x in batches], [4, 1, 4, 3])
        expected = np.concatenate([np.load(a), np.load(b)[:, 0, :]])
        np.testing.assert_array_equal(torch.cat(batches).numpy(), expected)
        shuffled = torch.cat(list(iter_batches([a, b], 8, 4, shuffle=True, seed=5)))
        self.assertCountEqual(shuffled[:, 0].tolist(), expected[:, 0].tolist())

    def test_rejects_ambiguous_bytes_and_nonfinite_data(self):
        legacy = self.array("legacy.npy", np.zeros((2, 8), dtype="V2"))
        with self.assertRaisesRegex(ValueError, "must not be reinterpreted"):
            list(iter_batches([legacy], 8, 4))
        bad = self.array("nan.npy", np.full((2, 8), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            list(iter_batches([bad], 8, 4))

    def test_validation_scores_the_normalized_reconstruction_target(self):
        path = self.array(
            "x.npy", np.arange(32, dtype=np.float32).reshape(4, 8) * 7 + 100
        )

        class PerfectNormalizedReconstruction(torch.nn.Module):
            def forward(self, x):
                return (
                    normalize_targets(x),
                    x.new_zeros((len(x), 16)),
                    torch.zeros_like(x),
                )

        metrics = evaluate(
            PerfectNormalizedReconstruction(),
            [path],
            self.config(),
            torch.zeros(8),
            torch.device("cpu"),
        )
        self.assertEqual(metrics["reconstruction_mse"], 0)
        self.assertAlmostEqual(metrics["mean_baseline_mse"], 1, places=6)

    def test_auxiliary_residual_does_not_backpropagate_into_main_reconstruction(self):
        target = torch.ones(3, 8)
        recon = torch.zeros(3, 8, requires_grad=True)
        aux = torch.full((3, 8), 0.25, requires_grad=True)
        _, _, auxiliary = compute_losses(
            target, recon, aux, has_dead=True, aux_weight=1 / 32
        )
        main_gradient, aux_gradient = torch.autograd.grad(
            auxiliary, (recon, aux), allow_unused=True
        )
        self.assertIsNone(main_gradient)
        self.assertGreater(aux_gradient.abs().sum().item(), 0)
        total, reconstruction, auxiliary = compute_losses(
            target, recon, aux, has_dead=False, aux_weight=1 / 32
        )
        self.assertEqual(auxiliary.item(), 0)
        self.assertEqual(total.item(), reconstruction.item())

    def test_training_learns_and_checkpoints_keep_topk_contract(self):
        train, val = self.corpus()
        config = self.config()
        result = self.run_training(train, val, config)
        checkpoint = torch.load(
            result["best_checkpoint"], map_location="cpu", weights_only=True
        )
        self.assertLess(
            checkpoint["best_val_mse"],
            result["initial_validation"]["reconstruction_mse"],
        )
        self.assertEqual(
            checkpoint["best_val_mse"],
            min(row["val_reconstruction_mse"] for row in result["history"]),
        )
        self.assertEqual(result["history"][-1]["samples_seen"], 129 * config.epochs)
        self.assertEqual(result["history"][-1]["val_samples"], 65)
        model = SAE(8, 16, 3, 4, dead_window=64)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        torch.testing.assert_close(model.dec.weight.norm(dim=0), torch.ones(16))
        misses = model.miss_counts.clone()
        evaluate(model, val, config, checkpoint["training_mean"], torch.device("cpu"))
        torch.testing.assert_close(model.miss_counts, misses)
        _, code, _ = model(torch.from_numpy(np.load(train[0])))
        self.assertTrue((code >= 0).all())
        self.assertTrue(((code > 0).sum(dim=1) <= config.k).all())

    def test_epoch_resume_matches_uninterrupted_training(self):
        train, val = self.corpus()
        full = self.run_training(train, val, self.config("full", epochs=4))
        short_config = self.config("resumed", epochs=2)
        short = self.run_training(train, val, short_config)
        resumed = self.run_training(
            train, val, replace(short_config, epochs=4), resume=short["last_checkpoint"]
        )
        full_state = torch.load(full["last_checkpoint"], weights_only=True)
        resumed_state = torch.load(resumed["last_checkpoint"], weights_only=True)
        self.assertEqual(full_state["history"], resumed_state["history"])
        for key in full_state["model_state"]:
            torch.testing.assert_close(
                full_state["model_state"][key],
                resumed_state["model_state"][key],
                rtol=0,
                atol=0,
            )

    def test_resume_rejects_changed_settings_and_existing_runs(self):
        train, val = self.corpus()
        config = self.config(epochs=1)
        result = self.run_training(train, val, config)
        with self.assertRaisesRegex(FileExistsError, "already contains a run"):
            self.run_training(train, val, config)
        with self.assertRaisesRegex(ValueError, "configuration mismatch: k"):
            self.run_training(
                train,
                val,
                replace(config, epochs=2, k=2),
                resume=result["last_checkpoint"],
            )


if __name__ == "__main__":
    unittest.main()
