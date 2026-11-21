"""Independent spline references, gradients, checkpointing, and CPU learning checks."""

import io
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphcast_interpretability.kan_sae import KANSAE, PerFeatureBSpline


class KANSAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(691)

    def test_spline_values_match_scipy_at_knots_endpoints_and_outside(self):
        for degree, n_basis in [(1, 4), (2, 5), (3, 9), (3, 4)]:
            with self.subTest(degree=degree, n_basis=n_basis):
                spline = PerFeatureBSpline(3, degree=degree, n_basis=n_basis).double()
                lower = torch.tensor([-3., 0.2, 10.], dtype=torch.float64)
                upper = torch.tensor([2., 4., 13.], dtype=torch.float64)
                spline.set_grid_(lower, upper)
                with torch.no_grad():
                    spline.coefficients.normal_()
                fractions = torch.linspace(-0.5, 1.5, 21, dtype=torch.float64)[:, None]
                x = lower + fractions * (upper - lower)
                x = torch.cat((x, spline.knots.T), dim=0)
                expected = np.stack([
                    BSpline(spline.knots[j].numpy(), spline.coefficients[j].detach().numpy(), degree)(
                        x[:, j].numpy().clip(lower[j].item(), upper[j].item())
                    ) for j in range(3)
                ], axis=1)
                np.testing.assert_allclose(spline(x).detach().numpy(), expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(spline(lower[None, :])[0], spline.coefficients[:, 0])
                torch.testing.assert_close(spline(upper[None, :])[0], spline.coefficients[:, -1])

    def test_spline_derivatives_and_coefficient_gradients_match_scipy(self):
        spline = PerFeatureBSpline(2).double()
        spline.set_grid_(torch.tensor([-2., 1.]), torch.tensor([3., 5.]))
        with torch.no_grad():
            spline.coefficients.normal_()
        x = torch.tensor([[-3., 0.], [-1.4, 1.23], [0.12, 3.13], [2.6, 4.8], [4., 6.]],
                         dtype=torch.float64, requires_grad=True)
        spline(x).sum().backward()
        for j in range(2):
            knots = spline.knots[j].numpy()
            values = x.detach().numpy()[:, j]
            lower, upper = knots[spline.degree], knots[spline.n_basis]
            clipped = values.clip(lower, upper)
            reference = BSpline(knots, spline.coefficients[j].detach().numpy(), spline.degree)
            expected_dx = reference.derivative()(clipped)
            expected_dx[(values < lower) | (values > upper)] = 0
            np.testing.assert_allclose(x.grad[:, j].numpy(), expected_dx, rtol=1e-12, atol=1e-12)
            expected_dc = BSpline(knots, np.eye(spline.n_basis), spline.degree)(clipped).sum(axis=0)
            np.testing.assert_allclose(spline.coefficients.grad[j].numpy(), expected_dc, atol=1e-12)
        h = x.detach()[1:-1].requires_grad_()
        coefficients = spline.coefficients.detach().clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda h, c: functional_call(spline, {"coefficients": c}, (h,)),
            (h, coefficients), eps=1e-6, atol=1e-5, rtol=1e-4,
        ))

    def test_constant_and_linear_splines_and_feature_independence(self):
        spline = PerFeatureBSpline(3).double()
        with torch.no_grad():
            spline.coefficients[0].fill_(1.7)
            spline.coefficients[1].fill_(-2.4)
            # Greville abscissae reproduce the identity function exactly.
            spline.coefficients[2].copy_(torch.stack([
                spline.knots[2, i + 1:i + spline.degree + 1].mean()
                for i in range(spline.n_basis)
            ]))
        x = torch.linspace(-1.3, 1.3, 23, dtype=torch.float64)[:, None].expand(-1, 3)
        expected = torch.stack((torch.full_like(x[:, 0], 1.7),
                                torch.full_like(x[:, 1], -2.4), x[:, 2].clamp(-1, 1)), dim=1)
        torch.testing.assert_close(spline(x), expected, rtol=1e-12, atol=1e-12)

    def test_calibration_uses_per_feature_training_quantiles_and_supplied_mean(self):
        model = KANSAE(2, 3, normalize_inputs=False).double()
        sample = torch.stack((torch.arange(100.), torch.arange(100.) ** 2), dim=1).double()
        training_mean = torch.tensor([30., 200.], dtype=torch.float64)
        with torch.no_grad():
            model.enc.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-2., 0.25]]))
            model.b_enc.copy_(torch.tensor([0.2, -1., 2.]))
        model.initialize_from_data(sample, training_mean=training_mean, feature_chunk_size=1)
        h = (sample.numpy() - training_mean.numpy()) @ model.enc.weight.detach().numpy().T
        h += model.b_enc.detach().numpy()
        bounds = np.quantile(h, [0.01, 0.99], axis=0)
        for j in range(3):
            breaks = np.linspace(*bounds[:, j], 7)
            expected = np.r_[[breaks[0]] * 4, breaks[1:-1], [breaks[-1]] * 4]
            np.testing.assert_allclose(model.spline.knots[j].numpy(), expected, atol=1e-10)
        torch.testing.assert_close(model.b_pre, training_mean)
        self.assertNotIn("spline.knots", dict(model.named_parameters()))
        self.assertTrue(model.grid_initialized)
        other = KANSAE(**model.config()).double()
        with torch.no_grad():
            other.enc.weight.copy_(model.enc.weight)
            other.b_enc.copy_(model.b_enc)
        other.initialize_from_data(sample, training_mean=training_mean, feature_chunk_size=3)
        torch.testing.assert_close(other.spline.knots, model.spline.knots)

    def test_normalization_matches_baseline_and_constant_samples_are_finite(self):
        model = KANSAE(8, 12)
        sample = torch.randn(100, 8)
        target = model.reconstruction_target(sample)
        torch.testing.assert_close(target.mean(dim=1), torch.zeros(100), atol=1e-7, rtol=0)
        torch.testing.assert_close(target.norm(dim=1), torch.ones(100))
        torch.testing.assert_close(model.reconstruction_target(sample * 4 + 7), target)
        model.initialize_from_data(sample)
        torch.testing.assert_close(model.b_pre, target.mean(dim=0))
        raw_model = KANSAE(8, 12, normalize_inputs=False)
        torch.testing.assert_close(raw_model.reconstruction_target(sample), sample)
        constant_model = KANSAE(8, 12)
        constant_model.initialize_from_data(torch.ones(10, 8))
        self.assertTrue((torch.diff(constant_model.spline.knots[:, 3:10], dim=1) > 0).all())
        reconstruction, code = constant_model(torch.ones(3, 8))
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.isfinite(code).all())
        self.assertEqual(torch.count_nonzero(code).item(), 0)

    def test_loss_uses_signed_l1_and_per_row_sums(self):
        model = KANSAE(2, 3, normalize_inputs=False)
        x = torch.tensor([[1., 2.], [3., 4.]])
        reconstruction = torch.tensor([[0., 0.5], [1., 6.]], requires_grad=True)
        code = torch.tensor([[-1., 2., 0.], [4., -5., 6.]], requires_grad=True)
        losses = model.loss(x, reconstruction, code, l1_weight=0.1)
        self.assertAlmostEqual(losses["reconstruction_loss"].item(), 5.625)
        self.assertAlmostEqual(losses["l1_loss"].item(), 9.)
        self.assertAlmostEqual(losses["loss"].item(), 6.525, places=5)
        losses["loss"].backward()
        torch.testing.assert_close(code.grad, torch.tensor([[-0.05, 0.05, 0.], [0.05, -0.05, 0.05]]))

    def test_zero_initialization_can_start_learning_and_all_parameters_receive_gradients(self):
        model = KANSAE(8, 16)
        x = torch.randn(64, 8)
        model.initialize_from_data(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        reconstruction, code = model(x)
        torch.testing.assert_close(reconstruction, model.b_pre.expand_as(x))
        self.assertEqual(torch.count_nonzero(code).item(), 0)
        model.loss(x, reconstruction, code, l1_weight=1e-4)["loss"].backward()
        self.assertGreater(model.spline.coefficients.grad.abs().sum().item(), 0)
        self.assertEqual(model.enc.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.dec.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        model.renormalize_decoder_()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, code = model(x)
        model.loss(x, reconstruction, code, l1_weight=1e-4)["loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_synthetic_training_reduces_reconstruction_error(self):
        x = torch.randn(128, 8)
        model = KANSAE(8, 24)
        model.initialize_from_data(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = model.loss(x, *model(x), l1_weight=1e-4)["reconstruction_loss"].item()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(x, *model(x), l1_weight=1e-4)
            losses["loss"].backward()
            optimizer.step()
            model.renormalize_decoder_()
        final = model.loss(x, *model(x), l1_weight=1e-4)["reconstruction_loss"].item()
        self.assertLess(final, initial * 0.5)
        torch.testing.assert_close(model.dec.weight.norm(dim=0), torch.ones(24))

    def test_checkpoint_roundtrip_preserves_knots_codes_and_reconstruction(self):
        model = KANSAE(4, 7, normalize_inputs=False).double()
        x = torch.randn(40, 4, dtype=torch.float64)
        model.initialize_from_data(x)
        with torch.no_grad():
            model.spline.coefficients.normal_()
        expected = model(x)
        checkpoint = io.BytesIO()
        torch.save({"model_type": "kan_sae", "config": model.config(), "model_state": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = KANSAE(**saved["config"]).double()
        restored.load_state_dict(saved["model_state"], strict=True)
        self.assertTrue(restored.grid_initialized)
        self.assertEqual(restored.spline.knots.dtype, torch.float64)
        for actual, reference in zip(restored(x), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(restored.spline.knots, model.spline.knots, rtol=0, atol=0)

    def test_graphcast_dimensions_forward_and_backward(self):
        model = KANSAE(d_in=512, latent=4096)
        calibration = torch.randn(32, 512)
        model.initialize_from_data(calibration)
        x = torch.randn(4, 512)
        reconstruction, code = model(x)
        self.assertEqual(reconstruction.shape, (4, 512))
        self.assertEqual(code.shape, (4, 4096))
        self.assertEqual(model.spline.coefficients.numel(), 4096 * 9)
        model.loss(x, reconstruction, code, l1_weight=5e-5)["loss"].backward()
        self.assertTrue(torch.isfinite(model.spline.coefficients.grad).all())

    def test_validation_and_recalibration_guard(self):
        for kwargs in [dict(d_in=0, latent=8), dict(d_in=4, latent=True),
                       dict(d_in=4, latent=8, degree=3, n_basis=3)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KANSAE(**kwargs)
        model = KANSAE(4, 8)
        with self.assertRaisesRegex(RuntimeError, "initialize_from_data"):
            model(torch.randn(2, 4))
        for bad_sample in [torch.empty(0, 4), torch.full((2, 4), float("nan")), torch.ones(3, 5)]:
            with self.subTest(shape=bad_sample.shape), self.assertRaises(ValueError):
                model.initialize_from_data(bad_sample)
        with self.assertRaises(ValueError):
            model.initialize_from_data(torch.randn(5, 4), quantiles=(0.99, 0.01))
        model.initialize_from_data(torch.randn(20, 4))
        with self.assertRaisesRegex(RuntimeError, "recalibrating"):
            model.initialize_from_data(torch.randn(20, 4))
        with self.assertRaises(ValueError):
            model.loss(torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 8), l1_weight=-1)
        with torch.no_grad():
            model.dec.weight[:, 0].zero_()
        with self.assertRaisesRegex(ValueError, "zero or nonfinite"):
            model.renormalize_decoder_()

    def test_alive_criterion_uses_magnitude_and_explicit_threshold(self):
        model = KANSAE(2, 3)
        with torch.no_grad():
            model.spline.coefficients[0, 2] = -2.
            model.spline.coefficients[1, 3] = 1.
        torch.testing.assert_close(model.coefficient_alive_mask(1.5), torch.tensor([True, False, False]))
        with self.assertRaises(ValueError):
            model.coefficient_alive_mask(0)


if __name__ == "__main__":
    unittest.main()
