"""Numerical regression checks for balanced single-feature logistic probes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cyclone_probes import DenseFeatureMatrix, RIDGE, fit_probes, release_features


class ProbeConvergenceTests(unittest.TestCase):
    def test_imbalanced_signed_and_nearly_constant_features_match_scalar_optimum(self):
        y = np.arange(4096) < 6  # Similar cyclone/background imbalance to the course run.
        positive = [-1., 1., -2. + 1e-5, 0., -3.]
        negative = [1., -1., -2. - 1e-5, 0., -3.]
        values = np.where(y[:, None], positive, negative).astype(np.float32)
        expected = np.zeros((5, 2))
        for j in range(3):
            scale = float(np.abs(values[:, j]).max())
            yes, no = float(values[0, j]) / scale, float(values[-1, j]) / scale
            center, half_gap = (yes + no) / 2, (yes - no) / 2
            # With two class-specific values and equal class weights, symmetry
            # gives b=-a*center and leaves a one-dimensional derivative root.
            a = brentq(lambda a: RIDGE * a - half_gap * expit(-a * half_gap), -100, 100)
            expected[j] = [a / scale, -a * center]
        with tempfile.TemporaryDirectory() as temporary:
            dense = DenseFeatureMatrix(values.shape, temporary)
            try:
                dense.array[:] = values
                for matrix in [sparse.csc_matrix(values), dense]:
                    fitted = fit_probes(matrix, y)
                    np.testing.assert_allclose(fitted, expected, atol=2e-5, rtol=1e-5)
                    self.assertLess(fitted[0, 0], 0)
                    self.assertGreater(fitted[1, 0], 0)
            finally:
                release_features(dense)

    def test_bad_optimizer_result_uses_independent_hessian_fallback(self):
        y = np.arange(1000) < 2
        values = np.where(y, -1., 1.).astype(np.float32)[:, None]
        expected_slope = brentq(lambda a: expit(a) + RIDGE * a, -100, 0)
        for success, params in [(False, [0., 0.]), (True, [0., 0.]), (False, [np.nan, np.nan])]:
            with self.subTest(success=success, params=params):
                methods = []

                def failed_line_search(fun, x0, *, method, **kwargs):
                    methods.append(method)
                    if method == "L-BFGS-B":
                        return OptimizeResult(x=np.array(params), success=success, message="ABNORMAL: ")
                    # Check the analytic Hessian independently against finite
                    # differences of the gradient before exercising the solver.
                    point, eps = np.array([0.3, -0.7]), 1e-5
                    finite_difference = np.column_stack([
                        (fun(point + d)[1] - fun(point - d)[1]) / (2 * eps)
                        for d in np.eye(2) * eps])
                    np.testing.assert_allclose(kwargs["hess"](point), finite_difference,
                                               atol=1e-10, rtol=1e-8)
                    return minimize(fun, x0, method=method, **kwargs)

                with patch("cyclone_probes.minimize", side_effect=failed_line_search):
                    fitted = fit_probes(sparse.csc_matrix(values), y)
                self.assertEqual(methods, ["L-BFGS-B", "trust-exact"])
                np.testing.assert_allclose(fitted[0], [expected_slope, 0.], atol=2e-6)

    def test_warning_at_converged_solution_is_accepted(self):
        y = np.arange(1000) < 2
        values = np.where(y, -1., 1.).astype(np.float32)[:, None]
        optimum = brentq(lambda a: expit(a) + RIDGE * a, -100, 0)
        warning = OptimizeResult(x=np.array([optimum, 0.]), success=False, message="ABNORMAL: ")
        with patch("cyclone_probes.minimize", return_value=warning) as solver:
            fitted = fit_probes(sparse.csc_matrix(values), y)
        self.assertEqual(solver.call_count, 1)
        np.testing.assert_allclose(fitted[0], [optimum, 0.], atol=1e-12)

    def test_unconverged_feature_is_not_silently_skipped(self):
        y = np.array([False, False, True])
        values = sparse.csc_matrix(np.array([1., 1., -1.])[:, None])
        bad = OptimizeResult(x=np.zeros(2), success=True, message="reported success")
        with patch("cyclone_probes.minimize", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "Probe 0.*both solvers.*gradient norm"):
                fit_probes(values, y)

    def test_nonfinite_constant_feature_is_rejected(self):
        y = np.array([False, False, True])
        for bad_value in [np.nan, np.inf, -np.inf]:
            values = sparse.csc_matrix(np.full((len(y), 1), bad_value))
            with self.subTest(value=bad_value), self.assertRaisesRegex(ValueError, "Feature 0.*finite"):
                fit_probes(values, y)


if __name__ == "__main__":
    unittest.main()
