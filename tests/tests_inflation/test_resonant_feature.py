import unittest
import os

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("ResonantFeature")

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../src/gwb_templates/inflation_templates/data",
    "Resonant_coefficients.npz",
)

with open(_DATA_PATH, "rb") as fh:
    raw = jnp.load(fh, allow_pickle=True)
    _omega_grid = jnp.array(raw["omega"])

# Use an omega_log value well within the precomputed grid
_omega_mid = float((_omega_grid[0] + _omega_grid[-1]) / 2.0)
PARS = jnp.array([0.1, _omega_mid, 0.5])  # A_log, omega_log, phi_log

# Parameters used for the analytic-vs-autodiff gradient comparison: chosen well
# inside the omega grid (which spans [0.001, 100] with ~0.1 spacing) so we stay
# clear of interpax's non-extrapolating edges.
GRAD_PARS = jnp.array([0.5, 10.0, 0.3])


class TestResonantFeatureTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.grad_theta_omega_gw_h2(fvec, GRAD_PARS)

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            GRAD_PARS,
        )

        # The analytic gradient (`_resonant_feature_grad_lin`) uses precomputed
        # closed-form derivative tables (C0p, C1p, ...) of the true continuous
        # coefficient functions; autodiff instead differentiates interpax's cubic
        # interpolant of the value tables (C0, C1, ...) on the omega grid (1000
        # points, spacing ~0.1). These are two legitimate but distinct
        # approximations to the true derivative, so they only agree to
        # interpolation-grid resolution, not machine precision: measured
        # max|grad - grad_fwd| / max|grad_fwd| ~ 3.3e-6 at GRAD_PARS.
        scale = jnp.max(jnp.abs(grad_fwd))
        self.assertTrue(jnp.allclose(grad, grad_fwd, rtol=1e-5, atol=1e-5 * scale))

    def test_near_unity_no_oscillation(self):
        """With A_log ≈ 0 the modulation should be ~1."""
        pars_zero = jnp.array([0.0, _omega_mid, 0.5])
        out = model.omega_gw_h2(fvec, *pars_zero)
        self.assertAlmostEqual(jnp.mean(out).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
