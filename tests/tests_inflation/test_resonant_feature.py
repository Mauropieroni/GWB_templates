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


class TestResonantFeatureTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_near_unity_no_oscillation(self):
        """With A_log ≈ 0 the modulation should be ~1."""
        pars_zero = jnp.array([0.0, _omega_mid, 0.5])
        out = model.omega_gw_h2(fvec, *pars_zero)
        self.assertAlmostEqual(jnp.mean(out).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
