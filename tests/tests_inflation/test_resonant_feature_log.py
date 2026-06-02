import unittest

import os
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_log = get_template_from_registry("ResonantFeatureLog")

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../src/gwb_templates/inflation_templates/data",
    "Resonant_coefficients.npz",
)

with open(_DATA_PATH, "rb") as fh:
    raw = jnp.load(fh, allow_pickle=True)
    _omega_grid = jnp.array(raw["omega"])

_omega_mid = float((_omega_grid[0] + _omega_grid[-1]) / 2.0)

PARS_LOG = jnp.array([-1.0, jnp.log10(_omega_mid), 0.5])


class TestResonantFeatureLogTemplate(unittest.TestCase):

    def test_shape(self):
        out = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, PARS_LOG)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LOG)))

    def test_gradient_vs_jacfwd(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, PARS_LOG)

        grad_fwd = gradient_autodiff(
            model_log._omega_from_parameter_vector,
            fvec,
            PARS_LOG,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_lin_log_agree(self):
        """
        Resonant_feature and resonant_feature_log should match for same physical params.
        """
        model_lin = get_template_from_registry("ResonantFeature")
        PARS_LIN = jnp.array([0.1, _omega_mid, 0.5])  # A_log, omega_log, phi_log
        out_lin = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        out_log = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
