import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.inflation_templates.resonant_feature import _omega_grid
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_log = get_template("resonant_feature_log")

# log_A_log = log10(0.1) = -1, log_omega_log = log10(_omega_mid)
_omega_mid = float((_omega_grid[0] + _omega_grid[-1]) / 2.0)

PARS_LOG = jnp.array([-1.0, jnp.log10(_omega_mid), 0.5])


class TestResonantFeatureLogTemplate(unittest.TestCase):

    def test_shape(self):
        out = model_log.template(fvec, PARS_LOG)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model_log.dtemplate(fvec, PARS_LOG)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LOG)))

    def test_gradient_vs_jacfwd(self):
        grad = model_log.dtemplate(fvec, PARS_LOG)
        grad_fwd = jax.jacfwd(model_log.template, argnums=1)(fvec, PARS_LOG)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=3)

    def test_lin_log_agree(self):
        """
        Resonant_feature and resonant_feature_log should match for same physical params.
        """
        model_lin = get_template("resonant_feature")
        PARS_LIN = jnp.array([0.1, _omega_mid, 0.5])  # A_log, omega_log, phi_log
        out_lin = model_lin.template(fvec, PARS_LIN)
        out_log = model_log.template(fvec, PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
