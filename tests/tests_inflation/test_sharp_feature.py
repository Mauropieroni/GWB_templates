import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_lin = get_template("sharp_feature")
model_log = get_template("sharp_feature_log")

# A_lin, omega_lin_Hz, theta_lin
PARS_LIN = jnp.array([0.1, 1e3, 0.5])
# log_A_lin, log_omega_lin_Hz, theta_lin
PARS_LOG = jnp.array([-1.0, 3.0, 0.5])


class TestSharpFeatureTemplate(unittest.TestCase):

    def test_shape_linear(self):
        out = model_lin.template(fvec, PARS_LIN)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_linear(self):
        grad = model_lin.dtemplate(fvec, PARS_LIN)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LIN)))

    def test_gradient_vs_jacfwd_linear(self):
        grad = model_lin.dtemplate(fvec, PARS_LIN)
        grad_fwd = jax.jacfwd(model_lin.template, argnums=1)(fvec, PARS_LIN)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_shape_log(self):
        out = model_log.template(fvec, PARS_LOG)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_log(self):
        grad = model_log.dtemplate(fvec, PARS_LOG)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LOG)))

    def test_gradient_vs_jacfwd_log(self):
        grad = model_log.dtemplate(fvec, PARS_LOG)
        grad_fwd = jax.jacfwd(model_log.template, argnums=1)(fvec, PARS_LOG)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=12)

    def test_lin_log_agree(self):
        """
        Linear and log variants should give the same value for matching params.
        """
        out_lin = model_lin.template(fvec, PARS_LIN)
        # PARS_LOG corresponds to 10^-1=0.1, 10^3=1e3, 0.5 → same as PARS_LIN
        out_log = model_log.template(fvec, PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
