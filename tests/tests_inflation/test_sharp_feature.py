import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_lin = get_template_from_registry("SharpFeature")
model_log = get_template_from_registry("SharpFeatureLog")

# A_lin, omega_lin_Hz, theta_lin
PARS_LIN = jnp.array([0.1, 1e3, 0.5])
# log_A_lin, log_omega_lin_Hz, theta_lin
PARS_LOG = jnp.array([-1.0, 3.0, 0.5])


class TestSharpFeatureTemplate(unittest.TestCase):

    def test_shape_linear(self):
        out = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_linear(self):
        grad = model_lin.grad_theta_omega_gw_h2(fvec, PARS_LIN)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LIN)))

    def test_gradient_vs_jacfwd(self):
        grad = model_lin.grad_theta_omega_gw_h2(fvec, PARS_LIN)

        grad_fwd = gradient_autodiff(
            model_lin._omega_from_parameter_vector,
            fvec,
            PARS_LIN,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_shape_log(self):
        out = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_log(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, PARS_LOG)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LOG)))

    def test_gradient_vs_jacfwd_log(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, PARS_LOG)
        grad_fwd = gradient_autodiff(
            model_log._omega_from_parameter_vector,
            fvec,
            PARS_LOG,
        )
        # places=15 (as used elsewhere in this suite) is unreachable here: the
        # cos/sin argument omega_sharp_Hz * frequency reaches ~500 rad at these
        # PARS_LOG (10**3 Hz^-1 * f_max), so float64's ~2e-16 relative precision
        # already limits the argument itself to ~500 * 2e-16 ~ 1e-13 absolute —
        # an irreducible floating-point floor, not an error in either gradient.
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=10)

    def test_lin_log_agree(self):
        """
        Linear and log variants should give the same value for matching params.
        """
        out_lin = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        # PARS_LOG corresponds to 10^-1=0.1, 10^3=1e3, 0.5 → same as PARS_LIN
        out_log = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
