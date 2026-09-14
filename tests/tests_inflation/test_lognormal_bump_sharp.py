import unittest


import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_lin = get_template_from_registry("LognormalBumpSharp")
model_log = get_template_from_registry("LognormalBumpSharpLog")

# [log_amplitude, log_pivot, log_width, A_lin, omega_lin_Hz, theta_lin]
PARS_LIN = jnp.array([-10.0, -2.0, -0.5, 0.1, 1e3, 0.5])
# [log_amplitude, log_pivot, log_width, log_A_lin, log_omega_lin_Hz, theta_lin]
PARS_LOG = jnp.array([-10.0, -2.0, -0.5, -1.0, 3.0, 0.5])


class TestLognormalBumpSharpTemplate(unittest.TestCase):

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
        # PARS_LOG uses log_amplitude=-10.0, which scales the whole spectrum (and
        # its gradient) down by ~1e-10 and hides the sharp-feature term's real
        # floating-point floor. Override just log_amplitude to 0.0 here (order-1
        # envelope) so the comparison actually probes the unmasked regime; the
        # module-level PARS_LOG is left untouched for the other tests.
        pars_unmasked = PARS_LOG.at[0].set(0.0)
        grad = model_log.grad_theta_omega_gw_h2(fvec, pars_unmasked)
        grad_fwd = gradient_autodiff(
            model_log._omega_from_parameter_vector,
            fvec,
            pars_unmasked,
        )
        # places=15 is unreachable here: the cos/sin argument
        # omega_sharp_Hz * frequency reaches ~500 rad at these params (10**3
        # Hz^-1 * f_max), so float64's ~2e-16 relative precision already limits
        # the argument itself to ~500 * 2e-16 ~ 1e-13 absolute — an irreducible
        # floating-point floor (measured ~3.6e-14 here), not an error in either
        # gradient.
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=10)

    def test_lin_log_agree(self):
        """
        Linear and log variants should produce the same spectrum for matching params.
        """
        out_lin = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        # PARS_LOG: log_A=-1 → A=0.1; log_omega=3 → omega=1e3 → same as PARS_LIN
        out_log = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
