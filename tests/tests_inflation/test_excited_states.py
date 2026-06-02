import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("ExcitedStates")

# Choose parameters such that x = 0.5 * f * 10^log_omega_ES is well below
# the cutoff x_cut = 2 * 10^log_gamma_ES for all PTA frequencies.
# With log_omega_ES = 8 (omega = 1e8 Hz) and fvec ~ 1e-9..1e-7 Hz:
#   x ~ 0.5 * 1e-7 * 1e8 = 5
# With log_gamma_ES = 1 (gamma_ES = 10): x_cut = 20 >> x → well inside.
PARS = jnp.array([-15.0, 1.0, 8.0])


class TestExcitedStatesTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_positive(self):
        """Spectrum should be positive for well-chosen parameters."""
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
