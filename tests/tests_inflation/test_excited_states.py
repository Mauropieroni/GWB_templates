import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("ExcitedStates")

# Choose parameters so x = 0.5 * f * 10^log_omega_ES actually crosses the
# Heaviside cutoff x_cut = 2 * 10^log_gamma_ES within [c.f_min, c.f_max]:
# x ranges over [0.5 * c.f_min * 10**1.5, 0.5 * c.f_max * 10**1.5] ~ [5e-4, 8],
# straddling x_cut = 2 * 10**0.0 = 2 — the cutoff this template is defined by.
PARS = jnp.array([-10.0, 0.0, 1.5])  # log_amplitude, log_gamma_ES, log_omega_ES

_X = 0.5 * fvec * 10.0 ** PARS[2]
_X_CUT = 2.0 * 10.0 ** PARS[1]
_BELOW_CUTOFF = _X < _X_CUT
_AWAY_FROM_JUMP = jnp.abs(_X - _X_CUT) > 1e-3 * _X_CUT


class TestExcitedStatesTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_nonnegative(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_nonzero_below_cutoff(self):
        """The oscillatory factor should actually be exercised, not just the cutoff."""
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.any(out[_BELOW_CUTOFF] != 0.0).item())

    def test_zero_beyond_cutoff(self):
        """Heaviside(x_cut - x) should zero out the spectrum past the cutoff."""
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.allclose(out[~_BELOW_CUTOFF], 0.0))

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        """
        Compare away from the exact cutoff: the Heaviside jump has a formally
        undefined derivative there, dropped by design (see the analytic override's
        docstring) — both the analytic and autodiff gradients agree everywhere else.
        """
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        diff = jnp.abs(grad - grad_fwd)[_AWAY_FROM_JUMP]
        self.assertAlmostEqual(jnp.sum(diff).item(), 0.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
