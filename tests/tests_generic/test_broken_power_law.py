import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("BrokenPowerLaw")
# log_amplitude, log_pivot, tilt_1, tilt_2, log_transition
PARS = jnp.array([-10.0, -3.0, 3.0, -3.0, 0.0])
# Sharp transitions: x ** (1 / delta) overflows float64 here in the direct formula.
SHARP_PARS = [
    jnp.array([-12.0, -2.7, 3.0, -4.0, -2.5]),
    jnp.array([-12.0, -2.7, 3.0, -4.0, -3.0]),  # prior lower edge
]


def direct_formula(f, log_amplitude, log_pivot, tilt_1, tilt_2, log_transition):
    """The direct expression, valid where it does not overflow."""
    x = f / 10.0**log_pivot
    delta = 10.0**log_transition
    return (
        10.0**log_amplitude
        * x**tilt_1
        / (0.5 * (1.0 + x ** (1.0 / delta))) ** ((tilt_1 - tilt_2) * delta)
    )


class TestBrokenPowerLawTemplate(unittest.TestCase):

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

    def test_matches_direct_formula(self):
        # Where the direct formula is finite, the log-space evaluation agrees.
        for log_transition in (-1.0, 0.0, 1.0, 3.0):
            pars = PARS.at[4].set(log_transition)
            out = model.omega_gw_h2(fvec, *pars)
            ref = direct_formula(fvec, *pars)
            self.assertLess(float(jnp.max(jnp.abs(jnp.log(out / ref)))), 1e-10)

    def test_sharp_transition_is_finite(self):
        # Regression: the direct formula returns 0 or inf here.
        for pars in SHARP_PARS:
            out = model.omega_gw_h2(fvec, *pars)
            self.assertTrue(bool(jnp.all(jnp.isfinite(out) & (out > 0))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
