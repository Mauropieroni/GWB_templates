import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("ExtragalacticSobbhBnsA")
model_full = get_template_from_registry("ExtragalacticSobbhBns")

PARS = jnp.array([-12.0])
PARS_FULL = jnp.array([-12.0, 2.0 / 3.0])


class TestExtragalacticSobbhBnsATemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, 1))

    def test_gradient_vs_jacfwd(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_nonnegative(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_power_law_index(self):
        """Spectral index should be exactly 2/3."""
        out = model.omega_gw_h2(fvec, PARS)
        ratio_spectrum = float(out[1] / out[0])
        ratio_expected = float((fvec[1] / fvec[0]) ** (2.0 / 3.0))
        self.assertAlmostEqual(ratio_spectrum, ratio_expected, places=10)

    def test_consistent_with_full_model(self):
        """Amplitude-only model must match the full model at the same amplitude."""
        out_A = model.omega_gw_h2(fvec, PARS)
        out_full = model_full.omega_gw_h2(fvec, *PARS_FULL)
        self.assertAlmostEqual(
            jnp.max(jnp.abs(out_A - out_full)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
