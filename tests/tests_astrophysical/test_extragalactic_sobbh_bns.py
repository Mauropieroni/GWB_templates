import unittest

import jax.numpy as jnp

from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
F_MIN = 3e-5
F_MAX = 5e-1
fvec = jnp.geomspace(F_MIN, F_MAX, N_FREQ)

# Use new registry API
model = get_template_from_registry("ExtragalacticSobbhBns")
# log_amplitude, tilt
PARS = jnp.array([-12.0, 2.0 / 3.0])


class TestExtragalacticSobbhBnsTemplate(unittest.TestCase):

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

    def test_nonnegative(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_power_law_index(self):
        """Spectral index should match the tilt parameter."""
        out = model.omega_gw_h2(fvec, *PARS)
        ratio_spectrum = float(out[1] / out[0])
        ratio_expected = float((fvec[1] / fvec[0]) ** PARS[1])
        self.assertAlmostEqual(ratio_spectrum, ratio_expected, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
