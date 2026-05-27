import unittest

import jax
import jax.numpy as jnp

from gwb_templates.templates import get_template

N_FREQ = 100
F_MIN = 3e-5
F_MAX = 5e-1
fvec = jnp.geomspace(F_MIN, F_MAX, N_FREQ)

model = get_template("extragalactic_sobbh_bns")
# log_amplitude, tilt
PARS = jnp.array([-12.0, 2.0 / 3.0])


class TestExtragalacticSobbhBnsTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.template(fvec, PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.dtemplate(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.dtemplate(fvec, PARS)
        grad_fwd = jax.jacfwd(model.template, argnums=1)(fvec, PARS)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=8)

    def test_nonnegative(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_power_law_index(self):
        """Spectral index should match the tilt parameter."""
        out = model.template(fvec, PARS)
        ratio_spectrum = float(out[1] / out[0])
        ratio_expected = float((fvec[1] / fvec[0]) ** PARS[1])
        self.assertAlmostEqual(ratio_spectrum, ratio_expected, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
