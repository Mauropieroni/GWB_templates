import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("lognormal_bump")
PARS = jnp.array([-10.0, -3.0, -0.5])  # log_amplitude, log_pivot, log_width


class TestLognormalBumpTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.template(fvec, PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.dtemplate(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.dtemplate(fvec, PARS)
        grad_fwd = jax.jacfwd(model.template, argnums=1)(fvec, PARS)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_peak_at_pivot(self):
        """Spectrum should peak at 10^log_pivot."""
        out = model.template(fvec, PARS)
        pivot = 10.0 ** PARS[1]
        # frequency closest to pivot
        idx = jnp.argmin(jnp.abs(fvec - pivot))
        self.assertEqual(jnp.argmax(out).item(), idx.item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
