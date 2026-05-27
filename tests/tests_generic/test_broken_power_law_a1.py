import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("broken_power_law_a1")
# log_amplitude, log_f_b, n_1, n_2, a_1
PARS = jnp.array([-10.0, -3.0, 3.0, -3.0, 1.5])


class TestBrokenPowerLawA1Template(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
