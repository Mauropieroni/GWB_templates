import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("fopt_old")
# log_amplitude, log_f_star
PARS = jnp.array([-10.0, -3.0])


class TestFoptBrokenPowerLawOldTemplate(unittest.TestCase):

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

    def test_positive(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(out > 0).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
