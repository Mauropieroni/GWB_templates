import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("extragalactic_wd")
# log_amplitude, f_knee [Hz], delta, alpha1, alpha2
PARS = jnp.array([-10.0, 1e-3, 2.0, 2.0 / 3.0, -1.0])


class TestExtragalacticWDTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.template(fvec, PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.dtemplate(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.dtemplate(fvec, PARS)
        grad_fwd = jax.jacfwd(model.template, argnums=1)(fvec, PARS)
        rel_err = jnp.max(jnp.abs(grad - grad_fwd)) / (
            jnp.max(jnp.abs(grad_fwd)) + 1e-30
        )
        self.assertLess(float(rel_err), 1e-8)

    def test_nonnegative(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
