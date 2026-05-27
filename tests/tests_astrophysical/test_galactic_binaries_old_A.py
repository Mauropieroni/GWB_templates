import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("galactic_binaries_old_A")
model_full = get_template("galactic_binaries_old")

PARS = jnp.array([-10.0])

# Fiducial values (must match _FIDUCIAL_SHAPE_PARS_OLD in galactic_binaries_old.py)
PARS_FULL = jnp.array([-10.0, 0.138, 221.0, 521.0, 1680.0, 0.00113])


class TestGalacticBinariesOldATemplate(unittest.TestCase):

    def test_shape(self):
        out = model.template(fvec, PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.dtemplate(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, 1))

    def test_gradient_vs_jacfwd(self):
        grad = model.dtemplate(fvec, PARS)
        grad_fwd = jax.jacfwd(model.template, argnums=1)(fvec, PARS)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=8)

    def test_nonnegative(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_consistent_with_full_model(self):
        """Amplitude-only model must match the full model at the same amplitude."""
        out_A = model.template(fvec, PARS)
        out_full = model_full.template(fvec, PARS_FULL)
        self.assertAlmostEqual(
            jnp.max(jnp.abs(out_A - out_full)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
