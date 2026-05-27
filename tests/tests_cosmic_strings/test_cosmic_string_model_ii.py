import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 50
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("cosmic_string_model_ii")
# log_Gmu (must be within grid range -18 .. -9.5)
PARS = jnp.array([-12.0])


class TestCosmicStringModelII(unittest.TestCase):
    """
    The template uses JAX bilinear interpolation so it is fully differentiable.
    Gradient tests compare the explicit analytical formula (dtemplate) against
    jax.jacfwd to machine precision.
    """

    def test_shape(self):
        out = model.template(fvec, PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_nonnegative(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_finite(self):
        out = model.template(fvec, PARS)
        self.assertTrue(jnp.all(jnp.isfinite(out)).item())

    def test_returns_jax_array(self):
        out = model.template(fvec, PARS)
        self.assertIsInstance(out, jax.Array)

    def test_gradient_shape(self):
        grad = model.dtemplate(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        """Analytical gradient must match jax.jacfwd to machine precision."""
        grad_analytic = model.dtemplate(fvec, PARS)
        grad_auto = jax.jacfwd(model.template, argnums=1)(fvec, PARS)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(grad_analytic - grad_auto)).item(),
            0.0,
            places=15,
        )

    def test_larger_gmu_gives_larger_spectrum(self):
        """Increasing G*mu should increase the signal amplitude."""
        pars_small = jnp.array([-14.0])
        pars_large = jnp.array([-11.0])
        out_small = model.template(fvec, pars_small)
        out_large = model.template(fvec, pars_large)
        mid = N_FREQ // 2
        self.assertGreater(float(out_large[mid]), float(out_small[mid]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
