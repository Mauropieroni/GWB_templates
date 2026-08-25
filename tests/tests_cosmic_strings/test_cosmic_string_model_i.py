import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.template import get_template_from_registry

N_FREQ = 50  # fewer points: NumPy computation is slow
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

# New registry API
model = get_template_from_registry("CosmicStringModelI")
# log_Gmu, log_alpha, q
PARS = jnp.array([-9.0, -1.0, 1.5])


class TestCosmicStringModelITemplate(unittest.TestCase):
    """
    Evaluation-only tests.  Gradient tests are skipped because the template
    uses scipy.special.hyp2f1 which has no JAX-differentiable equivalent.
    """

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_nonnegative(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())

    def test_finite(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(jnp.isfinite(out)).item())

    def test_returns_jax_array(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertIsInstance(out, jax.Array)


if __name__ == "__main__":
    unittest.main(verbosity=2)
