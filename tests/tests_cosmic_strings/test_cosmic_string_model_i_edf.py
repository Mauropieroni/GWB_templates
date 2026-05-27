import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 50  # fewer points: NumPy computation is slow
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("cosmic_string_model_i_edf")
model_base = get_template("cosmic_string_model_i")

# log_Gmu, log_alpha, q, log_T_Extra, Dg_Extra
PARS = jnp.array([-9.0, -1.0, 1.5, -1.0, 10.0])
# Base model params (same Gmu, alpha, q)
PARS_BASE = jnp.array([-9.0, -1.0, 1.5])


class TestCosmicStringModelIEdfTemplate(unittest.TestCase):
    """
    Evaluation-only tests.  Gradient tests are skipped because the template
    uses scipy.special.hyp2f1 which has no JAX-differentiable equivalent.
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

    def test_differs_from_base_model(self):
        """EDF model with extra DOF should differ from the base model."""
        out_edf = model.template(fvec, PARS)
        out_base = model_base.template(fvec, PARS_BASE)
        diff = float(jnp.sum(jnp.abs(out_edf - out_base)))
        self.assertGreater(diff, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
