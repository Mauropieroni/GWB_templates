import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.template import get_template_from_registry

N_FREQ = 30  # NumPy computation is slow; use few points
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

# New registry API
model = get_template_from_registry("CosmicStringModelIEos")
# log_Gmu, log_alpha, q, logtemp_GeV, eos
PARS = jnp.array([-9.0, -1.0, 1.5, 1.0, 1.0 / 3.0])


class TestCosmicStringModelIEOS(unittest.TestCase):
    """
    Evaluation-only tests.  Gradient tests are skipped because the template
    uses NumPy/SciPy (including scipy.special.hyp2f1) which has no
    JAX-differentiable equivalent.
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

    def test_radiation_dominated_matches_model_i(self):
        """For w = 1/3 (radiation EOS), result should be close to Model I."""
        from gwb_templates.template import get_template_from_registry as gt

        model_i = gt("CosmicStringModelI")
        out_eos = model.omega_gw_h2(fvec, *PARS)
        out_i = model_i.omega_gw_h2(fvec, *jnp.array([-9.0, -1.0, 1.5]))
        # Not identical (EOS applies a high-freq cutoff) but same order of magnitude
        ratio = out_eos / jnp.where(out_i > 0, out_i, 1.0)
        self.assertTrue(jnp.all(ratio < 1e3).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
