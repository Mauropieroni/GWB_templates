import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.templates import get_template

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template("fopt_broken_power_law")
# log_amplitude, log_f_star, n_IR, n_UV
PARS = jnp.array([-10.0, -3.0, 3.0, -3.0])


class TestFoptBrokenPowerLawTemplate(unittest.TestCase):

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

    def test_matches_broken_power_law_fixed_smoothness(self):
        """FOPT BPL uses the same function as broken_power_law_fixed_smoothness."""
        from gwb_templates.templates import get_template as gt

        bpl_fs = gt("broken_power_law_fixed_smoothness")
        out_fopt = model.template(fvec, PARS)
        out_bpl_fs = bpl_fs.template(fvec, PARS)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_fopt - out_bpl_fs)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
