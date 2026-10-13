import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 50
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("CosmicStringModelIIDof")
# log_Gmu, log_T_delta, delta_g (must be within the grid range)
PARS = jnp.array([-10.0, 0.0, 50.0])


class TestCosmicStringModelIIDof(unittest.TestCase):
    """
    The template uses JAX quadrilinear interpolation so it is fully
    differentiable. There's no hand-written analytical gradient (unlike
    CosmicStringModelII/AbelianHiggsModelII), so grad_theta_omega_gw_h2
    falls back to autodiff; the gradient test just checks it's finite and
    matches an independent jax.jacfwd call.
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

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_gradient_vs_jacfwd(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        grad_fwd = gradient_autodiff(model._omega_from_parameter_vector, fvec, PARS)
        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_delta_g_changes_spectrum(self):
        """Extra DOF should shift the spectrum relative to delta_g=0."""
        pars_zero = jnp.array([-10.0, 0.0, 0.0])
        pars_extra = jnp.array([-10.0, 0.0, 100.0])
        out_zero = model.omega_gw_h2(fvec, *pars_zero)
        out_extra = model.omega_gw_h2(fvec, *pars_extra)
        diff = float(jnp.sum(jnp.abs(out_extra - out_zero)))
        self.assertGreater(diff, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
