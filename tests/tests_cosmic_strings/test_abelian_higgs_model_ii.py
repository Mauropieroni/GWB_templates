import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 50
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("AbelianHiggsModelII")
# log_Gmu, logf
PARS = jnp.array([-12.0, 0.5])


class TestAbelianHiggsModelII(unittest.TestCase):
    """
    The template uses JAX bilinear interpolation so it is fully differentiable.
    Gradient tests compare the explicit analytical formula (dtemplate) against
    jax.jacfwd to machine precision.
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

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_logf_scales_amplitude(self):
        """Increasing logf by 1 should scale the whole spectrum by 10."""
        pars_base = jnp.array([-12.0, 0.0])
        pars_up = jnp.array([-12.0, 1.0])
        out_base = model.omega_gw_h2(fvec, *pars_base)
        out_up = model.omega_gw_h2(fvec, *pars_up)
        # All non-zero entries should be scaled by ~10 (ratio 10.0 ± 1%)
        mask = out_base > 0.0
        ratio = out_up[mask] / out_base[mask]
        self.assertTrue(jnp.allclose(ratio, 10.0, rtol=0.01).item())

    def test_same_shape_as_model_ii(self):
        """With logf=0, Abelian Higgs equals Model II exactly."""
        model_ii = get_template_from_registry("CosmicStringModelII")
        pars_ah = jnp.array([-12.0, 0.0])
        pars_ii = jnp.array([-12.0])
        out_ah = model.omega_gw_h2(fvec, *pars_ah)
        out_ii = model_ii.omega_gw_h2(fvec, *pars_ii)
        self.assertTrue(jnp.allclose(out_ah, out_ii, rtol=1e-6).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
