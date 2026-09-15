import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.scalar_induced_templates.PBH_clustering_double_peak_gw_template import (
    PBH_double_peak_non_Gaussian,
)
from gwb_templates.utils import gradient_autodiff

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = PBH_double_peak_non_Gaussian()
# M_PBH [grams], Omega_f, tau_NL
PARS = jnp.array([1.0e4, 1.0e-10, 1.0e-3])


class TestPBHClusteringDoublePeakTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

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

        rel_err = jnp.max(jnp.abs(grad - grad_fwd)) / (
            jnp.max(jnp.abs(grad_fwd)) + 1e-30
        )
        self.assertLess(float(rel_err), 1e-8)

    def test_nonnegative(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out >= 0.0).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)