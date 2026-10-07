import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

# a_b, omega_b_over_beta, spectral_index_IR, spectral_index_UV, transition_smoothness
HYPERPARS = jnp.array([0.05, 0.7, 2.4, -2.4, 1.2])
# New registry API
model = get_template_from_registry("PtCollision", *HYPERPARS)
# log_K_tilde, log_beta_over_H, log_T_star
PARS = jnp.array([-1.0, 2.0, 2.0])


class TestPtCollisionTemplate(unittest.TestCase):

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

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
