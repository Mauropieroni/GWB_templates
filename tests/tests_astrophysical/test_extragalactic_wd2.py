import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

# New registry API
model = get_template_from_registry("ExtragalacticWd2")
# log_amplitude, f_low [Hz], f_high [Hz], alpha_low, alpha_high
PARS = jnp.array([-10.0, 1e-4, 1e-2, 3.0, -1.0])


class TestExtragalacticWD2Template(unittest.TestCase):

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
