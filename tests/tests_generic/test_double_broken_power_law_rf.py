import unittest

import jax.numpy as jnp


from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("DoubleBrokenPowerLawRf")
# log_amplitude, log_f_2, log_r_f, n_1, n_2, n_3, a_1, a_2
PARS = jnp.array([-10.0, -3.0, 1.0, 3.0, 1.0, -3.0, 2.0, 4.0])


class TestDoubleBrokenPowerLawRfTemplate(unittest.TestCase):

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
