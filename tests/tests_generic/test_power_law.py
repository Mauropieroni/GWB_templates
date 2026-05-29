# Global
import unittest
import jax.numpy as jnp

# Local
from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry


def not_a_test(object):
    object.__test__ = False
    return object


# Define test parameters for different models
PARS = jnp.array([-10.0, 2.0])
model = get_template_from_registry("PowerLaw")

f_points = 100
fvec = jnp.geomspace(c.f_min, c.f_max, f_points)
pivot = model.pivot


class TestSignals(unittest.TestCase):

    def test_gradient_vs_jacfwd(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)

        grad_fwd = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        self.assertAlmostEqual(jnp.sum(jnp.abs(grad - grad_fwd)).item(), 0.0, places=15)

    def test_hessian_power_law(self):
        """
        Test function for the second derivative of a power law signal model.
        """

        signal = model.omega_gw_h2(fvec, *PARS)

        hessian = model.hess_theta_omega_gw_h2(fvec, PARS)

        hessian_analytic = signal[None, None, :] * jnp.array(
            [
                [
                    jnp.log(10.0) ** 2 * fvec**0,
                    jnp.log(10.0) * jnp.log(fvec / pivot),
                ],
                [
                    jnp.log(10.0) * jnp.log(fvec / pivot),
                    jnp.log(fvec / pivot) ** 2,
                ],
            ]
        )

        diff_sum = jnp.sum(jnp.abs(hessian - hessian_analytic.T))
        self.assertAlmostEqual(diff_sum.item(), 0.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
