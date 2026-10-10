import unittest

import jax.numpy as jnp

from gwb_templates.scalar_induced_templates.parity_violation.helper_functions import (
    evaluate_components_jax,
)
from gwb_templates import constants as c
from gwb_templates.template import get_template_from_registry
from gwb_templates.utils import gradient_autodiff

model = get_template_from_registry("OddSIGW")
fvec = jnp.geomspace(c.f_min, c.f_max, 100)
PARS = jnp.array([10.0, 0.5, -2.0, 0.0, 0.0, 0.0])


class TestParityOddTemplate(unittest.TestCase):
    def test_parameter_names(self):
        self.assertEqual(
            model.parameter_names,
            (
                "target_f_peak",
                "target_n2",
                "log10_A_zeta",
                "log10_f_NL",
                "log10_tau_NL",
                "log10_tilde_tau_NL",
            ),
        )

    def test_shape(self):
        spectrum = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(spectrum.shape, fvec.shape)

    def test_gradient_shape(self):
        gradient = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(gradient.shape, fvec.shape + (len(PARS),))

    def test_gradient_vs_jacfwd(self):
        gradient = model.grad_theta_omega_gw_h2(fvec, PARS)
        reference = gradient_autodiff(
            model._omega_from_parameter_vector,
            fvec,
            PARS,
        )

        relative_error = jnp.max(jnp.abs(gradient - reference)) / (
            jnp.max(jnp.abs(reference)) + 1e-30
        )
        self.assertLess(float(relative_error), 1e-8)

    def test_nonnegative(self):
        spectrum = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(bool(jnp.all(spectrum >= 0.0)))

    def test_n2_outside_precomputed_range_returns_zero(self):
        spectrum = model.omega_gw_h2(fvec, PARS[0], 1.1, *PARS[2:])
        self.assertTrue(bool(jnp.all(spectrum == 0.0)))

    def test_scale_parameter_gradients_are_analytical(self):
        components = evaluate_components_jax(
            fvec,
            PARS[0],
            PARS[1],
            PARS[2],
            PARS[3],
            PARS[4],
            PARS[5],
        )
        gradient = model.grad_theta_omega_gw_h2(fvec, PARS)
        log10 = jnp.log(10.0)
        expected = jnp.stack(
            [
                log10
                * (
                    2.0 * components[0]
                    + 3.0 * components[1]
                    + 3.0 * components[2]
                )
                + 3.0 * log10 * jnp.abs(components[3]),
                2.0 * log10 * components[1],
                log10 * components[2],
                log10 * jnp.abs(components[3]),
            ],
            axis=-1,
        )

        self.assertTrue(
            bool(jnp.allclose(gradient[:, 2:], expected, rtol=1e-12, atol=1e-12))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
