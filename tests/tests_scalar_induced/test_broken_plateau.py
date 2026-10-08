import unittest

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.scalar_induced_templates.broken_plateau import (
    c_si,
    omega_gw_r_broken_plateau,
)
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("BrokenPlateauSIGW")
PARS = jnp.array([-2.0, -2.0, 2.0, 0.95])  # log_A_zeta, log_f_star, Delta, n_s

# (kappa, Delta, n_s, Omega_GW,r / A_zeta^2) from the reference implementation
# of Eqs. (17)-(19) used for Fig. 3 of the LISA-ET multiband paper.
REFERENCE_VALUES = (
    (0.01, 2.0, 1.0, 0.00018661476092839501),
    (0.3, 2.0, 1.0, 0.04910467661124866),
    (1.0, 2.0, 1.0, 0.05137100354486492),
    (10.0, 2.0, 1.0, 0.0001712215573236408),
    (1.0, 1.0, 0.9, 0.19744384630418532),
    (1.0, 5.0, 1.1, 0.007847453447585738),
    (20.0, 3.0, 0.9, 0.0034226470824394164),
)


class TestBrokenPlateauSIGWTemplate(unittest.TestCase):

    def test_shape(self):
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS)))

    def test_reference_values(self):
        for kappa, Delta, n_s, expected in REFERENCE_VALUES:
            with self.subTest(kappa=kappa, Delta=Delta, n_s=n_s):
                out = omega_gw_r_broken_plateau(kappa, Delta, n_s)
                self.assertAlmostEqual(out.item() / expected, 1.0, places=12)

    def test_h2_conversion(self):
        """omega_gw_h2 is the dimensionless spectrum times A_zeta^2 and T_0."""
        log_A_zeta, log_f_star, Delta, n_s = PARS
        expected = (
            model.transfer_today
            * 10.0 ** (2.0 * log_A_zeta)
            * omega_gw_r_broken_plateau(fvec / 10.0**log_f_star, Delta, n_s)
        )
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.allclose(out, expected, rtol=1e-14, atol=0.0))

    def test_plateau_limit(self):
        """For a broad band the centre sits on C_SI(n_s) / (2 Delta)^2."""
        Delta = 5.0
        out = omega_gw_r_broken_plateau(1.0, Delta, 1.0)
        self.assertAlmostEqual(out.item() / (c_si(1.0) / (2.0 * Delta) ** 2), 1.0, 6)

    def test_uv_cutoff(self):
        """The spectrum vanishes identically above f = 2 e^Delta f_star."""
        log_A_zeta, log_f_star, Delta, n_s = PARS
        f_cut = 2.0 * jnp.exp(Delta) * 10.0**log_f_star
        out = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.all(out[fvec > f_cut] == 0.0))
        self.assertTrue(jnp.all(out[fvec < f_cut] > 0.0))

    def test_gradient_finite(self):
        """No NaNs at n_s = 1 (x / sinh x) or beyond the UV cutoff."""
        for n_s in (0.9, 1.0, 1.1):
            with self.subTest(n_s=n_s):
                pars = PARS.at[3].set(n_s)
                grad = model.grad_theta_omega_gw_h2(fvec, pars)
                hess = model.hess_theta_omega_gw_h2(fvec, pars)
                self.assertTrue(jnp.all(jnp.isfinite(grad)))
                self.assertTrue(jnp.all(jnp.isfinite(hess)))

    def test_gradient_vs_finite_difference(self):
        grad = model.grad_theta_omega_gw_h2(fvec, PARS)
        step = 1e-6
        for i in range(len(PARS)):
            with self.subTest(parameter=model.parameter_names[i]):
                up = model.omega_gw_h2(fvec, *PARS.at[i].add(step))
                down = model.omega_gw_h2(fvec, *PARS.at[i].add(-step))
                grad_fd = (up - down) / (2.0 * step)
                self.assertTrue(
                    jnp.allclose(grad[:, i], grad_fd, rtol=1e-6, atol=1e-20)
                )

    def test_jit(self):
        out = jax.jit(model.omega_gw_h2)(fvec, *PARS)
        expected = model.omega_gw_h2(fvec, *PARS)
        self.assertTrue(jnp.allclose(out, expected, rtol=1e-14, atol=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
