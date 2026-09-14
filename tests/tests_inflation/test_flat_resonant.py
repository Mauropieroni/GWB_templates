import os
import unittest

import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model_lin = get_template_from_registry("FlatResonant")
model_log = get_template_from_registry("FlatResonantLog")

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../src/gwb_templates/inflation_templates/data",
    "Resonant_coefficients.npz",
)

with open(_DATA_PATH, "rb") as fh:
    raw = jnp.load(fh, allow_pickle=True)
    _omega_grid = jnp.array(raw["omega"])

_omega_mid = float((_omega_grid[0] + _omega_grid[-1]) / 2.0)

# [log_amplitude, A_log, omega_log, phi_log]
PARS_LIN = jnp.array([-10.0, 0.1, _omega_mid, 0.5])
# [log_amplitude, log_A_log, log_omega_log, phi_log]
PARS_LOG = jnp.array([-10.0, jnp.log10(0.1), jnp.log10(_omega_mid), 0.5])

# Parameters used for the analytic-vs-autodiff gradient comparison: chosen well
# inside the omega grid (which spans [0.001, 100] with ~0.1 spacing) so we stay
# clear of interpax's non-extrapolating edges.
GRAD_PARS_LIN = jnp.array([-10.0, 0.5, 10.0, 0.3])
GRAD_PARS_LOG = jnp.array([-10.0, jnp.log10(0.5), jnp.log10(10.0), 0.3])


class TestFlatResonantTemplate(unittest.TestCase):

    def test_shape_linear(self):
        out = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_linear(self):
        grad = model_lin.grad_theta_omega_gw_h2(fvec, PARS_LIN)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LIN)))

    def test_gradient_vs_jacfwd_linear(self):
        grad = model_lin.grad_theta_omega_gw_h2(fvec, GRAD_PARS_LIN)

        grad_fwd = gradient_autodiff(
            model_lin._omega_from_parameter_vector,
            fvec,
            GRAD_PARS_LIN,
        )

        # The analytic gradient (via `_resonant_feature_grad_lin`) uses
        # precomputed closed-form derivative tables (C0p, C1p, ...) of the true
        # continuous coefficient functions; autodiff instead differentiates
        # interpax's cubic interpolant of the value tables (C0, C1, ...) on the
        # omega grid (1000 points, spacing ~0.1). These are two legitimate but
        # distinct approximations to the true derivative, so they only agree to
        # interpolation-grid resolution, not machine precision: measured
        # max|grad - grad_fwd| / max|grad_fwd| ~ 3.3e-6 at GRAD_PARS_LIN.
        scale = jnp.max(jnp.abs(grad_fwd))
        self.assertTrue(jnp.allclose(grad, grad_fwd, rtol=1e-5, atol=1e-5 * scale))

    def test_shape_log(self):
        out = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertEqual(out.shape, (N_FREQ,))

    def test_gradient_shape_log(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, PARS_LOG)
        self.assertEqual(grad.shape, (N_FREQ, len(PARS_LOG)))

    def test_gradient_vs_jacfwd_log(self):
        grad = model_log.grad_theta_omega_gw_h2(fvec, GRAD_PARS_LOG)

        grad_fwd = gradient_autodiff(
            model_log._omega_from_parameter_vector,
            fvec,
            GRAD_PARS_LOG,
        )

        # See test_gradient_vs_jacfwd_linear: the analytic gradient uses
        # precomputed derivative tables while autodiff differentiates the cubic
        # value-table interpolant, so agreement is limited by interpolation-
        # grid resolution rather than machine precision: measured
        # max|grad - grad_fwd| / max|grad_fwd| ~ 3.3e-6 at GRAD_PARS_LOG.
        scale = jnp.max(jnp.abs(grad_fwd))
        self.assertTrue(jnp.allclose(grad, grad_fwd, rtol=1e-5, atol=1e-5 * scale))

    def test_lin_log_agree(self):
        """
        Linear and log variants should produce the same spectrum for matching params.
        """
        out_lin = model_lin.omega_gw_h2(fvec, *PARS_LIN)
        out_log = model_log.omega_gw_h2(fvec, *PARS_LOG)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(out_lin - out_log)).item(), 0.0, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
