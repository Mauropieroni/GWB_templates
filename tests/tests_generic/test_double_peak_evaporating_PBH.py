import unittest


import jax.numpy as jnp
import numpy as np
from scipy.special import hyp2f1

from gwb_templates import constants as c
from gwb_templates.scalar_induced_templates.double_peak_evaporating_PBH import (
    _hyp2f1_series,
    _theta_uv_ad,
    _theta_uv_iso,
)
from gwb_templates.utils import gradient_autodiff
from gwb_templates.template import get_template_from_registry

N_FREQ = 100
fvec = jnp.geomspace(c.f_min, c.f_max, N_FREQ)

model = get_template_from_registry("EvaporatingPBHDoublyPeaked")
# log_m_pbh, log_beta, w, gamma, log_a_s, n_s
PARS = jnp.array([4.0, -12.0, 2.0 / 3.0, 0.2, -9, 0.97])


class TestDoublePeakEvaporatingPBH(unittest.TestCase):

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

        np.testing.assert_allclose(grad, grad_fwd, rtol=1e-12, atol=1e-12)

    def test_uv_windows_match_scipy_hyp2f1_series(self):
        s0 = np.linspace(0.0, 1.0, 100)
        n_eff = 2.5
        z = s0**2 / 3.0

        expected_iso = (
            3.0 * s0 * (5.0 / 9.0 - 2.0 / 3.0 * (2.0 * s0**2 + 5.0) + 9.0)
            - (5.0 / 3.0 * (19.0 / 3.0) - 27.0)
            * s0
            * (z - 1.0)
            * hyp2f1(5.0 / 6.0, 1.0, 1.5, z)
        ) / (10.0 / 9.0 * (1.0 - z) ** (2.0 / 3.0))
        expected_ad = (
            (2.0 / 5.0) * s0**5 * hyp2f1(2.5, -n_eff, 3.5, z)
            - (4.0 / 3.0) * s0**3 * hyp2f1(1.5, -n_eff, 2.5, z)
            + 2.0 * s0 * hyp2f1(0.5, -n_eff, 1.5, z)
        )

        np.testing.assert_allclose(_hyp2f1_series(2.5, -n_eff, 3.5, jnp.asarray(z)), hyp2f1(2.5, -n_eff, 3.5, z))
        np.testing.assert_allclose(_theta_uv_iso(jnp.asarray(s0)), expected_iso)
        np.testing.assert_allclose(
            _theta_uv_ad(jnp.asarray(s0), jnp.asarray(n_eff)), expected_ad
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
