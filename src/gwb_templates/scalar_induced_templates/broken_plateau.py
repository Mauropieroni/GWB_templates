r"""
Broken-plateau scalar-induced GW template (closed form, radiation domination).

The curvature spectrum is a power law of tilt :math:`n_s` restricted to a band
of :math:`2\Delta` e-folds around :math:`k_\star`, normalised so that
:math:`\int \mathcal{P}_\zeta\, \mathrm{d}\ln k = A_\zeta`:

.. math::

    \mathcal{P}_\zeta(k) = A_\zeta\,\frac{n_s-1}{2\sinh[(n_s-1)\Delta]}
        \left(\frac{k}{k_\star}\right)^{n_s-1}
        \Theta\big(\Delta - |\ln(k/k_\star)|\big).

It bridges a power law (:math:`\Delta\to\infty`) and a monochromatic peak
(:math:`\Delta\to0`). The induced spectrum during radiation domination is
approximated by a single elementary formula,

.. math::

    \frac{\Omega_{\mathrm{GW},r}}{A_\zeta^2} \simeq
        \left[\frac{n_s-1}{2\sinh[(n_s-1)\Delta]}\right]^2 \kappa^{2(n_s-1)}\,
        C_{\rm SI}(n_s)\left(1-e^{-Q_-(s)}\right)\left(1-e^{-Q_+(\epsilon)}\right),

for :math:`0<\kappa<2e^{\Delta}` and zero above, with :math:`\kappa = f/f_\star`,
:math:`s = \kappa e^{\Delta}` and :math:`\epsilon = e^{\Delta}/\kappa - 1/2`.
The edge functions :math:`Q_\pm` reproduce the infrared rise sourced by
:math:`k_- = k_\star e^{-\Delta}` and the quartic UV cutoff at :math:`k = 2k_+`
exactly; only the crossovers are fitted, once, independently of
:math:`\Delta` and :math:`n_s`. The result is redshifted to today with

.. math::

    h^2\Omega_{\mathrm{GW}}(f) = \Omega_{r,0}h^2
        \left(\frac{g_*}{g_{*,0}}\right)
        \left(\frac{g_{*s,0}}{g_{*s}}\right)^{4/3}
        \Omega_{\mathrm{GW},r}(f).

The fit is validated against the exact kernel integral for
:math:`\Delta\in[1,5]` and :math:`n_s\in[0.9,1.1]` (median error ~3%).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates import constants as c
from gwb_templates.scalar_induced_templates.base import ScalarInducedTemplate
from gwb_templates.template import Array

# Moments of the lower-edge kernel expansion (derived).
_I0, _I1, _I2 = 8.0 / 15.0, 0.558953, 0.587776

# Upper-edge coefficient, K = 6400 (5/2 ln(3/2) - 1)^2 (derived).
_K_UV = 6400.0 * (2.5 * math.log(1.5) - 1.0) ** 2

# Crossover constants (s_-, nu_-, A_-, sbar_-, mu_-) of the lower edge and
# (eps_+, nu_+, A_+, epsbar_+, mu_+) of the upper edge (fitted).
_LOWER_EDGE = (0.419, 2.647, 0.102, 1.207, 38.66)
_UPPER_EDGE = (0.531, 1.065, 0.086, 0.409, 15.49)

# Energy d.o.f. today (photons + 3 neutrinos), consistent with c.Omega_R.
_G_STAR_0 = 3.363


def c_si(n_s: Array) -> Array:
    r"""Quadratic surrogate for the plateau amplitude :math:`C_{\rm SI}(n_s)`."""
    x = n_s - 1.0
    return 0.822 + 0.242 * x + 0.616 * x**2


def _x_over_sinh_x(x: Array) -> Array:
    r"""
    :math:`x/\sinh x`, regular at :math:`x = 0` for values *and* derivatives.

    A plain ``jnp.where(x == 0, 1, x / sinh(x))`` gives NaN gradients at
    :math:`n_s = 1`, so the origin is handled with a Taylor series instead.
    """
    small = jnp.abs(x) < 1e-3
    x_safe = jnp.where(small, 1.0, x)
    series = 1.0 - x**2 / 6.0 + 7.0 * x**4 / 360.0
    return jnp.where(small, series, x_safe / jnp.sinh(x_safe))


def omega_gw_r_broken_plateau(kappa: Array, Delta: Array, n_s: Array) -> Array:
    r"""
    Dimensionless :math:`\Omega_{\mathrm{GW},r}/A_\zeta^2` at :math:`\kappa=k/k_\star`.

    Pure JAX, broadcasts over all arguments. Returns zero above the UV cutoff
    :math:`\kappa = 2e^{\Delta}`, with finite gradients everywhere.
    """
    x = n_s - 1.0
    beta = 5.0 - 2.0 * n_s
    norm = _x_over_sinh_x(x * Delta) / (2.0 * Delta)
    csi = c_si(n_s)

    # Lower edge: infrared rise sourced by k_-.
    c0 = 3.0 * _I0 / (2.0 * beta)
    c1 = 6.0 * _I0 / beta**2 - 3.0 * _I1 / beta
    c2 = (
        12.0 * _I0 / beta**3
        - 6.0 * _I1 / beta**2
        + 3.0 * (_I2 + jnp.pi**2 * _I0) / (2.0 * beta)
    )
    s = kappa * jnp.exp(Delta)
    m_s = jnp.log(4.0 / s**2) - 2.0
    f_minus = s**beta * (c0 * m_s**2 + c1 * m_s + c2)
    s_m, nu_m, a_m, sbar_m, mu_m = _LOWER_EDGE
    q_minus = f_minus / csi / (1.0 + (s / s_m) ** nu_m)
    q_minus += a_m * jax.nn.softplus(mu_m * jnp.log(s / sbar_m))

    # Upper edge: quartic cutoff at k = 2 k_+. Outside the support eps is
    # replaced by a dummy value so the masked branch stays NaN-free under grad.
    eps = jnp.exp(Delta) / kappa - 0.5
    support = eps > 0.0
    eps = jnp.where(support, eps, 1.0)
    f_plus = 0.25**x * _K_UV * eps**4
    eps_p, nu_p, a_p, epsbar_p, mu_p = _UPPER_EDGE
    q_plus = f_plus / csi / (1.0 + (eps / eps_p) ** nu_p)
    q_plus += a_p * jax.nn.softplus(mu_p * jnp.log(eps / epsbar_p))

    omega = norm**2 * kappa ** (2.0 * x) * csi
    omega *= (-jnp.expm1(-q_minus)) * (-jnp.expm1(-q_plus))
    return jnp.where(support, omega, 0.0)


class BrokenPlateauSIGW(ScalarInducedTemplate):
    r"""
    SIGW spectrum from a broken-plateau (tilted box) curvature spectrum.

    Free parameters
    ---------------
    log_A_zeta
        :math:`\log_{10}` of the band-integrated curvature amplitude
        :math:`A_\zeta = \int \mathcal{P}_\zeta\,\mathrm{d}\ln k`.
    log_f_star
        :math:`\log_{10}` of the band-centre frequency
        :math:`f_\star = k_\star/2\pi` in Hz.
    Delta
        Half-width :math:`\Delta` of the band in e-folds; the curvature
        spectrum is non-zero for :math:`|\ln(k/k_\star)| < \Delta`.
    n_s
        Tilt of the curvature spectrum inside the band.

    Configuration (constructor)
    ---------------------------
    g_star, g_star_s
        Energy and entropy relativistic d.o.f. at horizon re-entry of
        :math:`k_\star`. Default to the Standard Model value 106.75, which
        holds throughout the LISA and ET bands; ``g_star_s`` defaults to
        ``g_star``.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Chen:2026hnv,
    author = "Chen, Zu-Cheng and Liu, Lang",
    title = "{Scalar-induced gravitational waves from a box-shaped curvature
             power spectrum}",
    eprint = "2607.10730",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "7",
    year = "2026"
}
""".strip(),
    )

    def __init__(
        self,
        *,
        g_star: float = 106.75,
        g_star_s: float | None = None,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.g_star = float(g_star)
        self.g_star_s = self.g_star if g_star_s is None else float(g_star_s)
        #: :math:`h^2\Omega_{\mathrm{GW},0} / \Omega_{\mathrm{GW},r}`.
        self.transfer_today = (
            c.Omega_R
            * c.h**2
            * (self.g_star / _G_STAR_0)
            * (c.g_star_0 / self.g_star_s) ** (4.0 / 3.0)
        )

        default_labels = {
            "log_A_zeta": r"$\log_{10} A_\zeta$",
            "log_f_star": r"$\log_{10}(f_\star/\mathrm{Hz})$",
            "Delta": r"$\Delta$",
            "n_s": r"$n_s$",
        }
        default_priors = {
            "log_A_zeta": {"min": -5.0, "max": -1.0},
            "log_f_star": {"min": -5.0, "max": 3.0},
            "Delta": {"min": 1.0, "max": 5.0},
            "n_s": {"min": 0.85, "max": 1.15},
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Broken-plateau SIGW (RD)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def omega_gw_h2(
        self,
        frequency: Array,
        log_A_zeta: Array,
        log_f_star: Array,
        Delta: Array,
        n_s: Array,
    ) -> Array:
        r"""
        Evaluate :math:`h^2\Omega_{\mathrm{GW}}(f)` today at ``frequency`` (Hz).
        """
        kappa = frequency / 10.0**log_f_star
        omega_r = omega_gw_r_broken_plateau(kappa, Delta, n_s)
        return self.transfer_today * 10.0 ** (2.0 * log_A_zeta) * omega_r
