r"""
Broken-power-law peak scalar-induced GW template (SIGWAY, radiation domination).

Mirrors the ``bpl_rd`` configuration tested in arXiv:2501.11320:

.. math::

    \mathcal{P}_\zeta(k) = A\,
        \frac{(\alpha+\beta)^\gamma}
             {\bigl[\beta\,(k/k_s)^{-\alpha/\gamma}
                    + \alpha\,(k/k_s)^{\beta/\gamma}\bigr]^\gamma},

with :math:`A = 10^{\log A}`, :math:`k_s = 10^{\log k_s}`
(:math:`k` in :math:`\mathrm{s}^{-1}`). :math:`\alpha` is the low-frequency
slope, :math:`\beta` the high-frequency fall-off, and :math:`\gamma` the
smoothness of the break. Evolved through the radiation-domination kernel
(``"RD"`` norm → :math:`\Omega_{\mathrm{GW}} h^2`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from sigway.perturbations import AnalyticPerturbations
from sigway.kernels import RadiationKernel
from sigway.spectrum import OmegaGW

from gwb_templates.scalar_induced_templates.sigway.base import SIGWAYTemplate


def pzeta_broken_power_law(
    k: jax.Array,
    logA: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    logks: jax.Array,
) -> jax.Array:
    r""":math:`\mathcal{P}_\zeta(k)` for the broken power law."""
    A = 10.0**logA
    ks = 10.0**logks
    return (
        A
        * (alpha + beta) ** gamma
        / (beta * (k / ks) ** (-alpha / gamma) + alpha * (k / ks) ** (beta / gamma))
        ** gamma
    )


def t_grid_broken_power_law(
    k: jax.Array,
    logA: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    logks: jax.Array,
) -> jax.Array:
    r"""
    Parameter-dependent :math:`t` quadrature grid (verbatim from the paper run).

    Linear below the integrand peak (:math:`t \lesssim 1`), geometric above.
    The upper bound is set by the high-frequency slope :math:`\beta`: the BPL
    tail decays as :math:`(k/k_s)^{-\beta}`, so the grid extends to
    :math:`\min(10^{6/\beta}, 10^4)/(k/k_s)`, capped at :math:`10^5 k_s`.
    """
    ks = 10.0**logks
    upper = jnp.min(jnp.array([10 ** (6.0 / beta) / (k / ks), 1e4 / (k / ks)]), axis=0)
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-6 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 100)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 300)
    return jnp.concatenate([t1, t2], axis=0)


class SIGWAYBrokenPowerLaw(SIGWAYTemplate):
    r"""
    Broken-power-law-peak SIGW spectrum (radiation domination), via SIGWAY.

    Free parameters
    ---------------
    logA
        :math:`\log_{10}` of the peak amplitude :math:`A`.
    alpha
        Low-frequency (rising) slope.
    beta
        High-frequency (falling) slope.
    gamma
        Break smoothness.
    logks
        :math:`\log_{10}` of the break wavenumber :math:`k_s`
        (:math:`\mathrm{s}^{-1}`).

    Configuration (constructor)
    ---------------------------
    s, t, f
        SIGWAY grids; default to the paper-validated values, with the
        parameter-dependent :func:`t_grid_broken_power_law`.
    """

    DEFAULT_F: ClassVar[tuple[float, float, int]] = (1e-5, 1.0, 200)
    DEFAULT_NS: ClassVar[int] = 10

    def __init__(
        self,
        *,
        s: Any = None,
        t: Any = None,
        f: Any = None,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self._s = jnp.linspace(0.0, 1.0, self.DEFAULT_NS) if s is None else s
        self._t = t_grid_broken_power_law if t is None else t
        lo, hi, n = self.DEFAULT_F
        self._f = jnp.geomspace(lo, hi, n) if f is None else f

        default_labels = {
            "logA": r"$\log_{10} A$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "gamma": r"$\gamma$",
            "logks": r"$\log_{10} k_s$",
        }
        default_priors = {
            "logA": {"min": -4.0, "max": 0.0},
            "alpha": {"min": 0.0, "max": 6.0},
            "beta": {"min": 0.0, "max": 6.0},
            "gamma": {"min": 0.1, "max": 4.0},
            "logks": {"min": -4.0, "max": 1.0},
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Broken power law SIGW (RD)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def build_model(self) -> OmegaGW:
        return OmegaGW(
            AnalyticPerturbations(
                pzeta_broken_power_law,
                ("logA", "alpha", "beta", "gamma", "logks"),
            ),
            RadiationKernel(),
            s=self._s,
            t=self._t,
            f=self._f,
            upsample=True,
        )
