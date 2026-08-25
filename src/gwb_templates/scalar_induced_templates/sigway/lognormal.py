r"""
Log-normal peak scalar-induced GW template (SIGWAY, radiation domination).

Mirrors the ``lognormal_rd`` configuration tested in arXiv:2501.11320:

.. math::

    \mathcal{P}_\zeta(k) = \frac{A_s}{\sqrt{2\pi}\,\Delta}
        \exp\!\left[-\frac{\ln^2(k/k_s)}{2\Delta^2}\right],

with :math:`A_s = 10^{\log A_s}`, :math:`\Delta = 10^{\log\Delta}`,
:math:`k_s = 10^{\log k_s}` (:math:`k` in :math:`\mathrm{s}^{-1}`), evolved
through the radiation-domination kernel. The ``"RD"`` kernel normalisation
returns :math:`\Omega_{\mathrm{GW}} h^2` directly.
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


def pzeta_lognormal(
    k: jax.Array,
    logAs: jax.Array,
    logDelta: jax.Array,
    logks: jax.Array,
) -> jax.Array:
    r""":math:`\mathcal{P}_\zeta(k)` for the log-normal peak."""
    As = 10.0**logAs
    Delta = 10.0**logDelta
    ks = 10.0**logks
    return (
        As
        / ((2 * jnp.pi) ** 0.5)
        / Delta
        * jnp.exp(-(1.0 / (2.0 * Delta**2)) * (jnp.log(k / ks)) ** 2)
    )


def t_grid_lognormal(
    k: jax.Array,
    logAs: jax.Array,
    logDelta: jax.Array,
    logks: jax.Array,
) -> jax.Array:
    r"""
    Parameter-dependent :math:`t` quadrature grid (verbatim from the paper run).

    The induced-GW integrand peaks at :math:`t \approx 1`, so the grid is
    **linear** below 1 (to resolve the peak) and **geometric** above 1 (to
    cover the slowly-decaying tail). The upper bound tracks the source width:
    it scales as :math:`e^{4\Delta}\,(2 k_s / k)`, capped at :math:`10^5 k_s`,
    so wider peaks integrate further out in :math:`t`.
    """
    ks = 10.0**logks
    Delta = 10.0**logDelta
    upper = jnp.exp(4 * Delta) * (2 * ks / k)
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-5 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 200)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 600)
    return jnp.concatenate([t1, t2], axis=0)


class SIGWAYLognormal(SIGWAYTemplate):
    r"""
    Log-normal-peak SIGW spectrum (radiation domination), via SIGWAY.

    Free parameters
    ---------------
    logAs
        :math:`\log_{10}` of the peak integrated amplitude :math:`A_s`.
    logDelta
        :math:`\log_{10}` of the dimensionless peak width :math:`\Delta`.
    logks
        :math:`\log_{10}` of the peak wavenumber :math:`k_s` (:math:`\mathrm{s}^{-1}`).

    Configuration (constructor)
    ---------------------------
    s, t, f
        SIGWAY quadrature / internal-frequency grids. Default to the
        paper-validated grids; ``t`` is the parameter-dependent callable
        :func:`t_grid_lognormal`.
    """

    DEFAULT_F: ClassVar[tuple[float, float, int]] = (1e-5, 1e-1, 200)
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
        self._t = t_grid_lognormal if t is None else t
        lo, hi, n = self.DEFAULT_F
        self._f = jnp.geomspace(lo, hi, n) if f is None else f

        default_labels = {
            "logAs": r"$\log_{10} A_s$",
            "logDelta": r"$\log_{10}\Delta$",
            "logks": r"$\log_{10} k_s$",
        }
        default_priors = {
            "logAs": {"min": -4.0, "max": 0.0},
            "logDelta": {"min": -2.0, "max": 0.5},
            "logks": {"min": -4.0, "max": 1.0},
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Log-normal SIGW (RD)"
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
            AnalyticPerturbations(pzeta_lognormal, ("logAs", "logDelta", "logks")),
            RadiationKernel(),
            s=self._s,
            t=self._t,
            f=self._f,
            upsample=True,
        )
