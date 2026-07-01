r"""
Oscillatory-feature scalar-induced GW template (SIGWAY, radiation domination).

Mirrors the ``osc_multifield_rd`` configuration tested in arXiv:2501.11320 — a
peak with super-imposed oscillations from a sharp feature / multifield
transition during inflation. With :math:`\kappa = k/k_s`, :math:`P_0 = 10^{\log
A}`, the smooth envelope is

.. math::

    P_{\mathrm{env}}(k) = \frac{\kappa\,P_0\,e^{-2\eta_L\delta}\,
        e^{2\sqrt{(2-\kappa)\kappa}\,\eta_L\delta}}{4\,(2-\kappa)\,\kappa},

and the oscillatory spectrum is

.. math::

    P_{\mathrm{osc}}(k) = \frac{P_{\mathrm{env}}}{\kappa}
      \Bigl[1 + (\kappa-1)\cos\bigl(2 e^{-\delta/2}\eta_L\kappa\bigr)
        + \sqrt{(2-\kappa)\kappa}\,\sin\bigl(2 e^{-\delta/2}\eta_L\kappa\bigr)\Bigr]
      \,\Theta(2-\kappa),

interpolated by the mixing fraction :math:`F`:
:math:`\mathcal{P}_\zeta = (1-F)\,P_{\mathrm{env}} + F\,P_{\mathrm{osc}}`.
Evolved through the radiation-domination kernel (``"RD"`` norm →
:math:`\Omega_{\mathrm{GW}} h^2`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from sigway.perturbations import AnalyticPerturbations
from sigway.kernels import RadiationKernel
from sigway.spectrum import OmegaGW

from gwb_templates.scalar_induced_templates.base import SIGWAYTemplate


def pzeta_multifield_oscillations(
    k: jax.Array,
    log10A: jax.Array,
    log10ks: jax.Array,
    delta: jax.Array,
    eta_L: jax.Array,
    F: jax.Array,
) -> jax.Array:
    r""":math:`\mathcal{P}_\zeta(k)` for the oscillatory feature."""
    kappa = k / 10**log10ks
    P0 = 10**log10A
    p_env = (
        kappa
        * P0
        * jnp.exp(-2.0 * eta_L * delta)
        * jnp.exp(2.0 * jnp.sqrt((2.0 - kappa) * kappa) * eta_L * delta)
        / (4.0 * (2.0 - kappa) * kappa)
    )
    penv_kappa = p_env / kappa
    cos_term = jnp.cos(2.0 * jnp.exp(-delta / 2.0) * eta_L * kappa)
    sin_term = jnp.sqrt((2.0 - kappa) * kappa) * jnp.sin(
        2.0 * jnp.exp(-delta / 2.0) * eta_L * kappa
    )
    pz = (
        penv_kappa
        * (1.0 + (kappa - 1.0) * cos_term + sin_term)
        * jnp.heaviside(2.0 - kappa, 1.0)
    )
    p_env = jnp.nan_to_num(p_env, nan=0.0)
    pz = jnp.nan_to_num(pz, nan=0.0)
    return (1 - F) * p_env + F * pz


def t_grid_multifield_oscillations(
    k: jax.Array,
    log10A: jax.Array,
    log10ks: jax.Array,
    delta: jax.Array,
    eta_L: jax.Array,
    F: jax.Array,
) -> jax.Array:
    r"""
    Parameter-dependent :math:`t` quadrature grid (verbatim from the paper run).

    Linear below the integrand peak, geometric above. The upper bound scales
    with the feature sharpness :math:`\delta` as
    :math:`e^{5\delta}\,(2 k_s/k)`, capped at :math:`10^5 k_s`.
    """
    ks = 10.0**log10ks
    upper = jnp.exp(5 * delta) * (2 * ks / k)
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-5 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 100)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 300)
    return jnp.concatenate([t1, t2], axis=0)


class SIGWAYMultifieldOscillations(SIGWAYTemplate):
    r"""
    Oscillatory-feature SIGW spectrum (radiation domination), via SIGWAY.

    Free parameters
    ---------------
    log10A
        :math:`\log_{10}` of the amplitude :math:`P_0`.
    log10ks
        :math:`\log_{10}` of the feature wavenumber :math:`k_s`
        (:math:`\mathrm{s}^{-1}`).
    delta
        Feature sharpness :math:`\delta`.
    eta_L
        Phase / location parameter :math:`\eta_L` controlling the oscillation
        frequency.
    F
        Mixing fraction between the smooth envelope (:math:`F=0`) and the fully
        oscillatory spectrum (:math:`F=1`).

    Configuration (constructor)
    ---------------------------
    s, t, f
        SIGWAY grids; default to the paper-validated values, with the
        parameter-dependent :func:`t_grid_multifield_oscillations`.
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
        self._t = t_grid_multifield_oscillations if t is None else t
        lo, hi, n = self.DEFAULT_F
        self._f = jnp.geomspace(lo, hi, n) if f is None else f

        default_labels = {
            "log10A": r"$\log_{10} A$",
            "log10ks": r"$\log_{10} k_s$",
            "delta": r"$\delta$",
            "eta_L": r"$\eta_L$",
            "F": r"$F$",
        }
        default_priors = {
            "log10A": {"min": -4.0, "max": 0.0},
            "log10ks": {"min": -4.0, "max": 1.0},
            "delta": {"min": 0.0, "max": 2.0},
            "eta_L": {"min": 0.0, "max": 50.0},
            "F": {"min": 0.0, "max": 1.0},
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Multifield oscillations SIGW (RD)"
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
                pzeta_multifield_oscillations,
                ("log10A", "log10ks", "delta", "eta_L", "F"),
            ),
            RadiationKernel(),
            s=self._s,
            t=self._t,
            f=self._f,
            upsample=True,
        )
