r"""
Early-matter-domination scalar-induced GW template (SIGWAY).

Mirrors the ``emd_imd2rd`` configuration tested in arXiv:2501.11320: a flat
source power spectrum with a sharp ultraviolet cutoff,

.. math::

    \mathcal{P}_\zeta(k) = A_s\,\Theta(k_{\max} - k),

processed through the **instantaneous early-matter-domination → radiation**
kernel (:class:`sigway.kernels.InstantEMDKernel`), whose extra parameter is the
conformal time of the transition :math:`\eta_R`.

Free parameters (SIGWAY order: perturbation, then kernel)
---------------------------------------------------------
``As``    flat source amplitude.
``kmax``  cutoff wavenumber (:math:`\mathrm{s}^{-1}`); a *non-smooth*
          parameter (enters a Heaviside), so its Jacobian column is computed
          by central finite differences inside SIGWAY.
``etaR``  EMD → RD transition conformal time.

.. warning::
   **Normalisation.** :class:`sigway.kernels.InstantEMDKernel` defaults to the
   ``"CT"`` preset (dimensionless :math:`\Omega_{\mathrm{GW}}/\Omega_r`, the
   convention used for the paper figures). To keep this template consistent
   with the ``omega_gw_h2`` contract used everywhere else in
   ``gwb_templates``, the default here is ``norm="RD"`` so the output is
   :math:`\Omega_{\mathrm{GW}} h^2` today (the ``"RD"`` preset multiplies
   ``"CT"`` by :math:`c_g\,\Omega_{r,0} h^2`). Pass ``norm="CT"`` to reproduce
   the paper's dimensionless figures exactly.

.. warning::
   **Cutoff derivatives.** ``kmax`` enters a Heaviside, so its first-derivative
   (Jacobian) column is computed by central finite differences inside SIGWAY —
   that path is correct. The *second* derivative is not: the inherited
   :meth:`~gwb_templates.template.Template.hess_theta_omega_gw_h2` uses pure
   autodiff, which returns zero for a step-function parameter. Use the
   first-order (Fisher / Jacobian) information for ``kmax``; do not rely on the
   Hessian row/column for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from sigway.perturbations import AnalyticPerturbations
from sigway.kernels import InstantEMDKernel
from sigway.spectrum import OmegaGW

from gwb_templates.sigway_templates.base import SIGWAYTemplate


def pzeta_flat_cutoff(
    k: jax.Array,
    As: jax.Array,
    kmax: jax.Array,
) -> jax.Array:
    r""":math:`\mathcal{P}_\zeta(k) = A_s\,\Theta(k_{\max}-k)`."""
    return jnp.heaviside(kmax - k, 1.0) * As


def t_grid_early_matter_domination(
    k: jax.Array,
    As: jax.Array,
    kmax: jax.Array,
    etaR: jax.Array,
) -> jax.Array:
    r"""
    Parameter-dependent :math:`t` quadrature grid (verbatim from the paper run).

    Geometric from :math:`10^{-10}` to :math:`t_{\max} = 2 k_{\max}/k`: the
    cutoff :math:`k_{\max}` bounds the support of the source, so the integration
    domain in :math:`t` closes at :math:`2 k_{\max}/k` rather than extending to
    a fixed large value.
    """
    tmax = 2 * kmax / k
    return jnp.geomspace(1e-10 * jnp.ones_like(k), tmax, 100)


class SIGWAYEarlyMatterDomination(SIGWAYTemplate):
    r"""
    Flat-source + instant-EMD→RD SIGW spectrum, via SIGWAY.

    Free parameters
    ---------------
    As
        Flat source amplitude.
    kmax
        Cutoff wavenumber (:math:`\mathrm{s}^{-1}`); non-smooth (Heaviside),
        differentiated by finite differences in SIGWAY's Jacobian.
    etaR
        Conformal time of the EMD → RD transition.

    Configuration (constructor)
    ---------------------------
    norm
        Kernel normalisation preset. Defaults to ``"RD"`` (output in
        :math:`\Omega_{\mathrm{GW}} h^2`); use ``"CT"`` for the paper's
        dimensionless convention. See the module-level warning.
    s, t, f
        SIGWAY grids; default to the paper-validated values, with the
        parameter-dependent :func:`t_grid_early_matter_domination`.
    """

    DEFAULT_F: ClassVar[tuple[float, float, int]] = (2.1e-9, 5e-2, 350)
    DEFAULT_NS: ClassVar[int] = 100

    def __init__(
        self,
        *,
        norm: str = "RD",
        s: Any = None,
        t: Any = None,
        f: Any = None,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self._norm = norm
        self._s = jnp.linspace(0.0, 1.0, self.DEFAULT_NS) if s is None else s
        self._t = t_grid_early_matter_domination if t is None else t
        lo, hi, n = self.DEFAULT_F
        self._f = jnp.geomspace(lo, hi, n) if f is None else f

        default_labels = {
            "As": r"$A_s$",
            "kmax": r"$k_{\max}$",
            "etaR": r"$\eta_R$",
        }
        default_priors = {
            "As": {"min": 0.0, "max": 1e-6},
            "kmax": {"min": 1e-3, "max": 1.0},
            "etaR": {"min": 1.0, "max": 1e4},
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Early matter domination SIGW"
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
                pzeta_flat_cutoff, ("As", "kmax"), nonsmooth_params=("kmax",)
            ),
            InstantEMDKernel(norm=self._norm),
            s=self._s,
            t=self._t,
            f=self._f,
            upsample=True,
        )
