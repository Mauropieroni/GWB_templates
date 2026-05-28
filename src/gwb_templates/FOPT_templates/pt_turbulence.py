r"""
MHD-turbulence contribution to the GW spectrum from a cosmological
first-order phase transition.

Based on:

A. Roper Pol, C. Caprini, A. Neronov, and D. Semikoz,
"Gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band",
Phys.Rev.D 105 (2022) 12; [arXiv:2201.05630 [astro-ph.CO]].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.FOPT_templates.pt_base import (
    a_hubble,
    double_broken_power_law,
    redshift_omega,
)
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class PtTurbulence(AnalyticTemplate):
    r"""
    MHD-turbulence GW spectrum from a cosmological FOPT.

    Free parameters
    ---------------
    log_Omega_s
        :math:`\log_{10}` of the fractional energy density in the
        turbulent source.
    log_R_H_star
        :math:`\log_{10}` of the bubble size times the Hubble rate.
    log_T_star
        :math:`\log_{10}` of the transition temperature in GeV.
    """

    #: Spectral amplitude prefactor (Roper Pol et al. 2022).
    AMPLITUDE_PREFACTOR: ClassVar[float] = 0.085
    #: Eddy-turnover scaling N_eddy.
    N_EDDY: ClassVar[float] = 2.0
    #: Fixed spectral exponents (n_1, n_2, n_3, a_1, a_2).
    SPECTRAL_EXPONENTS: ClassVar[tuple[float, float, float, float, float]] = (
        3.0,
        1.0,
        -8.0,
        4.0,
        2.15,
    )

    # TODO: cite (arXiv:2201.05630)
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_Omega_s": r"$\log_{10}\Omega_s$",
            "log_R_H_star": r"$\log_{10}(R_* H_*)$",
            "log_T_star": r"$\log_{10}(T_*/\mathrm{GeV})$",
        }
        default_priors = {
            "log_Omega_s": {"min": -6.0, "max": 0.0},
            "log_R_H_star": {"min": -3.0, "max": 0.0},
            "log_T_star": {"min": -2.0, "max": 4.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "PT MHD Turbulence"
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
        frequency: ArrayLike,
        log_Omega_s: ArrayLike,
        log_R_H_star: ArrayLike,
        log_T_star: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate the MHD-turbulence FOPT spectrum at ``frequency``."""
        Omega_s = 10.0**log_Omega_s
        R_H_star = 10.0**log_R_H_star
        T_star = 10.0**log_T_star

        # Alfven speed
        vA = jnp.sqrt(0.75 * Omega_s)

        aH_star = a_hubble(T_star)
        h2FGW0 = redshift_omega(T_star)

        f_1 = vA / self.N_EDDY * aH_star / R_H_star
        f_2 = 2.2 * aH_star / R_H_star
        r_f = f_2 / f_1  # = 2.2 * N_eddy / vA

        n_1, n_2, n_3, a_1, a_2 = self.SPECTRAL_EXPONENTS
        norm = (
            r_f**3 * (1.0 + r_f**a_1) ** (-2.0 / a_1) * 2.0 ** (-11.0 / 3.0 / a_2)
        )
        prefactor = 3.0 * self.AMPLITUDE_PREFACTOR * norm / (4.0 * jnp.pi**2 * self.N_EDDY)
        h2Omega2 = h2FGW0 * prefactor * vA * Omega_s**2 * R_H_star**2

        return double_broken_power_law(
            frequency,
            jnp.log10(h2Omega2),
            jnp.log10(f_1),
            jnp.log10(f_2),
            n_1,
            n_2,
            n_3,
            a_1,
            a_2,
        )
