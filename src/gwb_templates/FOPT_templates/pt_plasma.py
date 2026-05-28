r"""
Combined sound-wave + MHD-turbulence GW spectrum from a cosmological
first-order phase transition.

The turbulence source energy is set to a fraction ``epsilon`` of the bulk
kinetic energy: ``log_Omega_s = log_K + log10(epsilon)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.FOPT_templates.pt_sound_waves import PtSoundWaves
from gwb_templates.FOPT_templates.pt_turbulence import PtTurbulence
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class PtPlasma(AnalyticTemplate):
    r"""
    Combined sound-wave + MHD-turbulence GW spectrum from a cosmological FOPT.

    Free parameters
    ---------------
    log_K
        :math:`\log_{10}` of the fluid kinetic-energy fraction.
    log_R_H_star
        :math:`\log_{10}` of the bubble size times the Hubble rate.
    xi_w
        Bubble wall velocity.
    log_T_star
        :math:`\log_{10}` of the transition temperature in GeV.
    epsilon
        Fraction of bulk kinetic energy that feeds the MHD-turbulence
        source.
    """

    # TODO: cite
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
            "log_K": r"$\log_{10}K$",
            "log_R_H_star": r"$\log_{10}(R_* H_*)$",
            "xi_w": r"$\xi_w$",
            "log_T_star": r"$\log_{10}(T_*/\mathrm{GeV})$",
            "epsilon": r"$\epsilon$",
        }
        default_priors = {
            "log_K": {"min": -4.0, "max": 0.0},
            "log_R_H_star": {"min": -3.0, "max": 0.0},
            "xi_w": {"min": 0.01, "max": 0.99},
            "log_T_star": {"min": -2.0, "max": 4.0},
            "epsilon": {"min": 0.0, "max": 1.0},
        }

        # Sub-template instances; reused on every call. Both are pure JAX,
        # so this is safe to construct once at init time.
        self._sound_waves = PtSoundWaves()
        self._turbulence = PtTurbulence()

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "PT Plasma (SW + Turbulence)"
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
        log_K: ArrayLike,
        log_R_H_star: ArrayLike,
        xi_w: ArrayLike,
        log_T_star: ArrayLike,
        epsilon: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate the combined SW + MHD-turbulence FOPT spectrum."""
        sw = self._sound_waves.omega_gw_h2(
            frequency, log_K, log_R_H_star, xi_w, log_T_star
        )
        log_Omega_s = log_K + jnp.log10(epsilon)
        mhd = self._turbulence.omega_gw_h2(
            frequency, log_Omega_s, log_R_H_star, log_T_star
        )
        return sw + mhd
