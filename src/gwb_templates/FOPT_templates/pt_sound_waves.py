r"""
Sound-wave contribution to the GW spectrum from a cosmological first-order
phase transition.

Based on:

R. Jinno, T. Konstandin, H. Rubira and I. Stomberg,
"Higgsless simulations of cosmological phase transitions and gravitational
waves",
JCAP 02 (2023), 011; [arXiv:2209.04369 [astro-ph.CO]].
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
    h_star_tau,
    redshift_omega,
)
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class PtSoundWaves(AnalyticTemplate):
    r"""
    Sound-wave GW spectrum from a cosmological first-order phase transition.

    Free parameters
    ---------------
    log_K
        :math:`\log_{10}` of the fluid kinetic-energy fraction.
    log_R_H_star
        :math:`\log_{10}` of the bubble size times the Hubble rate.
    xi_w
        Bubble wall velocity.
    log_T_star
        :math:`\log_{10}` of the transition temperature (in GeV).
    """

    #: Spectral-shape amplitude prefactor (Jinno et al. 2023).
    AMPLITUDE_PREFACTOR: ClassVar[float] = 0.11
    #: Sound speed in the relativistic plasma (1/sqrt(3)).
    SOUND_SPEED: ClassVar[float] = 0.5773502691896258
    #: Fixed spectral exponents (n_1, n_2, n_3, a_1, a_2).
    SPECTRAL_EXPONENTS: ClassVar[tuple[float, float, float, float, float]] = (
        3.0,
        1.0,
        -3.0,
        2.0,
        4.0,
    )

    # TODO: cite (arXiv:2209.04369)
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
        }
        default_priors = {
            "log_K": {"min": -4.0, "max": 0.0},
            "log_R_H_star": {"min": -3.0, "max": 0.0},
            "xi_w": {"min": 0.01, "max": 0.99},
            "log_T_star": {"min": -2.0, "max": 4.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "PT Sound Waves",
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
    ) -> jax.Array:
        r"""Evaluate the sound-wave FOPT spectrum at ``frequency``."""
        K = 10.0**log_K
        R_H_star = 10.0**log_R_H_star
        T_star = 10.0**log_T_star

        c_s = self.SOUND_SPEED
        xi_shell = jnp.abs(xi_w - c_s)
        xi_bubble = jnp.maximum(xi_w, c_s)

        aH_star = a_hubble(T_star)
        h2FGW0 = redshift_omega(T_star)
        H_tau = h_star_tau(K, R_H_star)

        f_1 = 0.2 * aH_star / R_H_star
        f_2 = 0.5 * aH_star / R_H_star * xi_bubble / xi_shell

        r_f = 2.5 * xi_bubble / xi_shell
        norm = (jnp.sqrt(2.0) + 2.0 * r_f / (1.0 + r_f**2)) / jnp.pi

        h2Omega2 = norm * h2FGW0 * self.AMPLITUDE_PREFACTOR * K**2 * H_tau * R_H_star

        n_1, n_2, n_3, a_1, a_2 = self.SPECTRAL_EXPONENTS
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
