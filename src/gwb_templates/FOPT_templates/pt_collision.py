r"""
Bubble-collision contribution to the GW spectrum from a cosmological
first-order phase transition.

Based on:

M. Lewicki and V. Vaskonen,
"Gravitational waves from bubble collisions and fluid motion in
strongly supercooled phase transitions",
Eur.Phys.J. C83 (2023) 2, 109; [arXiv:2208.11697 [astro-ph.CO]].

The spectral shape is fixed to the U(1)-symmetric scalar field scenario
(A=0.05, omega_p/beta=0.7, a=b=2.4, c=4.0).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.FOPT_templates.pt_base import (
    a_hubble,
    broken_power_law_a1,
    jac_broken_power_law_a1_amp_freq,
    redshift_omega,
)
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


def _build_collision_constants() -> tuple[float, float, float, float, float]:
    """Compute the fixed (A_b, omega_b/beta, n_1, n_2, a_1) constants once."""
    a_slope = 2.4
    b_slope = 2.4
    c_sharp = 4.0
    a_raw = 0.05
    omega_p_over_beta = 0.7

    a_b = (
        a_raw
        * (
            0.5 * (a_slope / b_slope) ** (b_slope / (a_slope + b_slope))
            + 0.5 * (b_slope / a_slope) ** (a_slope / (a_slope + b_slope))
        )
        ** c_sharp
    )
    omega_b_over_beta = omega_p_over_beta * (b_slope / a_slope) ** (
        c_sharp / (a_slope + b_slope)
    )
    n_1 = a_slope
    n_2 = -b_slope
    a_1 = (a_slope + b_slope) / c_sharp
    return a_b, omega_b_over_beta, n_1, n_2, a_1


_A_B, _OMEGA_B_OVER_BETA, _N_1, _N_2, _A_1 = _build_collision_constants()


class PtCollision(AnalyticTemplate):
    r"""
    Bubble-collision GW spectrum from a cosmological FOPT.

    Free parameters
    ---------------
    log_K_tilde
        :math:`\log_{10}` of the vacuum-to-kinetic-energy conversion
        fraction.
    log_beta_over_H
        :math:`\log_{10}` of :math:`\beta/H_*` (inverse transition
        duration in Hubble units).
    log_T_star
        :math:`\log_{10}` of the transition temperature in GeV.
    """

    #: Fixed spectral-shape constants (U(1)-symmetric scalar scenario).
    A_B: ClassVar[float] = _A_B
    OMEGA_B_OVER_BETA: ClassVar[float] = _OMEGA_B_OVER_BETA
    SPECTRAL_INDEX_IR: ClassVar[float] = _N_1
    SPECTRAL_INDEX_UV: ClassVar[float] = _N_2
    TRANSITION_SMOOTHNESS: ClassVar[float] = _A_1

    # TODO: cite (arXiv:2208.11697)
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
            "log_K_tilde": r"$\log_{10}\tilde{K}$",
            "log_beta_over_H": r"$\log_{10}(\beta/H_*)$",
            "log_T_star": r"$\log_{10}(T_*/\mathrm{GeV})$",
        }
        default_priors = {
            "log_K_tilde": {"min": -4.0, "max": 0.0},
            "log_beta_over_H": {"min": 0.0, "max": 4.0},
            "log_T_star": {"min": -2.0, "max": 4.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "PT Bubble Collisions"
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
        log_K_tilde: ArrayLike,
        log_beta_over_H: ArrayLike,
        log_T_star: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate the bubble-collision FOPT spectrum at ``frequency``."""
        K_tilde = 10.0**log_K_tilde
        beta_over_H = 10.0**log_beta_over_H
        T_star = 10.0**log_T_star

        h2FGW0 = redshift_omega(T_star)
        h2Omega_b = h2FGW0 * self.A_B * K_tilde**2 / beta_over_H**2

        aH_star = a_hubble(T_star)
        f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * self.OMEGA_B_OVER_BETA

        return broken_power_law_a1(
            frequency,
            jnp.log10(h2Omega_b),
            jnp.log10(f_b),
            self.SPECTRAL_INDEX_IR,
            self.SPECTRAL_INDEX_UV,
            self.TRANSITION_SMOOTHNESS,
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the bubble-collision FOPT spectrum w.r.t.
        ``(log_K_tilde, log_beta_over_H, log_T_star)``.

        Uses the chain rule through :func:`broken_power_law_a1`.  The
        :math:`g_*` tables are piecewise-constant, so their T-derivatives
        vanish almost everywhere (consistent with autodiff through
        :func:`jnp.select`).
        """
        log_K_tilde, log_beta_over_H, log_T_star = theta[0], theta[1], theta[2]

        K_tilde = 10.0**log_K_tilde
        beta_over_H = 10.0**log_beta_over_H
        T_star = 10.0**log_T_star

        h2FGW0 = redshift_omega(T_star)
        h2Omega_b = h2FGW0 * self.A_B * K_tilde**2 / beta_over_H**2
        aH_star = a_hubble(T_star)
        f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * self.OMEGA_B_OVER_BETA

        # Partials of broken_power_law_a1 w.r.t. (log_amplitude, log_f_b):
        # shape (..., 2)
        J_inner = jac_broken_power_law_a1_amp_freq(
            frequency,
            jnp.log10(h2Omega_b),
            jnp.log10(f_b),
            self.SPECTRAL_INDEX_IR,
            self.SPECTRAL_INDEX_UV,
            self.TRANSITION_SMOOTHNESS,
        )

        # d([log_h2Omega_b, log_f_b])
        #     / d([log_K_tilde, log_beta_over_H, log_T_star])
        # h2Omega_b ∝ K_tilde^2 / beta_over_H^2  →  [2, -2, 0]
        # f_b ∝ a_hubble(T) * beta_over_H,  a_hubble ∝ T  →  [0, 1, 1]
        dq_dp = jnp.array([[2.0, -2.0, 0.0], [0.0, 1.0, 1.0]])

        return J_inner @ dq_dp
