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

Also see: R. Jinno and M. Takimoto, "Gravitational waves from bubble
collisions: An analytic derivation", Phys.Rev.D 95 (2017) 024009;
[arXiv:1605.01403 [astro-ph.CO]] (original analytic bubble-collision
derivation).
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

    Configuration
    -------------
    a_b
        Numerical factor :math:`A_b` in the amplitude at the break frequency. Defaults
        to `PtCollision.DEFAULT_A_B`.
    omega_b_over_beta
        Angular frequency of the break, :math:`\omega_b = 2 \pi f_b` in units of the
        inverse transition duration :math:`\beta`. Defaults to
        `PtCollision.DEFAULT_OMEGA_B_OVER_BETA`.
    spectral_index_IR
        Low-frequency spectral index. Defaults to
        `PtCollision.DEFAULT_SPECTRAL_INDEX_IR`.
    spectral_index_UV
        High-frequency spectral index. Defaults to
        `PtCollision.DEFAULT_SPECTRAL_INDEX_UV`.
    transition_smoothness
        Smoothness of the transition between the two spectral slopes. Defaults to
        `PtCollision.DEFAULT_TRANSITION_SMOOTHNESS`.
    """

    #: Defaults for fixed spectral-shape constants (U(1)-symmetric scalar scenario).
    DEFAULT_A_B: ClassVar[float] = _A_B
    DEFAULT_OMEGA_B_OVER_BETA: ClassVar[float] = _OMEGA_B_OVER_BETA
    DEFAULT_SPECTRAL_INDEX_IR: ClassVar[float] = _N_1
    DEFAULT_SPECTRAL_INDEX_UV: ClassVar[float] = _N_2
    DEFAULT_TRANSITION_SMOOTHNESS: ClassVar[float] = _A_1

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Lewicki:2022pdb,
    author = "Lewicki, Marek and Vaskonen, Ville",
    title = "{Gravitational waves from bubble collisions and fluid motion in strongly
        supercooled phase transitions}",
    eprint = "2208.11697",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1140/epjc/s10052-023-11241-3",
    journal = "Eur. Phys. J. C",
    volume = "83",
    number = "2",
    pages = "109",
    year = "2023"
}
""",
        r"""
@article{Jinno:2016vai,
    author = "Jinno, Ryusuke and Takimoto, Masahiro",
    title = "{Gravitational waves from bubble collisions: An analytic derivation}",
    eprint = "1605.01403",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "KEK-TH-1900",
    doi = "10.1103/PhysRevD.95.024009",
    journal = "Phys. Rev. D",
    volume = "95",
    number = "2",
    pages = "024009",
    year = "2017"
}
""",
    )

    def __init__(
        self,
        a_b: float = DEFAULT_A_B,
        omega_b_over_beta: float = DEFAULT_OMEGA_B_OVER_BETA,
        spectral_index_IR: float = DEFAULT_SPECTRAL_INDEX_IR,
        spectral_index_UV: float = DEFAULT_SPECTRAL_INDEX_UV,
        transition_smoothness: float = DEFAULT_TRANSITION_SMOOTHNESS,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.a_b: float = float(a_b)
        self.omega_b_over_beta: float = float(omega_b_over_beta)
        self.spectral_index_IR: float = float(spectral_index_IR)
        self.spectral_index_UV: float = float(spectral_index_UV)
        self.transition_smoothness: float = float(transition_smoothness)

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
        h2Omega_b = h2FGW0 * self.a_b * K_tilde**2 / beta_over_H**2

        aH_star = a_hubble(T_star)
        f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * self.omega_b_over_beta

        return broken_power_law_a1(
            frequency,
            jnp.log10(h2Omega_b),
            jnp.log10(f_b),
            self.spectral_index_IR,
            self.spectral_index_UV,
            self.transition_smoothness,
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
        h2Omega_b = h2FGW0 * self.a_b * K_tilde**2 / beta_over_H**2
        aH_star = a_hubble(T_star)
        f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * self.omega_b_over_beta

        # Partials of broken_power_law_a1 w.r.t. (log_amplitude, log_f_b):
        # shape (..., 2)
        J_inner = jac_broken_power_law_a1_amp_freq(
            frequency,
            jnp.log10(h2Omega_b),
            jnp.log10(f_b),
            self.spectral_index_IR,
            self.spectral_index_UV,
            self.transition_smoothness,
        )

        # d([log_h2Omega_b, log_f_b])
        #     / d([log_K_tilde, log_beta_over_H, log_T_star])
        # h2Omega_b ∝ K_tilde^2 / beta_over_H^2  →  [2, -2, 0]
        # f_b ∝ a_hubble(T) * beta_over_H,  a_hubble ∝ T  →  [0, 1, 1]
        dq_dp = jnp.array([[2.0, -2.0, 0.0], [0.0, 1.0, 1.0]])

        return J_inner @ dq_dp
