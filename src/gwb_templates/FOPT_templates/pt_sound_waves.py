r"""
Sound-wave contribution to the GW spectrum from a cosmological first-order
phase transition.

Based on:

R. Jinno, T. Konstandin, H. Rubira and I. Stomberg,
"Higgsless simulations of cosmological phase transitions and gravitational
waves",
JCAP 02 (2023), 011; [arXiv:2209.04369 [astro-ph.CO]].

Also see: M. Hindmarsh, S.J. Huber, K. Rummukainen and D.J. Weir,
"Shape of the acoustic gravitational wave power spectrum from a first
order phase transition", Phys.Rev.D 96 (2017) 103520; [arXiv:1704.05871
[astro-ph.CO]] (original double-broken-power-law acoustic shape).
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
    jac_double_broken_power_law_amp_freqs,
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

    Configuration
    -------------
    amplitude_prefactor
        Numerical factor in the amplitude. Defaults to
        `PtSoundWaves.DEFAULT_AMPLITUDE_PREFACTOR`.
    spectral_index_low_f
        Low-frequency spectral index. Defaults to
        `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[0]`.
    spectral_index_mid_f
        Intermediate-frequency spectral index. Defaults to
        `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[1]`.
    spectral_index_high_f
        High-frequency spectral index. Defaults to
        `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[2]`.
    transition_smoothness_low_f
        Smoothness of the transition between the low and intermediate frequency
        spectral slopes. Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[3]`.
    transition_smoothness_high_f
        Smoothness of the transition between the intermediate and high frequency
        spectral slopes. Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[4]`.
    """

    #: Default spectral amplitude prefactor (Jinno et al. 2023).
    DEFAULT_AMPLITUDE_PREFACTOR: ClassVar[float] = 0.11
    #: Sound speed in the relativistic plasma (1/sqrt(3)).
    SOUND_SPEED: ClassVar[float] = 0.5773502691896258
    #: Default values for fixed spectral exponents (n_1, n_2, n_3, a_1, a_2).
    DEFAULT_SPECTRAL_EXPONENTS: ClassVar[tuple[float, float, float, float, float]] = (
        3.0,
        1.0,
        -3.0,
        2.0,
        4.0,
    )

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Jinno:2022mie,
    author = "Jinno, Ryusuke and Konstandin, Thomas and Rubira, Henrique and Stomberg,
        Isak",
    title = "{Higgsless simulations of cosmological phase transitions and gravitational
        waves}",
    eprint = "2209.04369",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESY 22-148, IFT-UAM/CSIC-22-100, TUM-HEP-1416/22",
    doi = "10.1088/1475-7516/2023/02/011",
    journal = "JCAP",
    volume = "02",
    pages = "011",
    year = "2023"
}
""",
        r"""
@article{Ellis:2020awk,
    author = "Ellis, John and Lewicki, Marek and No, José Miguel",
    title = "{Gravitational waves from first-order cosmological phase transitions:
        lifetime of the sound wave source}",
    eprint = "2003.07360",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "KCL-PH-TH/2020-04, CERN-TH-2020-016, IFT-UAM/CSIC-20-35",
    doi = "10.1088/1475-7516/2020/07/050",
    journal = "JCAP",
    volume = "07",
    pages = "050",
    year = "2020"
}
""",
        r"""
@article{Hindmarsh:2017gnf,
    author = "Hindmarsh, Mark and Huber, Stephan J. and Rummukainen, Kari and Weir,
        David J.",
    title = "{Shape of the acoustic gravitational wave power spectrum from a first order
        phase transition}",
    eprint = "1704.05871",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "HIP-2017-02-TH, HIP-2017-02/TH",
    doi = "10.1103/PhysRevD.96.103520",
    journal = "Phys. Rev. D",
    volume = "96",
    number = "10",
    pages = "103520",
    year = "2017",
    note = "[Erratum: Phys.Rev.D 101, 089902 (2020)]"
}
""",
    )

    def __init__(
        self,
        amplitude_prefactor: float = DEFAULT_AMPLITUDE_PREFACTOR,
        spectral_index_low_f: float = DEFAULT_SPECTRAL_EXPONENTS[0],
        spectral_index_mid_f: float = DEFAULT_SPECTRAL_EXPONENTS[1],
        spectral_index_high_f: float = DEFAULT_SPECTRAL_EXPONENTS[2],
        transition_smoothness_low_f: float = DEFAULT_SPECTRAL_EXPONENTS[3],
        transition_smoothness_high_f: float = DEFAULT_SPECTRAL_EXPONENTS[4],
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.amplitude_prefactor: float = float(amplitude_prefactor)
        self.spectral_exponents: tuple[float, float, float, float, float] = (
            float(spectral_index_low_f),
            float(spectral_index_mid_f),
            float(spectral_index_high_f),
            float(transition_smoothness_low_f),
            float(transition_smoothness_high_f),
        )
        
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

        h2Omega2 = norm * h2FGW0 * self.amplitude_prefactor * K**2 * H_tau * R_H_star

        n_1, n_2, n_3, a_1, a_2 = self.spectral_exponents
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

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the sound-wave FOPT spectrum w.r.t.
        ``(log_K, log_R_H_star, xi_w, log_T_star)``.

        Uses the chain rule through :func:`double_broken_power_law`.  The
        :math:`g_*` tables are piecewise-constant, so their T-derivatives
        vanish almost everywhere (consistent with autodiff through
        :func:`jnp.select`).
        """
        log_K, log_R_H_star, xi_w, log_T_star = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
        )

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
        h2Omega2 = (
            norm * h2FGW0 * self.amplitude_prefactor * K**2 * H_tau * R_H_star
        )

        n_1, n_2, n_3, a_1, a_2 = self.spectral_exponents
        J_inner = jac_double_broken_power_law_amp_freqs(
            frequency,
            jnp.log10(h2Omega2),
            jnp.log10(f_1),
            jnp.log10(f_2),
            n_1,
            n_2,
            n_3,
            a_1,
            a_2,
        )  # shape (..., 3)

        ln10 = jnp.log(10.0)

        # d(log_h2Omega2)/d(log_K), d(log_h2Omega2)/d(log_R_H_star):
        # H_tau = min(1, R/sqrt(0.75*K))
        # When H_tau < 1: K^2*H_tau*R ∝ K^{1.5}*R^2  →  1.5, 2
        # When H_tau == 1: K^2*H_tau*R ∝ K^2*R       →  2.0, 1
        d_logOmega2_d_logK = jnp.where(H_tau < 1.0, 1.5, 2.0)
        d_logOmega2_d_logR = jnp.where(H_tau < 1.0, 2.0, 1.0)
        d_norm_d_rf = 2.0 * (1.0 - r_f**2) / (jnp.pi * (1.0 + r_f**2) ** 2)
        d_rf_d_xiw = -2.5 * c_s * jnp.sign(xi_w - c_s) / xi_shell**2
        d_logOmega2_d_xiw = d_norm_d_rf * d_rf_d_xiw / (norm * ln10)

        d_logf2_d_xiw = -c_s * jnp.sign(xi_w - c_s) / (xi_shell * xi_bubble * ln10)

        dq_dp = jnp.array(
            [
                [
                    d_logOmega2_d_logK,
                    d_logOmega2_d_logR,
                    d_logOmega2_d_xiw,
                    0.0,
                ],  # log_h2Omega2
                [0.0, -1.0, 0.0, 1.0],  # log_f_1
                [0.0, -1.0, d_logf2_d_xiw, 1.0],  # log_f_2
            ]
        )

        return J_inner @ dq_dp
