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

    Configuration
    -------------
    amplitude_prefactor_sw
        Numerical factor in the amplitude of the sound wave contribution. 
        Defaults to `PtSoundWaves.DEFAULT_AMPLITUDE_PREFACTOR`.
    spectral_index_low_f_sw
        Low-frequency spectral index of the sound wave contribution.
        Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[0]`.
    spectral_index_mid_f_sw
        Intermediate-frequency spectral index of the sound wave contribution.
        Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[1]`.
    spectral_index_high_f_sw
        High-frequency spectral index of the sound wave contribution.
        Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[2]`.
    transition_smoothness_low_f_sw
        Smoothness of the transition between the low and intermediate frequency
        spectral slopes of the sound wave contribution.
        Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[3]`.
    transition_smoothness_high_f_sw
        Smoothness of the transition between the intermediate and high frequency
        spectral slopes of the sound wave contribution.
        Defaults to `PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[4]`.
    amplitude_prefactor_turb
        Numerical factor in the amplitude of the turbulence contribution. 
        Defaults to `PtTurbulence.DEFAULT_AMPLITUDE_PREFACTOR`.
    spectral_index_low_f_turb
        Low-frequency spectral index of the turbulence contribution.
        Defaults to `PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[0]`.
    spectral_index_mid_f_turb
        Intermediate-frequency spectral index of the turbulence contribution.
        Defaults to `PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[1]`.
    spectral_index_high_f_turb
        High-frequency spectral index of the turbulencee contribution.
        Defaults to `PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[2]`.
    transition_smoothness_low_f_turb
        Smoothness of the transition between the low and intermediate frequency
        spectral slopes of the turbulence contribution.
        Defaults to `PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[3]`.
    transition_smoothness_high_f_turb
        Smoothness of the transition between the intermediate and high frequency
        spectral slopes of the turbulence contribution.
        Defaults to `PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[4]`.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2024hue,
    author = "Caprini, Chiara and Jinno, Ryusuke and Lewicki, Marek and Madge, Eric
        and Merchand, Marco and Nardini, Germano and Pieroni, Mauro and Roper Pol,
        Alberto and Vaskonen, Ville",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from first-order phase transitions in LISA:
        reconstruction pipeline and physics interpretation}",
    eprint = "2403.03723",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-01, CERN-TH-2024-029",
    doi = "10.1088/1475-7516/2024/10/020",
    journal = "JCAP",
    volume = "10",
    pages = "020",
    year = "2024"
}
""",
        r"""
@article{Caprini:2019egz,
    author = "Caprini, Chiara and others",
    title = "{Detecting gravitational waves from cosmological phase transitions with LISA: an update}",
    eprint = "1910.13125",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESY-19-159, IPPP/19/27, HIP-2019-14/TH, MITP/19-066, IFT-UAM/CSIC-19-139",
    doi = "10.1088/1475-7516/2020/03/024",
    journal = "JCAP",
    volume = "03",
    pages = "024",
    year = "2020"
}
""",
        r"""
@article{Caprini:2015zlo,
    author = "Caprini, Chiara and others",
    title = "{Science with the space-based interferometer eLISA. II: Gravitational waves from cosmological phase transitions}",
    eprint = "1512.06239",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESY-15-246",
    doi = "10.1088/1475-7516/2016/04/001",
    journal = "JCAP",
    volume = "04",
    pages = "001",
    year = "2016"
}
""",
    )

    def __init__(
        self,
        amplitude_prefactor_sw: float = PtSoundWaves.DEFAULT_AMPLITUDE_PREFACTOR,
        spectral_index_low_f_sw: float = PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[0],
        spectral_index_mid_f_sw: float = PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[1],
        spectral_index_high_f_sw: float = PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[2],
        transition_smoothness_low_f_sw: float =
        PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[3],
        transition_smoothness_high_f_sw: float =
        PtSoundWaves.DEFAULT_SPECTRAL_EXPONENTS[4],
        amplitude_prefactor_turb: float = PtTurbulence.DEFAULT_AMPLITUDE_PREFACTOR,
        spectral_index_low_f_turb: float = PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[0],
        spectral_index_mid_f_turb: float = PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[1],
        spectral_index_high_f_turb: float = PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[2],
        transition_smoothness_low_f_turb: float =
        PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[3],
        transition_smoothness_high_f_turb: float =
        PtTurbulence.DEFAULT_SPECTRAL_EXPONENTS[4],
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
        self._sound_waves = PtSoundWaves(
            float(amplitude_prefactor_sw),
            float(spectral_index_low_f_sw),
            float(spectral_index_mid_f_sw),
            float(spectral_index_high_f_sw),
            float(transition_smoothness_low_f_sw),
            float(transition_smoothness_high_f_sw) 
        )
        self._turbulence = PtTurbulence(
            float(amplitude_prefactor_turb),
            float(spectral_index_low_f_turb),
            float(spectral_index_mid_f_turb),
            float(spectral_index_high_f_turb),
            float(transition_smoothness_low_f_turb),
            float(transition_smoothness_high_f_turb) 
        )

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

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the combined SW + MHD-turbulence FOPT spectrum
        w.r.t. ``(log_K, log_R_H_star, xi_w, log_T_star, epsilon)``.

        Composes the analytic Jacobians of the held :class:`PtSoundWaves`
        and :class:`PtTurbulence` sub-templates via the chain rule.
        """
        log_K, log_R_H_star, xi_w, log_T_star, epsilon = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
        )

        # Sound-wave Jacobian (..., 4) for [log_K, log_R, xi_w, log_T].
        J_sw = self._sound_waves._grad_theta_omega_gw_h2_analytical(
            frequency,
            jnp.stack([log_K, log_R_H_star, xi_w, log_T_star]),
        )
        # Pad with zero column for epsilon (last axis).
        J_sw_full = jnp.concatenate(
            [J_sw, jnp.zeros(J_sw.shape[:-1] + (1,), dtype=J_sw.dtype)],
            axis=-1,
        )  # (..., 5)

        # Turbulence Jacobian (..., 3) for [log_Omega_s, log_R, log_T].
        log_Omega_s = log_K + jnp.log10(epsilon)
        J_mhd = self._turbulence._grad_theta_omega_gw_h2_analytical(
            frequency,
            jnp.stack([log_Omega_s, log_R_H_star, log_T_star]),
        )

        # d([log_Omega_s, log_R_H_star, log_T_star])
        #     / d([log_K, log_R_H_star, xi_w, log_T_star, epsilon])
        ln10 = jnp.log(10.0)
        d_inner_d_outer = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0 / (ln10 * epsilon)],  # log_Omega_s
                [0.0, 1.0, 0.0, 0.0, 0.0],  # log_R_H_star
                [0.0, 0.0, 0.0, 1.0, 0.0],  # log_T_star
            ]
        )  # (3, 5)
        J_mhd_full = J_mhd @ d_inner_d_outer  # (..., 5)

        return J_sw_full + J_mhd_full
