"""
Cosmic String Model I with non-standard equation of state (EOS) template (5 parameters).

Extends :class:`CosmicStringModelI` with a modified high-frequency spectrum above a
cutoff frequency set by a cosmological phase transition. The EOS parameter ``w``
controls the spectral tilt above the cutoff via Appendix B of arXiv:2405.03740.

Reference: arXiv:2405.03740, Appendix B.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from gwb_templates import constants as ct
from gwb_templates.template import AnalyticTemplate
from gwb_templates.utils import log_log_interpolate
from gwb_templates.cosmic_string_templates.cosmic_string_model_i import (
    _Gamma,
    _Omega_m,
    _Omega_r_dof,
    _Omega_rm,
    _Omega_small_alpha,
    _get_f_min_m,
    _get_f_min_r,
    _hyperharmonic,
    _safe_q,
    _xi_r,
)

ArrayLike: TypeAlias = float | int | np.ndarray | jax.Array

# Effective relativistic DOF at the QCD transition (~0.15 GeV)
_GSTAR_QCD = 10.75
# Effective DOF for energy density at low temperature today
_GSTAR_LOW_T = 3.36
# Effective DOF for entropy density at low temperature today
_G_ENTROPY_LOW_T = 3.91


# ── EOS helper functions ──────────────────────────────────────────────────────


def _get_eos_slope(eos: ArrayLike) -> jax.Array:
    """High-frequency spectral tilt induced by a non-standard EOS w (Eq. B.1)."""
    return jnp.where(
        eos <= 1.0 / 9.0,
        1.0,
        -2.0 * (3.0 * eos - 1.0) / (3.0 * eos + 1.0),
    )


def _get_eos_f_cut(
    temperature_GeV: ArrayLike, alpha: ArrayLike, Gmu: ArrayLike
) -> jax.Array:
    """Cutoff frequency above which the EOS-modified spectrum applies (Eq. 2.10)."""
    third_factor = (1e-12 * _xi_r / alpha / Gmu) ** 0.5
    gstar = jnp.where(temperature_GeV > 0.2, ct.g_star_high_T, _GSTAR_QCD)
    last_factors = (
        gstar ** (1.0 / 6.0)
        * _GSTAR_LOW_T ** (-4.0 / 3.0)
        * _G_ENTROPY_LOW_T ** (7.0 / 6.0)
    )
    return 8.67e-3 * temperature_GeV * third_factor * last_factors * ct.h_bar_eV_s


def _compute_radiation_eos(
    f_eV: jax.Array,
    Gmu: ArrayLike,
    alpha: ArrayLike,
    q: ArrayLike,
    f_cut: ArrayLike,
    slope: ArrayLike,
) -> jax.Array:
    """Radiation contribution with EOS-modified high-frequency tail (Eq. B.1)."""
    f_min_r = _get_f_min_r(Gmu, alpha)
    N_standard = jnp.maximum(1.0, jnp.floor(f_eV / f_min_r))
    Omega_rad = _Omega_r_dof(f_eV, Gmu, alpha, q, N_standard)

    # Guard against f_cut=0 (division by zero) by clamping.
    safe_f_cut = jnp.where(f_cut > 0.0, f_cut, 1.0)
    N_high = jnp.maximum(1.0, jnp.floor(f_eV / safe_f_cut))

    # Spectrum value at f_cut with N=1.
    Omega_cut = _Omega_r_dof(jnp.reshape(safe_f_cut, (1,)), Gmu, alpha, q, jnp.ones(1))[
        0
    ]
    Omega_ref = _Omega_r_dof(f_eV, Gmu, alpha, q, N_high)
    Omega_high = (
        Omega_cut * (safe_f_cut / f_eV) ** slope * _hyperharmonic(q - slope, N_high)
        + Omega_rad
        - Omega_ref
    )
    apply = (f_eV >= f_cut) & (slope != 0.0) & (f_cut > 0.0)
    return jnp.where(apply, Omega_high, Omega_rad)


def _compute_spectrum_eos(
    freq: jax.Array,
    log_Gmu: ArrayLike,
    log_alpha: ArrayLike,
    q: ArrayLike,
    logtemp_GeV: ArrayLike,
    eos: ArrayLike,
) -> jax.Array:
    """Evaluate h^2 * Omega_GW(freq) for the EOS model (JAX-traceable)."""
    q = _safe_q(q)
    f_eV = jnp.asarray(freq, dtype=float) * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha
    temperature_GeV = 10.0**logtemp_GeV

    slope = _get_eos_slope(eos)
    f_cut = _get_eos_f_cut(temperature_GeV, alpha, Gmu)

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    N_m = jnp.maximum(1.0, jnp.floor(f_eV / f_min_m))

    def large_alpha_branch(_):
        omega_r = jnp.where(
            f_eV >= f_min_r,
            _compute_radiation_eos(f_eV, Gmu, alpha, q, f_cut, slope),
            0.0,
        )
        omega_rm = jnp.where(f_eV >= f_min_m, _Omega_rm(f_eV, Gmu, alpha, q, N_m), 0.0)
        omega_m = jnp.where(f_eV >= f_min_m, _Omega_m(f_eV, Gmu, alpha, q, N_m), 0.0)
        omega_m_contrib = jnp.where(alpha < 1e-3, omega_m, jnp.zeros_like(f_eV))
        return omega_r + omega_rm + omega_m_contrib

    def small_alpha_branch(_):
        return jnp.where(
            f_eV > f_min_m,
            _Omega_small_alpha(f_eV, Gmu, alpha, q, N_m),
            0.0,
        )

    Omega = jax.lax.cond(
        alpha > _Gamma * Gmu * _xi_r,  # type: ignore[operator]
        large_alpha_branch,
        small_alpha_branch,
        None,
    )
    return jnp.where(Omega < 0, 0.0, Omega * ct.h**2)


# NB: For the moment we jit compile in this way, but it should be handled at a higher
# level (e.g. via a class method) if we want to support both JIT and non-JIT versions
# of the template.
_compute_spectrum_eos_jit = jax.jit(_compute_spectrum_eos)


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelIEos(AnalyticTemplate):
    r"""
    Cosmic String Model I with non-standard equation of state
    (arXiv:2405.03740, App. B).

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`.
    log_alpha
        :math:`\log_{10}` of the loop-size parameter :math:`\alpha`.
    q
        Harmonic power-law index.
    logtemp_GeV
        :math:`\log_{10}` of the phase-transition temperature in GeV.
    eos
        Equation-of-state parameter :math:`w`.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Blanco-Pillado:2024aca,
    author = "Blanco-Pillado, Jose J. and Cui, Yanou and Kuroyanagi, Sachiko and
        Lewicki, Marek and Nardini, Germano and Pieroni, Mauro and Rybak, Ivan Yu. and
        Sousa, Lara and Wachter, Jeremy M.",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from cosmic strings in LISA: reconstruction pipeline
        and physics interpretation}",
    eprint = "2405.03740",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-02, CERN-TH-2024-085",
    doi = "10.1088/1475-7516/2025/05/006",
    journal = "JCAP",
    volume = "05",
    pages = "006",
    year = "2025"
}
""",
    )

    def __init__(
        self,
        n_interp_points: int = 100,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.n_interp_points: int = int(n_interp_points)

        default_labels = {
            "log_Gmu": r"$\log_{10}(G\mu)$",
            "log_alpha": r"$\log_{10}\alpha$",
            "q": r"$q$",
            "logtemp_GeV": r"$\log_{10}(T/\mathrm{GeV})$",
            "eos": r"$w$",
        }
        default_priors = {
            "log_Gmu": {"min": -12.0, "max": -6.0},
            "log_alpha": {"min": -3.0, "max": 0.0},
            "q": {"min": 1.01, "max": 2.0},
            "logtemp_GeV": {"min": -2.0, "max": 5.0},
            "eos": {"min": 0.0, "max": 1.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Cosmic String Model I (EOS)"
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
        log_Gmu: ArrayLike,
        log_alpha: ArrayLike,
        q: ArrayLike,
        logtemp_GeV: ArrayLike,
        eos: ArrayLike,
    ) -> jax.Array:
        return log_log_interpolate(
            jnp.asarray(frequency, dtype=float),
            _compute_spectrum_eos_jit,
            log_Gmu,
            log_alpha,
            q,
            logtemp_GeV,
            eos,
            n_points=self.n_interp_points,
        )
