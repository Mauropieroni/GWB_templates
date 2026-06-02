"""
Cosmic String Model I with non-standard equation of state (EOS) template (5 parameters).

Extends :class:`CosmicStringModelI` with a modified high-frequency spectrum
above a cutoff frequency set by a cosmological phase transition. The EOS
parameter ``w`` controls the spectral tilt above the cutoff via
Appendix B of arXiv:2405.03740.

Reference: arXiv:2405.03740, Appendix B.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.typing as jtp
import numpy as np

from gwb_templates import constants as ct
from gwb_templates.template import NumericalTemplate
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
    _xi_r,
)

ArrayLike: TypeAlias = jtp.ArrayLike

# Effective relativistic DOF at the QCD transition (~0.15 GeV)
_GSTAR_QCD = 10.75
# Effective DOF for energy density at low temperature today
_GSTAR_LOW_T = 3.36
# Effective DOF for entropy density at low temperature today
_G_ENTROPY_LOW_T = 3.91


# ── EOS helper functions ──────────────────────────────────────────────────────


def _get_eos_slope(eos: float) -> float:
    """High-frequency spectral tilt induced by a non-standard EOS w (Eq. B.1)."""
    if eos <= 1.0 / 9.0:
        return 1.0
    return -2.0 * (3.0 * eos - 1.0) / (3.0 * eos + 1.0)


def _get_eos_f_cut(temperature_GeV: float, alpha: float, Gmu: float) -> float:
    """Cutoff frequency above which the EOS-modified spectrum applies (Eq. 2.10)."""
    third_factor = (1e-12 * _xi_r / alpha / Gmu) ** 0.5
    gstar = ct.g_star_high_T if temperature_GeV > 0.2 else _GSTAR_QCD
    last_factors = (
        gstar ** (1.0 / 6.0)
        * _GSTAR_LOW_T ** (-4.0 / 3.0)
        * _G_ENTROPY_LOW_T ** (7.0 / 6.0)
    )
    return 8.67e-3 * temperature_GeV * third_factor * last_factors * ct.h_bar_eV_s


def _compute_radiation_eos(
    f_eV_r: np.ndarray,
    Gmu: float,
    alpha: float,
    q: float,
    f_cut: float,
    slope: float,
) -> np.ndarray:
    """Radiation contribution with EOS-modified high-frequency tail (Eq. B.1)."""
    f_min_r = _get_f_min_r(Gmu, alpha)
    N_standard = np.floor(f_eV_r / f_min_r)
    Omega_rad = np.array(_Omega_r_dof(f_eV_r, Gmu, alpha, q, N_standard))

    high_mask = f_eV_r >= f_cut
    if slope == 0.0 or f_cut <= 0.0 or not np.any(high_mask):
        return Omega_rad

    freq_high = f_eV_r[high_mask]
    Omega_cut = _Omega_r_dof(np.array([f_cut]), Gmu, alpha, q, np.array([1.0]))
    N_high = np.floor(freq_high / f_cut)
    Omega_ref = _Omega_r_dof(freq_high, Gmu, alpha, q, N_high)

    Omega_rad[high_mask] = (
        Omega_cut * (f_cut / freq_high) ** slope * _hyperharmonic(q - slope, N_high)
        + Omega_rad[high_mask]
        - Omega_ref
    )
    return Omega_rad


def _compute_spectrum_eos(
    freq: np.ndarray,
    log_Gmu: float,
    log_alpha: float,
    q: float,
    logtemp_GeV: float,
    eos: float,
) -> np.ndarray:
    """Evaluate h^2 * Omega_GW(freq) for the EOS model (no log-interpolation)."""
    freq = np.asarray(freq)
    f_eV = freq * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha
    temperature_GeV = 10.0**logtemp_GeV

    slope = _get_eos_slope(eos)
    f_cut = _get_eos_f_cut(temperature_GeV, alpha, Gmu)

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    Omega = np.zeros(len(freq))

    if alpha > _Gamma * Gmu * _xi_r:
        mask_r = f_eV >= f_min_r
        if mask_r.any():
            fn_r = f_eV[mask_r]
            Omega[mask_r] += _compute_radiation_eos(fn_r, Gmu, alpha, q, f_cut, slope)

        mask_m = f_eV >= f_min_m
        if mask_m.any():
            fn_m = f_eV[mask_m]
            N_m = np.floor(fn_m / f_min_m)
            Omega[mask_m] += _Omega_rm(fn_m, Gmu, alpha, q, N_m)
            if alpha < 1e-3:
                Omega[mask_m] += _Omega_m(fn_m, Gmu, alpha, q, N_m)
    else:
        mask_m = f_eV > f_min_m
        if mask_m.any():
            fn_m = f_eV[mask_m]
            N_m = np.floor(fn_m / f_min_m)
            Omega[mask_m] += _Omega_small_alpha(fn_m, Gmu, alpha, q, N_m)

    return np.where(Omega < 0, 0.0, Omega * ct.h**2)


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelIEos(NumericalTemplate):
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

    jittable: ClassVar[bool] = False
    bibtex_entries: ClassVar[tuple[str, ...]] = ()  # TODO: cite

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
        frequency: jax.Array,
        log_Gmu: jax.Array,
        log_alpha: jax.Array,
        q: jax.Array,
        logtemp_GeV: jax.Array,
        eos: jax.Array,
    ) -> jax.Array:
        freq = np.asarray(frequency, dtype=float)
        return log_log_interpolate(
            freq,
            _compute_spectrum_eos,
            float(log_Gmu),
            float(log_alpha),
            float(q),
            float(logtemp_GeV),
            float(eos),
            n_points=self.n_interp_points,
        )
