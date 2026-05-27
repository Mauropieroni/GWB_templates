"""
Cosmic String Model I with non-standard equation of state (EOS) template (6 parameters).

Extends cosmic_string_model_i with a modified high-frequency spectrum above a
cutoff frequency set by a cosmological phase transition.  The EOS parameter w
controls the spectral tilt above the cutoff via Appendix B of arXiv:2405.03740.

Reference: arXiv:2405.03740, Appendix B.
"""

from collections.abc import Sequence

import jax
import numpy as np

from gwb_templates import constants as ct
from gwb_templates import utils as ut
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

jax.config.update("jax_enable_x64", True)

# Effective relativistic DOF at the QCD transition (~0.15 GeV)
_GSTAR_QCD = 10.75
# Effective DOF for energy density at low temperature today  (≠ g_star_0 for entropy)
_GSTAR_LOW_T = 3.36
# Effective DOF for entropy density at low temperature today
_G_ENTROPY_LOW_T = 3.91


# ── EOS helper functions ──────────────────────────────────────────────────────


def _get_eos_slope(eos: float) -> float:
    """
    High-frequency spectral tilt induced by a non-standard EOS w.

    Returns +1 for w ≤ 1/9 (radiation-dominated or softer),
    and the EOS-dependent value from Eq. B.1 of arXiv:2405.03740 otherwise.

    Args:
        eos: Equation-of-state parameter w.
    Returns:
        Spectral slope above the EOS cutoff frequency.
    """
    if eos <= 1.0 / 9.0:
        return 1.0
    return -2.0 * (3.0 * eos - 1.0) / (3.0 * eos + 1.0)


def _get_eos_f_cut(temperature_GeV: float, alpha: float, Gmu: float) -> float:
    """
    Cutoff frequency (in Planck natural units) above which the EOS-modified
    spectrum applies.  This implements Eq. 2.10 of arXiv:2405.03740.

    Args:
        temperature_GeV: Phase-transition temperature [GeV].
        alpha: Loop-size parameter.
        Gmu: String tension G*mu.
    Returns:
        f_cut in Planck natural units.
    """
    # Third factor of Eq. 2.10 of arXiv:2405.03740
    third_factor = (1e-12 * _xi_r / alpha / Gmu) ** 0.5
    # Effective degrees of freedom at the transition temperature
    gstar = ct.g_star_high_T if temperature_GeV > 0.2 else _GSTAR_QCD
    # Last two factors of Eq. 2.10 of arXiv:2405.03740
    last_factors = (
        gstar ** (1.0 / 6.0)
        * _GSTAR_LOW_T ** (-4.0 / 3.0)
        * _G_ENTROPY_LOW_T ** (7.0 / 6.0)
    )
    # 8.67e-3 * T_GeV * ... gives f_cut in Hz; multiply by t_planck
    # to get the same Planck-unit frequency used throughout cosmic_string_model_i.
    return 8.67e-3 * temperature_GeV * third_factor * last_factors * ct.h_bar_eV_s


def _compute_radiation_eos(
    f_eV_r: np.ndarray,
    Gmu: float,
    alpha: float,
    q: float,
    f_cut: float,
    slope: float,
) -> np.ndarray:
    """
    Radiation contribution with EOS-modified high-frequency tail.

    Below f_cut uses the standard _Omega_r_dof; above f_cut replaces with
    the EOS-tilted continuation as in the last line of Eq. B.1 of 2405.03740.

    Args:
        f_eV_r: Natural-unit frequency array (radiation mask only).
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        q: Harmonic power-law index.
        f_cut: EOS cutoff frequency in natural units.
        slope: Spectral tilt above the cutoff (from ``_get_eos_slope``).
    Returns:
        np.ndarray of h^2 * Omega_GW radiation contribution at f_eV_r.
    """
    # Below f_cut: standard radiation-era contribution
    f_min_r = _get_f_min_r(Gmu, alpha)
    N_standard = np.floor(f_eV_r / f_min_r)
    Omega_rad = np.array(_Omega_r_dof(f_eV_r, Gmu, alpha, q, N_standard))

    # Identify frequencies above the EOS cutoff
    high_mask = f_eV_r >= f_cut
    if slope == 0.0 or f_cut <= 0.0 or not np.any(high_mask):
        return Omega_rad

    freq_high = f_eV_r[high_mask]

    # Evaluate spectrum at the cutoff (single mode, N=1)
    Omega_cut = _Omega_r_dof(np.array([f_cut]), Gmu, alpha, q, np.array([1.0]))

    # EOS-tilted continuation above f_cut (last line of Eq. B.1 of arXiv:2405.03740)
    N_high = np.floor(freq_high / f_cut)
    Omega_ref = _Omega_r_dof(freq_high, Gmu, alpha, q, N_high)

    Omega_rad[high_mask] = (
        Omega_cut * (f_cut / freq_high) ** slope * _hyperharmonic(q - slope, N_high)
        + Omega_rad[high_mask]
        - Omega_ref
    )
    return Omega_rad


# ── Full spectrum + log-interpolated wrapper ──────────────────────────────────


def _compute_spectrum_eos(
    freq: np.ndarray,
    log_Gmu: float,
    log_alpha: float,
    q: float,
    logtemp_GeV: float,
    eos: float,
) -> np.ndarray:
    """
    Evaluate h^2 * Omega_GW(freq) for the EOS model without log-interpolation.

    Args:
        freq: Frequency array [Hz].
        log_Gmu: log10 of the string tension.
        log_alpha: log10 of the loop-size parameter.
        q: Harmonic power-law index.
        logtemp_GeV: log10 of the phase-transition temperature [GeV].
        eos: Equation-of-state parameter w.
    Returns:
        np.ndarray of h^2 * Omega_GW values at each frequency.
    """
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


# ── Public template function ──────────────────────────────────────────────────


def cosmic_string_model_i_eos(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Cosmic String Model I with non-standard equation of state (arXiv:2405.03740, App B).

    The EOS parameter w modifies the spectral tilt above a cutoff frequency
    determined by a cosmological phase transition at temperature T.

    No JAX automatic differentiation is available (uses NumPy/SciPy internally).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu, log10 alpha, q, log10 T (GeV), w (EOS)].
    Returns:
        jax.Array of h^2 * Omega_GW at each frequency (via log-log interpolation).
    """

    log_Gmu = pars[0]
    log_alpha = pars[1]
    q = pars[2]
    logtemp_GeV = pars[3]
    eos = pars[4]
    return ut.log_log_interpolate(
        freq, _compute_spectrum_eos, log_Gmu, log_alpha, q, logtemp_GeV, eos
    )


cosmic_string_model_i_eos_model = ut.Signal_model(
    "cosmic_string_model_i_eos",
    cosmic_string_model_i_eos,
    model_label="Cosmic String Model I (EOS)",
    parameter_names=["log_Gmu", "log_alpha", "q", "logtemp_GeV", "eos"],
    parameter_labels=[
        r"$\log_{10}(G\mu)$",
        r"$\log_{10}\alpha$",
        r"$q$",
        r"$\log_{10}(T/\mathrm{GeV})$",
        r"$w$",
    ],
    prior={
        "log_Gmu": {"min": -12.0, "max": -6.0},
        "log_alpha": {"min": -3.0, "max": 0.0},
        "q": {"min": 1.01, "max": 2.0},
        "logtemp_GeV": {"min": -2.0, "max": 5.0},
        "eos": {"min": 0.0, "max": 1.0},
    },
)
