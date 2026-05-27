"""
Galactic compact binary foreground template.

Implements Eq. 6 of Karnesis et al. 2021 (arXiv:2103.14598).
"""

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# ── Observation-dependent fiducial parameter function ────────────────────────


def galactic_pars(
    Tobs_yrs: float = 4.0, snr: float = 7.0, links: int = 6
) -> tuple[float, float, float, float, float]:
    """
    Compute the fiducial shape parameters for the galactic WD binary foreground.

    Fit coefficients from Table 1 of Karnesis et al. 2021 (arXiv:2103.14598).

    Args:
        Tobs_yrs: Observation time in years (must be in [0.25, 10]).
        snr: Signal-to-noise ratio threshold (5 or 7).
        links: Number of LISA links (6 only).
    Returns:
        Tuple (Ampl, alpha, fr1, frk, fr2) — shape parameters in physical units
        (fr* in Hz).
    """
    if snr not in (5.0, 7.0):
        raise ValueError(f"snr must be 5 or 7, got {snr}")
    if links != 6:
        raise ValueError("Only links=6 is supported")
    if not (0.25 <= Tobs_yrs <= 10.0):
        raise ValueError(
            f"Galaxy fit is valid for Tobs in [0.25, 10] years, got {Tobs_yrs}"
        )

    # Columns: Ampl, alpha, fr2, af1, bf1, afk, bfk
    _L6_snr5 = [1.14e-44, 1.66, 0.00059, -0.15, -2.78, -0.34, -2.55]
    _L6_snr7 = [1.15e-44, 1.56, 0.00067, -0.15, -2.72, -0.37, -2.49]

    Ampl, alpha, fr2, af1, bf1, afk, bfk = _L6_snr5 if snr == 5.0 else _L6_snr7

    log10_T = math.log10(Tobs_yrs)
    fr1 = 10.0 ** (af1 * log10_T + bf1)
    frk = 10.0 ** (afk * log10_T + bfk)

    return Ampl, alpha, fr1, frk, fr2


# ── Pre-computed fiducial constants (4 yr, SNR 7, 6 links) ───────────────────

_Ampl_fid, _alpha_fid, _fr1_fid, _frk_fid, _fr2_fid = galactic_pars(4.0, 7.0, 6)
_LOG_GALACTIC_FID = math.log10(_Ampl_fid)
_LOG_FR1_FID = math.log10(_fr1_fid)
_LOG_FRK_FID = math.log10(_frk_fid)
_LOG_FR2_FID = math.log10(_fr2_fid)


def galactic_binaries(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Galactic binary foreground (Eq. 6 of arXiv:2103.14598).

    Omega(f) = 10^log_A * f^(2/3) * exp(-(f/f_r1)^alpha)
               * (1/2) * (1 + tanh(-(f - f_rk) / f_r2))

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, alpha, log10 f_r1, log10 f_rk, log10 f_r2].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_galactic, alpha, log_fr1, log_frk, log_fr2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fr1 = 10.0**log_fr1
    frk = 10.0**log_frk
    fr2 = 10.0**log_fr2

    return (
        10.0**log_galactic
        * freq ** (2.0 / 3.0)
        * jnp.exp(-((freq / fr1) ** alpha))
        * 0.5
        * (1.0 + jnp.tanh(-(freq - frk) / fr2))
    )


def d1galactic_binaries(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``galactic_binaries`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, alpha, log10 f_r1, log10 f_rk, log10 f_r2].
    Returns:
        jax.Array of shape (N_freq, 5).
    """
    _, alpha, log_fr1, log_frk, log_fr2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fvec = jnp.asarray(freq)
    fr1 = 10.0**log_fr1
    frk = 10.0**log_frk
    fr2 = 10.0**log_fr2
    u1 = (fvec / fr1) ** alpha
    t = jnp.tanh(-(fvec - frk) / fr2)
    model = galactic_binaries(fvec, pars)
    ln10 = jnp.log(10.0)

    d_logA = model * ln10
    d_alpha = model * (-u1 * jnp.log(fvec / fr1))
    d_logfr1 = model * (alpha * ln10 * u1)
    # d/d(log_frk): arg = -(f-frk)/fr2,  d(arg)/d(log_frk) = frk*ln10/fr2
    # d(model)/d(log_frk) = model * (1-t) * frk*ln10/fr2
    d_logfrk = model * (1.0 - t) * frk * ln10 / fr2
    # d/d(log_fr2): d(arg)/d(log_fr2) = (f-frk)*ln10/fr2
    d_logfr2 = model * (1.0 - t) * (fvec - frk) * ln10 / fr2

    return jnp.stack([d_logA, d_alpha, d_logfr1, d_logfrk, d_logfr2], axis=1)


galactic_binaries_model = ut.Signal_model(
    "galactic_binaries",
    galactic_binaries,
    dtemplate=d1galactic_binaries,
    model_label="Galactic Binaries",
    parameter_names=["log_galactic", "alpha", "log_fr1", "log_frk", "log_fr2"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
        r"$\alpha$",
        r"$\log_{10}(f_{r1}/\mathrm{Hz})$",
        r"$\log_{10}(f_{rk}/\mathrm{Hz})$",
        r"$\log_{10}(f_{r2}/\mathrm{Hz})$",
    ],
    prior={
        "log_galactic": {"min": -14.0, "max": -5.0},
        "alpha": {"min": 0.1, "max": 10.0},
        "log_fr1": {"min": -4.5, "max": -2.0},
        "log_frk": {"min": -4.5, "max": -2.0},
        "log_fr2": {"min": -5.0, "max": -2.5},
    },
)


# ── Amplitude-only variant (shape fixed to 4 yr / SNR 7 / 6 links) ───────────

_FIDUCIAL_SHAPE_PARS = jnp.array([_alpha_fid, _LOG_FR1_FID, _LOG_FRK_FID, _LOG_FR2_FID])


def galactic_binaries_A(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Galactic binary foreground with shape fixed to the 4 yr / SNR 7 fiducial.

    Free parameter: log10 amplitude only.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq,).
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_SHAPE_PARS])
    return galactic_binaries(freq, full_pars)


def d1galactic_binaries_A(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``galactic_binaries_A`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq, 1): d/d(log_galactic) only.
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_SHAPE_PARS])
    return d1galactic_binaries(freq, full_pars)[:, :1]


galactic_binaries_A_model = ut.Signal_model(
    "galactic_binaries_A",
    galactic_binaries_A,
    dtemplate=d1galactic_binaries_A,
    model_label="Galactic Binaries (amplitude only)",
    parameter_names=["log_galactic"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_{\rm Gal})$"],
    prior={"log_galactic": {"min": -14.0, "max": -5.0}},
)
