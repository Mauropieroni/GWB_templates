"""
Double-peak envelope modulated by a sharp-feature oscillation.

Two parametrizations:
  - ``double_peak_sharp``:     A_lin, omega_lin_Hz sampled directly.
  - ``double_peak_sharp_log``: log10 A_lin, log10 omega_lin_Hz for wide
                                prior ranges.

The template is the product of the double-peak spectrum and a sharp-feature
modulation:

    Omega(f) = double_peak(f; ...) * (1 + A_lin * cos(omega * f + theta))

All operations are pure JAX so automatic differentiation is fully supported.

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.inflation_templates.double_peak import double_peak, d1double_peak
from gwb_templates.inflation_templates.sharp_feature import (
    sharp_feature,
    d1sharp_feature,
    sharp_feature_log,
    d1sharp_feature_log,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

_DOUBLE_PEAK_NPAR = 7  # log_amplitude, log_pivot, beta, k1, k2, rho, gamma


# ── Linear parametrization ────────────────────────────────────────────────────


def double_peak_sharp(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Double-peak spectrum × sharp-feature modulation (linear A / omega).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma,
               A_lin, omega_lin_Hz, theta_lin].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return double_peak(freq, pars[:_DOUBLE_PEAK_NPAR]) * sharp_feature(
        freq, pars[_DOUBLE_PEAK_NPAR:]
    )


def d1double_peak_sharp(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``double_peak_sharp`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma,
               A_lin, omega_lin_Hz, theta_lin].
    Returns:
        jax.Array of shape (N_freq, 10).
    """
    pars = jnp.asarray(pars)
    E = double_peak(freq, pars[:_DOUBLE_PEAK_NPAR])
    F = sharp_feature(freq, pars[_DOUBLE_PEAK_NPAR:])
    dE = d1double_peak(freq, pars[:_DOUBLE_PEAK_NPAR])  # (N, 7)
    dF = d1sharp_feature(freq, pars[_DOUBLE_PEAK_NPAR:])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


double_peak_sharp_model = ut.Signal_model(
    "double_peak_sharp",
    double_peak_sharp,
    dtemplate=d1double_peak_sharp,
    model_label="Double Peak + Sharp Feature",
    parameter_names=[
        "log_amplitude",
        "log_pivot",
        "beta",
        "k1",
        "k2",
        "rho",
        "gamma",
        "A_lin",
        "omega_lin_Hz",
        "theta_lin",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\beta$",
        r"$\kappa_1$",
        r"$\kappa_2$",
        r"$\rho$",
        r"$\gamma$",
        r"$A_{\rm lin}$",
        r"$\omega_{\rm lin}\,[\mathrm{Hz}^{-1}]$",
        r"$\theta_{\rm lin}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "beta": {"min": 0.0, "max": 10.0},
        "k1": {"min": 0.1, "max": 10.0},
        "k2": {"min": 0.1, "max": 10.0},
        "rho": {"min": 0.01, "max": 5.0},
        "gamma": {"min": -5.0, "max": 5.0},
        "A_lin": {"min": -1.0, "max": 1.0},
        "omega_lin_Hz": {"min": 0.0, "max": 1e5},
        "theta_lin": {"min": -3.14159, "max": 3.14159},
    },
)


# ── Log parametrization ───────────────────────────────────────────────────────


def double_peak_sharp_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Double-peak spectrum × sharp-feature modulation (log-scaled A / omega).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma,
               log10 A_lin, log10 omega_lin_Hz, theta_lin].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return double_peak(freq, pars[:_DOUBLE_PEAK_NPAR]) * sharp_feature_log(
        freq, pars[_DOUBLE_PEAK_NPAR:]
    )


def d1double_peak_sharp_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``double_peak_sharp_log`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma,
               log10 A_lin, log10 omega_lin_Hz, theta_lin].
    Returns:
        jax.Array of shape (N_freq, 10).
    """
    pars = jnp.asarray(pars)
    E = double_peak(freq, pars[:_DOUBLE_PEAK_NPAR])
    F = sharp_feature_log(freq, pars[_DOUBLE_PEAK_NPAR:])
    dE = d1double_peak(freq, pars[:_DOUBLE_PEAK_NPAR])  # (N, 7)
    dF = d1sharp_feature_log(freq, pars[_DOUBLE_PEAK_NPAR:])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


double_peak_sharp_log_model = ut.Signal_model(
    "double_peak_sharp_log",
    double_peak_sharp_log,
    dtemplate=d1double_peak_sharp_log,
    model_label="Double Peak + Sharp Feature (log params)",
    parameter_names=[
        "log_amplitude",
        "log_pivot",
        "beta",
        "k1",
        "k2",
        "rho",
        "gamma",
        "log_A_lin",
        "log_omega_lin_Hz",
        "theta_lin",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\beta$",
        r"$\kappa_1$",
        r"$\kappa_2$",
        r"$\rho$",
        r"$\gamma$",
        r"$\log_{10}A_{\rm lin}$",
        r"$\log_{10}(\omega_{\rm lin}/\mathrm{Hz}^{-1})$",
        r"$\theta_{\rm lin}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "beta": {"min": 0.0, "max": 10.0},
        "k1": {"min": 0.1, "max": 10.0},
        "k2": {"min": 0.1, "max": 10.0},
        "rho": {"min": 0.01, "max": 5.0},
        "gamma": {"min": -5.0, "max": 5.0},
        "log_A_lin": {"min": -3.0, "max": 0.0},
        "log_omega_lin_Hz": {"min": 0.0, "max": 5.0},
        "theta_lin": {"min": -3.14159, "max": 3.14159},
    },
)
