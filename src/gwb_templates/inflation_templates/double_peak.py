"""
Double-peak inflationary GWB template.

Combines a power-law rise with a Heaviside cutoff (first peak) and a
log-normal with an erfc modulation (second peak).  Useful for modelling
PBH-related spectra.

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


def double_peak(
    freq: ArrayLike,
    pars: ParamLike,
    c1: float = 0.8164965809277261,  # sqrt(2/3)
    tilt_p: float = 2.5,
) -> jax.Array:
    """
    Double-peak inflationary spectrum.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma].
        c1: Fixed shape constant (default sqrt(2/3)).
        tilt_p: Fixed UV spectral index of the first peak (default 2.5).
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_pivot, beta, k1, k2, rho, gamma = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
    )
    amplitude = 10.0**log_amplitude
    pivot = 10.0**log_pivot

    x = freq / pivot
    x1 = x / k1
    c1_k1 = c1 / k1
    log10x2 = jnp.log10(x / k2)

    bump_factor = jnp.abs((c1_k1 - x1) / (c1_k1 - 1.0)) ** (tilt_p * (c1_k1 - 1.0))
    first_term = beta * x1**tilt_p * bump_factor * jnp.heaviside(c1_k1 - x1, 1.0)

    log_normal = jnp.exp(-0.5 * (log10x2 / rho) ** 2)
    erfc_term = jax.scipy.special.erfc(gamma * log10x2)
    second_term = log_normal * erfc_term

    return amplitude * (first_term + second_term)


def d1double_peak(
    freq: ArrayLike,
    pars: ParamLike,
    c1: float = 0.8164965809277261,
    tilt_p: float = 2.5,
) -> jax.Array:
    """
    Analytical Jacobian of ``double_peak`` w.r.t. pars.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, beta, k1, k2, rho, gamma].
    Returns:
        jax.Array of shape (N_freq, 7).
    """
    pars = jnp.asarray(pars)
    log_amplitude, log_pivot, beta, k1, k2, rho, gamma = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
    )
    amp = 10.0**log_amplitude
    pivot = 10.0**log_pivot

    x = freq / pivot
    x1 = x / k1
    c1_k1 = c1 / k1
    log10x2 = jnp.log10(x / k2)
    ln10 = jnp.log(10.0)

    arg1 = jnp.abs((c1_k1 - x1) / (c1_k1 - 1.0))
    exp1 = tilt_p * (c1_k1 - 1.0)
    bump_factor = arg1**exp1
    H = jnp.heaviside(c1_k1 - x1, 1.0)
    first_term = beta * x1**tilt_p * bump_factor * H

    log_normal = jnp.exp(-0.5 * (log10x2 / rho) ** 2)
    erfc_val = jax.scipy.special.erfc(gamma * log10x2)
    erfc_deriv = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-((gamma * log10x2) ** 2))
    second_term = log_normal * erfc_val

    model = amp * (first_term + second_term)

    # d/d(log_amplitude)
    d_logA = ln10 * model

    # d/d(log_pivot)
    sign_c1_k1 = jnp.where(c1_k1 >= 1.0, 1.0, -1.0)
    d_first_logpiv = (
        amp
        * first_term
        * (-tilt_p * ln10 + sign_c1_k1 * exp1 / arg1 * (x1 / (c1_k1 - 1.0)) * ln10)
    )
    d_second_logpiv = (
        amp * second_term * log10x2 / rho**2 + amp * log_normal * gamma * erfc_deriv
    )
    d_logpiv = d_first_logpiv + d_second_logpiv

    # d/d(beta)
    d_beta = amp * first_term / beta

    # d/d(k1)
    log_arg1 = jnp.log(arg1)
    d_k1 = (
        amp
        * first_term
        * (-tilt_p / k1 - log_arg1 * tilt_p * c1 / k1**2 + exp1 / (c1 - k1))
    )

    # d/d(k2)
    d_k2 = (
        amp * log_normal * erfc_deriv * gamma / k2
        + amp * second_term * log10x2 / (rho**2 * k2)
    ) / ln10

    # d/d(rho)
    d_rho = amp * second_term * log10x2**2 / rho**3

    # d/d(gamma)
    d_gamma = amp * log_normal * erfc_deriv * (-log10x2)

    return jnp.stack([d_logA, d_logpiv, d_beta, d_k1, d_k2, d_rho, d_gamma], axis=1)


double_peak_model = ut.Signal_model(
    "double_peak",
    double_peak,
    dtemplate=d1double_peak,
    model_label="Double Peak",
    parameter_names=["log_amplitude", "log_pivot", "beta", "k1", "k2", "rho", "gamma"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\beta$",
        r"$\kappa_1$",
        r"$\kappa_2$",
        r"$\rho$",
        r"$\gamma$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "beta": {"min": 0.0, "max": 10.0},
        "k1": {"min": 0.1, "max": 10.0},
        "k2": {"min": 0.1, "max": 10.0},
        "rho": {"min": 0.01, "max": 5.0},
        "gamma": {"min": -5.0, "max": 5.0},
    },
)
