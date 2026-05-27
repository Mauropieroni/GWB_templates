"""
Lognormal-bump envelope modulated by a sharp-feature oscillation.

Two parametrizations:
  - ``lognormal_bump_sharp``:     A_sharp, omega_sharp_Hz sampled directly.
  - ``lognormal_bump_sharp_log``: log10 A_sharp, log10 omega_sharp_Hz for wide
                                  prior ranges.

The template is the product of a lognormal bump and a sharp-feature modulation:

    Omega(f) = lognormal_bump(f; A, f*, sigma)
             * (1 + A_sharp * cos(omega * f + theta))

All operations are pure JAX so automatic differentiation is fully supported.

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.lognormal_bump import (
    lognormal_bump,
    d1lognormal_bump,
)
from gwb_templates.inflation_templates.sharp_feature import (
    sharp_feature,
    d1sharp_feature,
    sharp_feature_log,
    d1sharp_feature_log,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# ── Linear parametrization ────────────────────────────────────────────────────


def lognormal_bump_sharp(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Lognormal bump × sharp-feature modulation (linear amplitude / frequency).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, log10 width,
               A_sharp, omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return lognormal_bump(freq, pars[:3]) * sharp_feature(freq, pars[3:6])


def d1lognormal_bump_sharp(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``lognormal_bump_sharp`` w.r.t. pars.

    Product rule: d(E*F)/dp_env = dE/dp * F,  d(E*F)/dp_feat = E * dF/dp.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, log10 width,
               A_sharp, omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq, 6).
    """
    pars = jnp.asarray(pars)
    E = lognormal_bump(freq, pars[:3])
    F = sharp_feature(freq, pars[3:6])
    dE = d1lognormal_bump(freq, pars[:3])  # (N, 3)
    dF = d1sharp_feature(freq, pars[3:6])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


lognormal_bump_sharp_model = ut.Signal_model(
    "lognormal_bump_sharp",
    lognormal_bump_sharp,
    dtemplate=d1lognormal_bump_sharp,
    model_label="Lognormal Bump + Sharp Feature",
    parameter_names=[
        "log_amplitude",
        "log_pivot",
        "log_width",
        "A_sharp",
        "omega_sharp_Hz",
        "phase_sharp",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\log_{10}\sigma$",
        r"$A_{\rm s}$",
        r"$\omega_{\rm s}\,[\mathrm{Hz}^{-1}]$",
        r"$\phi_{\rm s}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "log_width": {"min": -2.0, "max": 1.0},
        "A_sharp": {"min": -1.0, "max": 1.0},
        "omega_sharp_Hz": {"min": 0.0, "max": 1e5},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    },
)


# ── Log parametrization ───────────────────────────────────────────────────────


def lognormal_bump_sharp_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Lognormal bump × sharp-feature modulation (log-scaled amplitude / frequency).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, log10 width,
               log10 A_sharp, log10 omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return lognormal_bump(freq, pars[:3]) * sharp_feature_log(freq, pars[3:6])


def d1lognormal_bump_sharp_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``lognormal_bump_sharp_log`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 pivot, log10 width,
               log10 A_sharp, log10 omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq, 6).
    """
    pars = jnp.asarray(pars)
    E = lognormal_bump(freq, pars[:3])
    F = sharp_feature_log(freq, pars[3:6])
    dE = d1lognormal_bump(freq, pars[:3])  # (N, 3)
    dF = d1sharp_feature_log(freq, pars[3:6])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


lognormal_bump_sharp_log_model = ut.Signal_model(
    "lognormal_bump_sharp_log",
    lognormal_bump_sharp_log,
    dtemplate=d1lognormal_bump_sharp_log,
    model_label="Lognormal Bump + Sharp Feature (log params)",
    parameter_names=[
        "log_amplitude",
        "log_pivot",
        "log_width",
        "log_A_sharp",
        "log_omega_sharp_Hz",
        "phase_sharp",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\log_{10}\sigma$",
        r"$\log_{10}A_{\rm s}$",
        r"$\log_{10}(\omega_{\rm s}/\mathrm{Hz}^{-1})$",
        r"$\phi_{\rm s}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "log_width": {"min": -2.0, "max": 1.0},
        "log_A_sharp": {"min": -3.0, "max": 0.0},
        "log_omega_sharp_Hz": {"min": 0.0, "max": 5.0},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    },
)
