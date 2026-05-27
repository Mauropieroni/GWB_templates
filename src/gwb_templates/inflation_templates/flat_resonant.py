"""
Flat (amplitude-only) envelope modulated by a resonant-feature oscillation.

Two parametrizations:
  - ``flat_resonant``:     A_log, omega_log sampled directly.
  - ``flat_resonant_log``: log10 A_log, log10 omega_log for wide prior ranges.

The template is the product of a flat spectrum and a resonant modulation:

    Omega(f) = 10^A * resonant_feature(f; A_log, omega_log, phi_log)

All operations are pure JAX so automatic differentiation is fully supported.

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.amplitude import amplitude, d1amplitude
from gwb_templates.inflation_templates.resonant_feature import (
    resonant_feature,
    d1resonant_feature,
    resonant_feature_log,
    d1resonant_feature_log,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# ── Linear parametrization ────────────────────────────────────────────────────


def flat_resonant(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Flat spectrum × resonant-feature modulation (linear A_log / omega_log).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, A_resonant, omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return amplitude(freq, pars[:1]) * resonant_feature(freq, pars[1:4])


def d1flat_resonant(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``flat_resonant`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, A_resonant, omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq, 4).
    """
    pars = jnp.asarray(pars)
    E = amplitude(freq, pars[:1])
    F = resonant_feature(freq, pars[1:4])
    dE = d1amplitude(freq, pars[:1])  # (N, 1)
    dF = d1resonant_feature(freq, pars[1:4])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


flat_resonant_model = ut.Signal_model(
    "flat_resonant",
    flat_resonant,
    dtemplate=d1flat_resonant,
    model_label="Flat + Resonant Feature",
    parameter_names=["log_amplitude", "A_resonant", "omega_resonant", "phase_resonant"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$A_{\rm r}$",
        r"$\omega_{\rm r}$",
        r"$\phi_{\rm r}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "A_resonant": {"min": 0.0, "max": 1.0},
        "omega_resonant": {"min": 1e-3, "max": 100.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    },
)


# ── Log parametrization ───────────────────────────────────────────────────────


def flat_resonant_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Flat spectrum × resonant-feature modulation (log-scaled A_log / omega_log).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 A_resonant, log10 omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq,).
    """
    return amplitude(freq, pars[:1]) * resonant_feature_log(freq, pars[1:4])


def d1flat_resonant_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``flat_resonant_log`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 A_resonant, log10 omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq, 4).
    """
    pars = jnp.asarray(pars)
    E = amplitude(freq, pars[:1])
    F = resonant_feature_log(freq, pars[1:4])
    dE = d1amplitude(freq, pars[:1])  # (N, 1)
    dF = d1resonant_feature_log(freq, pars[1:4])  # (N, 3)
    return jnp.concatenate([dE * F[:, None], E[:, None] * dF], axis=1)


flat_resonant_log_model = ut.Signal_model(
    "flat_resonant_log",
    flat_resonant_log,
    dtemplate=d1flat_resonant_log,
    model_label="Flat + Resonant Feature (log params)",
    parameter_names=[
        "log_amplitude",
        "log_A_resonant",
        "log_omega_resonant",
        "phase_resonant",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}A_{\rm r}$",
        r"$\log_{10}\omega_{\rm r}$",
        r"$\phi_{\rm r}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_A_resonant": {"min": -3.0, "max": 0.0},
        "log_omega_resonant": {"min": -3.0, "max": 2.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    },
)
