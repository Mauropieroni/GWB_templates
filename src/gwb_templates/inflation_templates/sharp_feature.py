"""
Sharp-feature oscillatory modulation template.

Provides a linear-in-amplitude cosine oscillation that can modulate a smooth
envelope.  Two parametrizations are offered:
  - ``sharp_feature``:     direct A, omega, theta sampling
  - ``sharp_feature_log``: log10 amplitude and frequency for wide-range priors

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# ── Linear parametrization ────────────────────────────────────────────────────


def sharp_feature(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Sharp-feature modulation: F(f) = 1 + A_sharp * cos(omega * f + theta).

    Args:
        freq: Frequency grid [Hz].
        pars: [A_sharp, omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq,).
    """
    A_sharp, omega_sharp_Hz, phase_sharp = pars[0], pars[1], pars[2]
    return 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * freq + phase_sharp)


def d1sharp_feature(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``sharp_feature`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [A_sharp, omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(A_sharp)       =  cos(omega*f + theta)
            d/d(omega_sharp)   = -A_sharp * sin(omega*f + theta) * f
            d/d(phase_sharp)   = -A_sharp * sin(omega*f + theta)
    """
    A_sharp, omega_sharp_Hz, phase_sharp = pars[0], pars[1], pars[2]
    arg = omega_sharp_Hz * freq + phase_sharp
    d_A = jnp.cos(arg)
    d_omega = -A_sharp * jnp.sin(arg) * freq
    d_theta = -A_sharp * jnp.sin(arg)
    return jnp.stack([d_A, d_omega, d_theta], axis=1)


sharp_feature_model = ut.Signal_model(
    "sharp_feature",
    sharp_feature,
    dtemplate=d1sharp_feature,
    model_label="Sharp Feature",
    parameter_names=["A_sharp", "omega_sharp_Hz", "phase_sharp"],
    parameter_labels=[
        r"$A_{\rm s}$",
        r"$\omega_{\rm s}\,[\mathrm{Hz}^{-1}]$",
        r"$\phi_{\rm s}$",
    ],
    prior={
        "A_sharp": {"min": -1.0, "max": 1.0},
        "omega_sharp_Hz": {"min": 0.0, "max": 1e5},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    },
)


# ── Log parametrization ───────────────────────────────────────────────────────


def sharp_feature_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Sharp-feature modulation with log-parametrized amplitude and frequency.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 A_sharp, log10 omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_A_sharp, log_omega_sharp_Hz, phase_sharp = pars[0], pars[1], pars[2]
    A_sharp = 10.0**log_A_sharp
    omega_sharp_Hz = 10.0**log_omega_sharp_Hz
    return 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * freq + phase_sharp)


def d1sharp_feature_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``sharp_feature_log`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 A_sharp, log10 omega_sharp_Hz, phase_sharp].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(log_A)     = ln(10)*A * cos(arg)
            d/d(log_omega) = -ln(10)*A*omega * sin(arg) * f
            d/d(theta)     = -A * sin(arg)
    """
    log_A_sharp, log_omega_sharp_Hz, phase_sharp = pars[0], pars[1], pars[2]
    A_sharp = 10.0**log_A_sharp
    omega_sharp_Hz = 10.0**log_omega_sharp_Hz
    arg = omega_sharp_Hz * freq + phase_sharp
    ln10 = jnp.log(10.0)
    d_logA = ln10 * A_sharp * jnp.cos(arg)
    d_logomega = -ln10 * A_sharp * omega_sharp_Hz * jnp.sin(arg) * freq
    d_theta = -A_sharp * jnp.sin(arg)
    return jnp.stack([d_logA, d_logomega, d_theta], axis=1)


sharp_feature_log_model = ut.Signal_model(
    "sharp_feature_log",
    sharp_feature_log,
    dtemplate=d1sharp_feature_log,
    model_label="Sharp Feature (log params)",
    parameter_names=["log_A_sharp", "log_omega_sharp_Hz", "phase_sharp"],
    parameter_labels=[
        r"$\log_{10}A_{\rm s}$",
        r"$\log_{10}(\omega_{\rm s}/\mathrm{Hz}^{-1})$",
        r"$\phi_{\rm s}$",
    ],
    prior={
        "log_A_sharp": {"min": -3.0, "max": 0.0},
        "log_omega_sharp_Hz": {"min": 0.0, "max": 5.0},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    },
)
