"""
Amplitude-only (flat) spectrum template.

Single-parameter model for a frequency-independent GWB:

    Omega(f) = 10^A

Used as a simple baseline model or as a building block for composite
spectra (e.g. ``flat_resonant``).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


def amplitude(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Flat (amplitude-only) spectrum: Omega(f) = 10^A.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude = pars[0]
    return 10.0**log_amplitude * jnp.ones_like(freq)


def d1amplitude(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``amplitude`` w.r.t. pars.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq, 1):
            d/d(log_A)  = model * ln(10)
    """
    model = amplitude(freq, pars)
    return (model * jnp.log(10.0))[:, None]


amplitude_model = ut.Signal_model(
    "amplitude",
    amplitude,
    dtemplate=d1amplitude,
    model_label="Amplitude",
    parameter_names=["log_amplitude"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_*)$"],
    prior={"log_amplitude": {"min": -20.0, "max": -5.0}},
)
