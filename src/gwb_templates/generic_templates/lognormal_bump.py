"""
Log-normal bump spectrum template.

Three-parameter Gaussian in log10-frequency space:

    Omega(f) = 10^A * exp(-0.5 * (log10(f / f_*) / sigma)^2)

Used as a standalone phenomenological model and as an envelope for combined
inflation templates (``lognormal_bump_sharp``).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


def lognormal_bump(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Log-normal bump: Omega(f) = 10^A * exp(-0.5 * (log10(f/f*) / sigma)^2).

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot frequency, log10 width].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_pivot, log_width = pars[0], pars[1], pars[2]
    pivot = 10.0**log_pivot
    width = 10.0**log_width
    return 10.0**log_amplitude * jnp.exp(-0.5 * (jnp.log10(freq / pivot) / width) ** 2)


def d1lognormal_bump(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``lognormal_bump`` w.r.t. pars.

    Let u = log10(f/f*), sigma = 10^log_width.
    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot frequency, log10 width].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(log_A)      = model * ln(10)
            d/d(log_pivot)  = model * u / sigma^2
            d/d(log_width)  = model * u^2 * ln(10) / sigma^2
    """
    log_pivot, log_width = pars[1], pars[2]
    pivot = 10.0**log_pivot
    width = 10.0**log_width  # sigma
    model = lognormal_bump(freq, pars)
    u = jnp.log10(freq / pivot)  # log10(f / f*)

    d_log_A = model * jnp.log(10.0)
    d_log_piv = model * u / width**2
    d_log_wid = model * u**2 * jnp.log(10.0) / width**2

    return jnp.stack([d_log_A, d_log_piv, d_log_wid], axis=1)


lognormal_bump_model = ut.Signal_model(
    "lognormal_bump",
    lognormal_bump,
    dtemplate=d1lognormal_bump,
    model_label="Lognormal Bump",
    parameter_names=["log_amplitude", "log_pivot", "log_width"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$\log_{10}\sigma$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "log_width": {"min": -2.0, "max": 1.0},
    },
)
