"""
Broken power-law template with fixed transition smoothness (delta = 1).

Four-parameter version of ``broken_power_law`` obtained by fixing
``log_transition = 0`` (delta = 1).  Parameters: log amplitude, log break
frequency, low-frequency tilt n1, high-frequency tilt n2.  Used directly
as the FOPT broken power-law template; see ``fopt_broken_power_law``.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.broken_power_law import (
    broken_power_law,
    d1broken_power_law,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


def broken_power_law_fixed_smoothness(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Broken power law with a fixed transition shape (4 parameters).

    Special case of :func:`broken_power_law` with ``log_transition = 0``
    (delta = 1), which yields:

        Omega(f) = A * x^n1 * ((1 + x) / 2)^(n2 - n1)  where x = f / f_*.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, tilt_1, tilt_2].
    Returns:
        jax.Array of shape (N_freq,).
    """
    # Append log_transition = 0 (delta = 1) to recover broken_power_law_fixed_smoothness
    # from the 5-parameter broken_power_law.
    pars_5 = jnp.concatenate([jnp.asarray(pars), jnp.zeros(1)])
    return broken_power_law(freq, pars_5)


def d1broken_power_law_fixed_smoothness(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``broken_power_law_fixed_smoothness`` w.r.t. pars.

    Delegates to ``d1broken_power_law`` with ``log_transition=0`` appended
    and discards the 5th column (d/d(log_transition)).

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, tilt_1, tilt_2].
    Returns:
        jax.Array of shape (N_freq, 4).
    """
    pars_5 = jnp.concatenate([jnp.asarray(pars), jnp.zeros(1)])
    J5 = d1broken_power_law(freq, pars_5)
    return J5[:, :4]


broken_power_law_fixed_smoothness_model = ut.Signal_model(
    "broken_power_law_fixed_smoothness",
    broken_power_law_fixed_smoothness,
    dtemplate=d1broken_power_law_fixed_smoothness,
    model_label="Broken Power Law (fixed smoothness)",
    parameter_names=["log_amplitude", "log_pivot", "tilt_1", "tilt_2"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$n_1$",
        r"$n_2$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "tilt_1": {"min": -10.0, "max": 10.0},
        "tilt_2": {"min": -10.0, "max": 10.0},
    },
)
