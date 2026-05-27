"""
FOPT broken power law (old / 2-parameter) template.

Implements the broken power law from Eq. 14 of arXiv:1512.06239:

    Omega(f) = A * x^3 * (7 / (4 + 3 x^2))^(7/2)   x = f / f_*

This is a special case of broken_power_law with fixed n1=3, n2=-4, delta=0.5.
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

# Fixed spectral parameters that reproduce the FOPT-motivated shape.
_N1 = 3.0
_N2 = -4.0
_LOG10_TRANSITION = jnp.log10(0.5)  # delta = 0.5

# Constant offsets so that log_amplitude == log10(peak value) and
# log_pivot == log10(peak frequency), matching the original standalone formula.
# Derived from: A_bpl = A_old * 7^3.5 / (2^7.5 * 3^1.5)
#               f_bpl = f_old * 2 / sqrt(3)
_LOG10_AMP_SHIFT = 3.5 * jnp.log10(7) - 7.5 * jnp.log10(2) - 1.5 * jnp.log10(3)
_LOG10_PIV_SHIFT = jnp.log10(2.0) - 0.5 * jnp.log10(3.0)


def broken_power_law_old(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Simple broken power law from Eq. 14 of arXiv:1512.06239 (2 parameters).

    Omega(f) = A * x^3 * (7 / (4 + 3 x^2))^(7/2)  where x = f / f_*.

    This is a special case of :func:`broken_power_law` with fixed n1=3, n2=-4,
    delta=0.5 (log_transition = log10(0.5)).  The amplitude and pivot offsets
    absorb the constant prefactors so that log_amplitude equals the log10 of
    the peak value and log_pivot equals the log10 of the peak frequency.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 f_star].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amp, log_piv = pars[0], pars[1]
    pars_5 = jnp.array(
        [
            log_amp + _LOG10_AMP_SHIFT,
            log_piv + _LOG10_PIV_SHIFT,
            _N1,
            _N2,
            _LOG10_TRANSITION,
        ]
    )
    return broken_power_law(freq, pars_5)


def d1broken_power_law_old(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``broken_power_law_old`` w.r.t. pars.

    Since log_amp_bpl = log_amp + const and log_piv_bpl = log_piv + const,
    d/d(log_amp) = d(BPL)/d(log_amp_bpl) and d/d(log_piv) = d(BPL)/d(log_piv_bpl).

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 f_star].
    Returns:
        jax.Array of shape (N_freq, 2).
    """
    log_amp, log_piv = pars[0], pars[1]
    pars_5 = jnp.array(
        [
            log_amp + _LOG10_AMP_SHIFT,
            log_piv + _LOG10_PIV_SHIFT,
            _N1,
            _N2,
            _LOG10_TRANSITION,
        ]
    )
    return d1broken_power_law(freq, pars_5)[:, :2]


fopt_broken_power_law_old_model = ut.Signal_model(
    "fopt_old",
    broken_power_law_old,
    dtemplate=d1broken_power_law_old,
    model_label="FOPT Broken Power Law (old)",
    parameter_names=["log_amplitude", "log_f_star"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_f_star": {"min": -5.0, "max": 0.0},
    },
)
