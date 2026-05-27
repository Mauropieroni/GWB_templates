"""
Broken power-law with direct smoothness parameter a_1 (5 parameters).

Used as the base spectral template for bubble-collision GW spectra.  The
smoothness a_1 is a direct (non-log) parameter for transparent physical
interpretation:

    Omega(f) = A * x^n_1 * (0.5 + 0.5 * x^a_1)^((n_2 - n_1) / a_1)

where x = f / f_b.

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline
and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def broken_power_law_a1(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Broken power law with a direct smoothness parameter a_1 (5 parameters).

    Omega(f) = A * x^n_1 * (0.5 + 0.5 * x^a_1)^((n_2 - n_1) / a_1)
    where x = f / f_b.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 f_b, n_1, n_2, a_1].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_f_b, n_1, n_2, a_1 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    x = freq / 10.0**log_f_b
    return 10.0**log_amplitude * x**n_1 * (0.5 + 0.5 * x**a_1) ** ((n_2 - n_1) / a_1)


def d1broken_power_law_a1(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``broken_power_law_a1`` w.r.t. pars.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 f_b, n_1, n_2, a_1].
    Returns:
        jax.Array of shape (N_freq, 5):
            d/d(log_A)  = model * ln(10)
            d/d(log_fb) = model * ln(10) * (-n_1 - n_2*x^a_1) / (1 + x^a_1)
            d/d(n_1)    = model * [ln(x) - ln(0.5*(1+x^a_1)) / a_1]
            d/d(n_2)    = model * ln(0.5*(1+x^a_1)) / a_1
            d/d(a_1)    = model * (n_2-n_1)/a_1^2 / (1+x^a_1)
                          * [a_1*x^a_1*ln(x) - (1+x^a_1)*ln(0.5*(1+x^a_1))]
    """
    log_f_b, n_1, n_2, a_1 = pars[1], pars[2], pars[3], pars[4]
    x = freq / 10.0**log_f_b
    xa = x**a_1
    ln10 = jnp.log(10.0)
    log_half_1_xa = jnp.log(0.5 * (1.0 + xa))
    model = broken_power_law_a1(freq, pars)

    d_logA = model * ln10
    d_logfb = model * ln10 * (-n_1 - n_2 * xa) / (1.0 + xa)
    d_n1 = model * (jnp.log(x) - log_half_1_xa / a_1)
    d_n2 = model * log_half_1_xa / a_1
    d_a1 = (
        model
        * (n_2 - n_1)
        / a_1**2
        / (1.0 + xa)
        * (a_1 * xa * jnp.log(x) - (1.0 + xa) * log_half_1_xa)
    )

    return jnp.stack([d_logA, d_logfb, d_n1, d_n2, d_a1], axis=1)


broken_power_law_a1_model = ut.Signal_model(
    "broken_power_law_a1",
    broken_power_law_a1,
    dtemplate=d1broken_power_law_a1,
    model_label="Broken Power Law (a1)",
    parameter_names=[
        "log_amplitude",
        "log_f_b",
        "n_1",
        "n_2",
        "a_1",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_b/\mathrm{Hz})$",
        r"$n_1$",
        r"$n_2$",
        r"$a_1$",
    ],
    prior={
        "log_amplitude": {"prior_type": "uniform", "minimum": -20.0, "maximum": -1.0},
        "log_f_b": {"prior_type": "uniform", "minimum": -10.0, "maximum": 0.0},
        "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
    },
)
