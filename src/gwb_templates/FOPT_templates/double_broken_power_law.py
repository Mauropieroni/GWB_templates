"""
Double broken power-law (DBPL) template for first-order phase transitions.

Generic 8-parameter spectral shape used as a phenomenological model for
GW sources from first-order phase transitions (sound waves, turbulence,
bubble collisions).  The amplitude is defined at the second break frequency
f_2 for stable normalisation:

    Omega(f) = N * A * (f/f_1)^n_1
               * (1 + (f/f_1)^a_1)^((n_2 - n_1) / a_1)
               * (1 + (f/f_2)^a_2)^((n_3 - n_2) / a_2)

where N normalises the spectrum so that Omega(f_2) = 10^log_A.

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


def double_broken_power_law(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Double broken power law (8 parameters).

    The amplitude is defined at f_2, and an internal normalization N
    ensures Omega(f_2) = 10^log_amplitude:

    Omega(f) = N * A * (f/f_1)^n_1
               * (1 + (f/f_1)^a_1)^((n_2-n_1)/a_1)
               * (1 + (f/f_2)^a_2)^((n_3-n_2)/a_2)

    where  N = (f_1/f_2)^n_1
               * (1 + (f_1/f_2)^(-a_1))^((n_1-n_2)/a_1)
               * 2^((n_2-n_3)/a_2)

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude at f_2, log10 f_1, log10 f_2,
               n_1, n_2, n_3, a_1, a_2].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
        pars[7],
    )
    amplitude = 10.0**log_amplitude
    ratio = 10.0 ** (log_f_1 - log_f_2)  # f_1 / f_2
    x_1 = freq / 10.0**log_f_1
    x_2 = freq / 10.0**log_f_2

    # normalisation so that Omega(f_2) = amplitude
    norm = (
        ratio**n_1
        * (1.0 + ratio ** (-a_1)) ** ((n_1 - n_2) / a_1)
        * 2.0 ** ((n_2 - n_3) / a_2)
    )

    return (
        norm
        * amplitude
        * x_1**n_1
        * (1.0 + x_1**a_1) ** ((n_2 - n_1) / a_1)
        * (1.0 + x_2**a_2) ** ((n_3 - n_2) / a_2)
    )


def d1double_broken_power_law(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``double_broken_power_law`` w.r.t. pars.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude at f_2, log10 f_1, log10 f_2,
               n_1, n_2, n_3, a_1, a_2].
    Returns:
        jax.Array of shape (N_freq, 8):
            0: d/d(log_A)  = model * ln(10)
            1: d/d(log_f1) = model * ln(10)*(n1-n2)*(x2^a1 - 1)
                           / ((1+r12^a1)*(1+x1^a1))
            2: d/d(log_f2) — complex, see code
            3: d/d(n1)     = model * ln((x1^a1 + x2^a1) / (1 + x1^a1)) / a1
            4: d/d(n2)     = model * [-ln(0.5*(1+x2^a2))/a2
                                + ln((r12^a1+x2^a1)/(1+r12^a1))/a1]
            5: d/d(n3)     = model * ln(0.5*(1+x2^a2)) / a2
            6: d/d(a1)     — complex, see code
            7: d/d(a2)     = model * (n3-n2)/a2^2/(1+x2^a2)
                           * [a2*x2^a2*ln(x2) - (1+x2^a2)*ln(0.5*(1+x2^a2))]
    """
    log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2 = (
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
        pars[7],
    )
    x_1 = freq / 10.0**log_f_1
    x_2 = freq / 10.0**log_f_2
    r_12 = 10.0 ** (log_f_1 - log_f_2)
    ln10 = jnp.log(10.0)
    model = double_broken_power_law(freq, pars)

    x1a1 = x_1**a_1
    x2a1 = x_2**a_1
    x2a2 = x_2**a_2
    r12a1 = r_12**a_1

    # index 0: d/d(log_A)
    d_logA = model * ln10

    # index 1: d/d(log_f1)
    d_logf1 = model * ln10 * (n_1 - n_2) * (x2a1 - 1.0) / ((1.0 + r12a1) * (1.0 + x1a1))

    # index 2: d/d(log_f2)
    num_f2 = (
        -n_1 * r12a1 - (n_3 + (n_1 + n_3) * r12a1) * x2a2 + n_2 * (r12a1 * x2a2 - 1.0)
    )
    d_logf2 = model * ln10 * num_f2 / ((1.0 + r12a1) * (1.0 + x2a2))

    # index 3: d/d(n1)
    d_n1 = model * jnp.log((x1a1 + x2a1) / (1.0 + x1a1)) / a_1

    # index 4: d/d(n2)
    d_n2 = model * (
        -jnp.log(0.5 * (1.0 + x2a2)) / a_2
        + jnp.log((r12a1 + x2a1) / (1.0 + r12a1)) / a_1
    )

    # index 5: d/d(n3)
    d_n3 = model * jnp.log(0.5 * (1.0 + x2a2)) / a_2

    # index 6: d/d(a1)
    term_r = 1.0 + r12a1
    term_1 = 1.0 + x1a1
    prefactor_a1 = (n_2 - n_1) / a_1**2 / term_r / term_1
    add1 = a_1 * jnp.log(r_12)
    add2 = term_r * term_1 * jnp.log(term_r / r12a1)
    add3 = -term_r * jnp.log(term_1)
    add4 = x2a1 * (a_1 * jnp.log(x_1) - jnp.log(term_1))
    add5 = x1a1 * (a_1 * jnp.log(x_2) - jnp.log(term_1))
    d_a1 = model * prefactor_a1 * (add1 + add2 + add3 + add4 + add5)

    # index 7: d/d(a2)
    prefactor_a2 = (n_3 - n_2) / a_2**2 / (1.0 + x2a2)
    d_a2 = (
        model
        * prefactor_a2
        * (a_2 * x2a2 * jnp.log(x_2) - (1.0 + x2a2) * jnp.log(0.5 * (1.0 + x2a2)))
    )

    return jnp.stack([d_logA, d_logf1, d_logf2, d_n1, d_n2, d_n3, d_a1, d_a2], axis=1)


double_broken_power_law_model = ut.Signal_model(
    "double_broken_power_law",
    double_broken_power_law,
    dtemplate=d1double_broken_power_law,
    model_label="Double Broken Power Law",
    parameter_names=[
        "log_amplitude",
        "log_f_1",
        "log_f_2",
        "n_1",
        "n_2",
        "n_3",
        "a_1",
        "a_2",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_1/\mathrm{Hz})$",
        r"$\log_{10}(f_2/\mathrm{Hz})$",
        r"$n_1$",
        r"$n_2$",
        r"$n_3$",
        r"$a_1$",
        r"$a_2$",
    ],
    prior={
        "log_amplitude": {"prior_type": "uniform", "minimum": -20.0, "maximum": -1.0},
        "log_f_1": {"prior_type": "uniform", "minimum": -10.0, "maximum": 0.0},
        "log_f_2": {"prior_type": "uniform", "minimum": -10.0, "maximum": 0.0},
        "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "n_3": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        "a_2": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
    },
)
