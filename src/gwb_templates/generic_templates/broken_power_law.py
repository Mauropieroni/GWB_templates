"""
Smooth broken-power-law (BPL) spectrum template.

General-purpose 5-parameter smooth BPL used throughout the GWB template
library.  The transition width is controlled by an additional parameter
``log_transition = log10(delta)``:

    Omega(f) = A * x^n1 / (0.5 * (1 + x^(1/delta)))^((n1 - n2) * delta)

where  x = f / f_*  and  delta = 10^log_transition.

At low frequencies the tilt approaches n1; at high frequencies n2.
Setting ``log_transition = 0`` (delta = 1) recovers
``broken_power_law_fixed_smoothness``.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


def broken_power_law(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Smooth broken power law with a tunable transition width (5 parameters).

    Omega(f) = A * x^n1 / (0.5 * (1 + x^(1/delta)))^((n1-n2)*delta)
    where x = f / f_*  and  delta = 10^log_transition.

    At low frequencies the tilt is n1; at high frequencies it is n2.
    The transition sharpness is controlled by delta.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, tilt_1, tilt_2, log10 transition].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_pivot, tilt_1, tilt_2, log_transition = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    x = freq / 10.0**log_pivot
    delta = 10.0**log_transition
    return (
        10.0**log_amplitude
        * x**tilt_1
        / (0.5 * (1.0 + x ** (1.0 / delta))) ** ((tilt_1 - tilt_2) * delta)
    )


def d1broken_power_law(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``broken_power_law`` w.r.t. pars.

    Let t = x^(1/delta), x = f/f_*.
    Args:
        freq: Frequency grid.
        pars: [log10 amplitude, log10 pivot, tilt_1, tilt_2, log10 transition].
    Returns:
        jax.Array of shape (N_freq, 5):
            d/d(log_A)         = model * ln(10)
            d/d(log_pivot)     = model * ln(10) * (-n1 - n2*t) / (1+t)
            d/d(tilt_1)        = model * [ln(x) + delta*(ln(2) - ln(1+t))]
            d/d(tilt_2)        = model * delta * (ln(1+t) - ln(2))
            d/d(log_transition)= model * ln(10)*(n1-n2)
                               * [t*ln(x)/(1+t)
                                    - delta*ln(0.5*(1+t))]
    """
    log_pivot, tilt_1, tilt_2, log_transition = pars[1], pars[2], pars[3], pars[4]
    x = freq / 10.0**log_pivot
    delta = 10.0**log_transition
    t = x ** (1.0 / delta)
    model = broken_power_law(freq, pars)
    ln10 = jnp.log(10.0)

    d_logA = model * ln10
    d_logpiv = model * ln10 * (-tilt_1 - tilt_2 * t) / (1.0 + t)
    d_t1 = model * (jnp.log(x) + delta * (jnp.log(2.0) - jnp.log(1.0 + t)))
    d_t2 = model * delta * (jnp.log(1.0 + t) - jnp.log(2.0))
    d_logtrans = (
        model
        * ln10
        * (tilt_1 - tilt_2)
        * (t * jnp.log(x) / (1.0 + t) - delta * jnp.log(0.5 * (1.0 + t)))
    )

    return jnp.stack([d_logA, d_logpiv, d_t1, d_t2, d_logtrans], axis=1)


broken_power_law_model = ut.Signal_model(
    "broken_power_law",
    broken_power_law,
    dtemplate=d1broken_power_law,
    model_label="Broken Power Law",
    parameter_names=[
        "log_amplitude",
        "log_pivot",
        "tilt_1",
        "tilt_2",
        "log_transition",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$n_1$",
        r"$n_2$",
        r"$\log_{10}\delta$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "tilt_1": {"min": -10.0, "max": 10.0},
        "tilt_2": {"min": -10.0, "max": 10.0},
        "log_transition": {"min": -3.0, "max": 3.0},
    },
)
