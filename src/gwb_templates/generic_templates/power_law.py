"""
Standard power-law spectrum template.

Two-parameter model for a power-law GWB:

    Omega(f) = 10^A * (f / f_pivot)^tilt

The default pivot frequency is 3 mHz (close to the centre of the LISA band).
This is the simplest phenomenological model for a GWB signal and is widely
used as a baseline in GWB searches.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

# Local
from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# A simple PL model for the signal
def power_law(
    freq: ArrayLike,
    pars: ParamLike,
    pivot: float = 3e-3,
) -> jax.Array:
    """
    Evaluate a power-law spectrum.

    Args:
        freq: Frequency grid where the model is evaluated.
        pars: Model parameters [log10 amplitude, spectral index].
        pivot: Pivot frequency used for normalization.

    Returns:
        Array with the model value at each input frequency.
    """

    # Unpack parameters
    log_amplitude, tilt = pars

    # Normalize frequencies with respect to the pivot value
    x = freq / pivot

    # Build the power-law spectrum
    return 10.0**log_amplitude * x**tilt


def d1power_law(
    frequency: ArrayLike,
    parameters: ParamLike,
    pivot: float = 3e-3,
) -> jax.Array:
    """
    Evaluate the first derivative of the power-law spectrum.

    Args:
        index: Parameter index (0 for log-amplitude, 1 for tilt).
        frequency: Frequency grid.
        parameters: Parameter vector [log10 amplitude, spectral index].
        pivot: Pivot frequency used for normalization.

    Returns:
        Gradient evaluated on frequency for all selected parameters.
        shape: (len(frequency), npars)
    """

    # compute the model
    model = power_law(frequency, parameters, pivot=pivot)

    dmodel_dlnA = model * jnp.full_like(frequency, jnp.log(10.0))
    # derivative of the log of the model w.r.t the tilt
    dmodel_dtilt = model * jnp.log(frequency / pivot)

    # stack the log derivatives along the last axis to get shape (len(frequency), npars)
    gradient = jnp.stack((dmodel_dlnA, dmodel_dtilt), axis=-1)

    # return the gradient with the parameter axis last: (len(frequency), npars)
    return gradient


# Initialize the signal model
power_law_model = ut.Signal_model(
    "power_law",
    power_law,
    dtemplate=d1power_law,
    model_label="Power Law Model",
    parameter_names=["log_amplitude", "tilt"],
    parameter_labels=[r"$\alpha_{\rm PL}$", r"$n_{\rm T}$"],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "tilt": {"min": -10.0, "max": 10.0},
    },
)
