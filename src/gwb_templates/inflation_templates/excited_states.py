"""
Excited-states inflationary GWB template.

Models the GWB produced by excited states during inflation, featuring a
power-law rise with a sharp Heaviside cutoff at x_cut = 2 * gamma_ES, where
x = 0.5 * f * omega_ES.

The spectrum is:

    Omega(f) = (10^A / 0.052) * x^{-3}
             * (1 - (x / (2*gamma_ES))^2)^2
             * (sin(x) - 4*sin(x/2)^2 / x)^2
             * Heaviside(2*gamma_ES - x)

where
    x          = 0.5 * f * 10^log_omega_ES
    gamma_ES   = 10^log_gamma_ES

All operations are pure JAX so automatic differentiation is fully supported.
The Heaviside is treated as a hard cutoff; its JVP w.r.t. the threshold
parameter (log_gamma_ES) ignores the boundary delta-function contribution,
consistent with the reference analytical derivative.

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


def excited_states(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Excited-states inflationary spectrum.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 gamma_ES, log10 omega_ES].
              log_gamma_ES and log_omega_ES set the cutoff and oscillation
              frequency of the spectrum respectively.
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_gamma_ES, log_omega_ES = pars[0], pars[1], pars[2]

    amplitude = 10.0**log_amplitude
    gamma_ES = 10.0**log_gamma_ES

    x = 0.5 * jnp.asarray(freq) * 10.0**log_omega_ES
    x_cut = 2.0 * gamma_ES

    factor_1 = amplitude / 0.052 * x ** (-3)
    factor_2 = (1.0 - (x / (2.0 * gamma_ES)) ** 2) ** 2
    factor_3 = (jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x) ** 2

    return factor_1 * factor_2 * factor_3 * jnp.heaviside(x_cut - x, 1.0)


def d1excited_states(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``excited_states`` w.r.t. pars.

    The Heaviside boundary term (delta-function contribution at x = x_cut)
    is ignored, which is consistent with JAX's treatment of ``jnp.heaviside``
    and numerically irrelevant for sampling.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, log10 gamma_ES, log10 omega_ES].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(log_amplitude) = model * ln(10)
            d/d(log_gamma_ES)  = factor_1 * factor_3 * H * 4*ln(10)*u^2*(1-u^2)
            d/d(log_omega_ES)  = chain rule through x = 0.5*f*10^log_omega_ES
    """
    log_amplitude, log_gamma_ES, log_omega_ES = pars[0], pars[1], pars[2]

    amplitude = 10.0**log_amplitude
    gamma_ES = 10.0**log_gamma_ES

    fvec = jnp.asarray(freq)
    x = 0.5 * fvec * 10.0**log_omega_ES
    x_cut = 2.0 * gamma_ES
    u = x / x_cut  # = x / (2 * gamma_ES)

    H = jnp.heaviside(x_cut - x, 1.0)
    factor_1 = amplitude / 0.052 * x ** (-3)
    factor_2 = (1.0 - u**2) ** 2
    g = jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x
    factor_3 = g**2
    model = factor_1 * factor_2 * factor_3 * H

    ln10 = jnp.log(10.0)

    # d/d(log_amplitude): model ∝ amplitude = 10^log_amplitude
    d_logA = model * ln10

    # d/d(log_gamma_ES): only factor_2 depends on gamma_ES (Heaviside boundary ignored)
    # u = x/(2*gamma_ES),  d(u)/d(log_gamma_ES) = -u*ln10
    # d(factor_2)/d(log_gamma_ES) = 2*(1-u^2)*(-2u)*(-u*ln10) = 4*ln10*u^2*(1-u^2)
    d_lgam = factor_1 * factor_3 * H * 4.0 * ln10 * u**2 * (1.0 - u**2)

    # d/d(log_omega_ES): x = 0.5*f*10^log_omega_ES,  d(x)/d(log_omega_ES) = x*ln10
    # d(factor_1)/d(log_omega_ES) = factor_1 * (-3) * ln10   [factor_1 ∝ x^{-3}]
    # d(factor_2)/d(log_omega_ES) = -4*ln10*u^2*(1-u^2)      [u ∝ x]
    # d(factor_3)/d(log_omega_ES) = 2*g * dg/dx * x*ln10
    #   dg/dx = cos(x) - 2*sin(x)/x + 2*(1-cos(x))/x^2
    dg_dx = jnp.cos(x) - 2.0 * jnp.sin(x) / x + 2.0 * (1.0 - jnp.cos(x)) / x**2
    d_lomega = H * (
        factor_1 * (-3.0 * ln10) * factor_2 * factor_3
        + factor_1 * (-4.0 * ln10 * u**2 * (1.0 - u**2)) * factor_3
        + factor_1 * factor_2 * (2.0 * g * dg_dx * x * ln10)
    )

    return jnp.stack([d_logA, d_lgam, d_lomega], axis=1)


excited_states_model = ut.Signal_model(
    "excited_states",
    excited_states,
    dtemplate=d1excited_states,
    model_label="Excited States",
    parameter_names=["log_amplitude", "log_gamma_ES", "log_omega_ES"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(\gamma_{\rm ES})$",
        r"$\log_{10}(\omega_{\rm ES}\,\mathrm{Hz})$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_gamma_ES": {"min": -3.0, "max": 3.0},
        "log_omega_ES": {"min": 0.0, "max": 12.0},
    },
)
