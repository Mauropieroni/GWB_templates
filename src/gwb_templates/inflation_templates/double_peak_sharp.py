r"""
Double-peak envelope modulated by a sharp-feature oscillation.

Two parametrizations:

* :class:`DoublePeakSharp` — linear ``A_sharp``, ``omega_sharp_Hz``.
* :class:`DoublePeakSharpLog` — log-scaled amplitude and frequency for
  wide-range priors.

The template is the product of the double-peak spectrum and a sharp-feature
modulation:

.. math::

    \Omega_{\rm GW} h^2(f) =
        \mathrm{DoublePeak}(f; \dots)\,
        \bigl[1 + A_{\rm sharp}\cos(\omega f + \theta)\bigr]

Reference: arXiv:2407.04356.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


def _double_peak_envelope_and_grad(
    frequency: ArrayLike,
    log_amplitude: ArrayLike,
    log_pivot: ArrayLike,
    beta: ArrayLike,
    k1: ArrayLike,
    k2: ArrayLike,
    rho: ArrayLike,
    gamma: ArrayLike,
    c1: float,
    tilt_p: float,
) -> tuple[jax.Array, jax.Array]:
    """Return (envelope, dE/dpars) with pars ordered like the envelope signature."""
    amp = 10.0**log_amplitude
    pivot = 10.0**log_pivot

    x = jnp.asarray(frequency) / pivot
    x1 = x / k1
    c1_k1 = c1 / k1
    log10x2 = jnp.log10(x / k2)
    ln10 = jnp.log(10.0)

    arg1 = jnp.abs((c1_k1 - x1) / (c1_k1 - 1.0))
    exp1 = tilt_p * (c1_k1 - 1.0)
    bump_factor = arg1**exp1
    H = jnp.heaviside(c1_k1 - x1, 1.0)
    first_term = beta * x1**tilt_p * bump_factor * H

    log_normal = jnp.exp(-0.5 * (log10x2 / rho) ** 2)
    erfc_val = jax.scipy.special.erfc(gamma * log10x2)
    erfc_deriv = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-((gamma * log10x2) ** 2))
    second_term = log_normal * erfc_val

    envelope = amp * (first_term + second_term)

    d_logA = ln10 * envelope
    sign_c1_k1 = jnp.where(c1_k1 >= 1.0, 1.0, -1.0)
    d_first_logpiv = (
        amp
        * first_term
        * (-tilt_p * ln10 + sign_c1_k1 * exp1 / arg1 * (x1 / (c1_k1 - 1.0)) * ln10)
    )
    d_second_logpiv = (
        amp * second_term * log10x2 / rho**2
        + amp * log_normal * gamma * erfc_deriv
    )
    d_logpiv = d_first_logpiv + d_second_logpiv
    d_beta = amp * first_term / beta
    log_arg1 = jnp.log(arg1)
    d_k1 = (
        amp
        * first_term
        * (-tilt_p / k1 - log_arg1 * tilt_p * c1 / k1**2 + exp1 / (c1 - k1))
    )
    d_k2 = (
        amp * log_normal * erfc_deriv * gamma / k2
        + amp * second_term * log10x2 / (rho**2 * k2)
    ) / ln10
    d_rho = amp * second_term * log10x2**2 / rho**3
    d_gamma = amp * log_normal * erfc_deriv * (-log10x2)

    dE = jnp.stack(
        [d_logA, d_logpiv, d_beta, d_k1, d_k2, d_rho, d_gamma], axis=-1
    )
    return envelope, dE


def _double_peak_envelope(
    frequency: ArrayLike,
    log_amplitude: ArrayLike,
    log_pivot: ArrayLike,
    beta: ArrayLike,
    k1: ArrayLike,
    k2: ArrayLike,
    rho: ArrayLike,
    gamma: ArrayLike,
    c1: float,
    tilt_p: float,
) -> jax.Array:
    """Internal pure-JAX double-peak envelope used by the composite classes."""
    amplitude = 10.0**log_amplitude
    pivot = 10.0**log_pivot

    x = frequency / pivot
    x1 = x / k1
    c1_k1 = c1 / k1
    log10x2 = jnp.log10(x / k2)

    bump_factor = jnp.abs((c1_k1 - x1) / (c1_k1 - 1.0)) ** (tilt_p * (c1_k1 - 1.0))
    first_term = beta * x1**tilt_p * bump_factor * jnp.heaviside(c1_k1 - x1, 1.0)

    log_normal = jnp.exp(-0.5 * (log10x2 / rho) ** 2)
    erfc_term = jax.scipy.special.erfc(gamma * log10x2)
    second_term = log_normal * erfc_term

    return amplitude * (first_term + second_term)


_ENVELOPE_LABELS = {
    "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
    "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
    "beta": r"$\beta$",
    "k1": r"$\kappa_1$",
    "k2": r"$\kappa_2$",
    "rho": r"$\rho$",
    "gamma": r"$\gamma$",
}
_ENVELOPE_PRIORS = {
    "log_amplitude": {"min": -20.0, "max": -5.0},
    "log_pivot": {"min": -5.0, "max": 0.0},
    "beta": {"min": 0.0, "max": 10.0},
    "k1": {"min": 0.1, "max": 10.0},
    "k2": {"min": 0.1, "max": 10.0},
    "rho": {"min": 0.01, "max": 5.0},
    "gamma": {"min": -5.0, "max": 5.0},
}


class DoublePeakSharp(AnalyticTemplate):
    r"""
    Double-peak envelope multiplied by a linear sharp-feature modulation.

    Free parameters
    ---------------
    log_amplitude, log_pivot, beta, k1, k2, rho, gamma
        Envelope parameters (see :class:`DoublePeak`).
    A_sharp
        Linear amplitude of the sharp oscillation.
    omega_sharp_Hz
        Angular frequency of the oscillation (Hz\ :sup:`-1`).
    phase_sharp
        Phase offset (radians).
    """

    DEFAULT_C1: ClassVar[float] = math.sqrt(2.0 / 3.0)
    DEFAULT_TILT_P: ClassVar[float] = 2.5

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{LISACosmologyWorkingGroup:2024hsc,
    author = "Braglia, Matteo and others",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from inflation in LISA: reconstruction pipeline and
        physics interpretation}",
    eprint = "2407.04356",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-03, CERN-TH-2024-072",
    doi = "10.1088/1475-7516/2024/11/032",
    journal = "JCAP",
    volume = "11",
    pages = "032",
    year = "2024"
}
""",
    )

    def __init__(
        self,
        c1: float = DEFAULT_C1,
        tilt_p: float = DEFAULT_TILT_P,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.c1: float = float(c1)
        self.tilt_p: float = float(tilt_p)

        default_labels = {
            **_ENVELOPE_LABELS,
            "A_sharp": r"$A_{\rm s}$",
            "omega_sharp_Hz": r"$\omega_{\rm s}\,[\mathrm{Hz}^{-1}]$",
            "phase_sharp": r"$\phi_{\rm s}$",
        }
        default_priors = {
            **_ENVELOPE_PRIORS,
            "A_sharp": {"min": -1.0, "max": 1.0},
            "omega_sharp_Hz": {"min": 0.0, "max": 1e5},
            "phase_sharp": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Double Peak + Sharp Feature"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_amplitude: ArrayLike,
        log_pivot: ArrayLike,
        beta: ArrayLike,
        k1: ArrayLike,
        k2: ArrayLike,
        rho: ArrayLike,
        gamma: ArrayLike,
        A_sharp: ArrayLike,
        omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        envelope = _double_peak_envelope(
            frequency,
            log_amplitude,
            log_pivot,
            beta,
            k1,
            k2,
            rho,
            gamma,
            self.c1,
            self.tilt_p,
        )
        modulation = 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * frequency + phase_sharp)
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on envelope x sharp-feature."""
        freq = jnp.asarray(frequency)
        E, dE = _double_peak_envelope_and_grad(
            freq, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
            theta[6], self.c1, self.tilt_p,
        )
        A_sharp, omega_sharp_Hz, phase_sharp = theta[7], theta[8], theta[9]
        arg = omega_sharp_Hz * freq + phase_sharp
        F = 1.0 + A_sharp * jnp.cos(arg)
        d_A = jnp.cos(arg)
        d_omega = -A_sharp * jnp.sin(arg) * freq
        d_phi = -A_sharp * jnp.sin(arg)
        dF = jnp.stack([d_A, d_omega, d_phi], axis=-1)
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)


class DoublePeakSharpLog(AnalyticTemplate):
    r"""
    Double-peak envelope multiplied by a log-parametrized sharp-feature
    modulation. Identical physics to :class:`DoublePeakSharp`.

    Free parameters
    ---------------
    log_amplitude, log_pivot, beta, k1, k2, rho, gamma
        Envelope parameters.
    log_A_sharp
        :math:`\log_{10}` amplitude of the sharp oscillation.
    log_omega_sharp_Hz
        :math:`\log_{10}` angular frequency.
    phase_sharp
        Phase offset (radians).
    """

    DEFAULT_C1: ClassVar[float] = math.sqrt(2.0 / 3.0)
    DEFAULT_TILT_P: ClassVar[float] = 2.5

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{LISACosmologyWorkingGroup:2024hsc,
    author = "Braglia, Matteo and others",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from inflation in LISA: reconstruction pipeline and
        physics interpretation}",
    eprint = "2407.04356",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-03, CERN-TH-2024-072",
    doi = "10.1088/1475-7516/2024/11/032",
    journal = "JCAP",
    volume = "11",
    pages = "032",
    year = "2024"
}
""",
    )

    def __init__(
        self,
        c1: float = DEFAULT_C1,
        tilt_p: float = DEFAULT_TILT_P,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.c1: float = float(c1)
        self.tilt_p: float = float(tilt_p)

        default_labels = {
            **_ENVELOPE_LABELS,
            "log_A_sharp": r"$\log_{10}A_{\rm s}$",
            "log_omega_sharp_Hz": r"$\log_{10}(\omega_{\rm s}/\mathrm{Hz}^{-1})$",
            "phase_sharp": r"$\phi_{\rm s}$",
        }
        default_priors = {
            **_ENVELOPE_PRIORS,
            "log_A_sharp": {"min": -3.0, "max": 0.0},
            "log_omega_sharp_Hz": {"min": 0.0, "max": 5.0},
            "phase_sharp": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Double Peak + Sharp Feature (log params)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_amplitude: ArrayLike,
        log_pivot: ArrayLike,
        beta: ArrayLike,
        k1: ArrayLike,
        k2: ArrayLike,
        rho: ArrayLike,
        gamma: ArrayLike,
        log_A_sharp: ArrayLike,
        log_omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        envelope = _double_peak_envelope(
            frequency,
            log_amplitude,
            log_pivot,
            beta,
            k1,
            k2,
            rho,
            gamma,
            self.c1,
            self.tilt_p,
        )
        A_sharp = 10.0**log_A_sharp
        omega_sharp_Hz = 10.0**log_omega_sharp_Hz
        modulation = 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * frequency + phase_sharp)
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on envelope x log sharp-feature."""
        freq = jnp.asarray(frequency)
        E, dE = _double_peak_envelope_and_grad(
            freq, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
            theta[6], self.c1, self.tilt_p,
        )
        log_A_sharp, log_omega_sharp_Hz, phase_sharp = theta[7], theta[8], theta[9]
        A_sharp = 10.0**log_A_sharp
        omega_sharp_Hz = 10.0**log_omega_sharp_Hz
        arg = omega_sharp_Hz * freq + phase_sharp
        F = 1.0 + A_sharp * jnp.cos(arg)
        ln10 = jnp.log(10.0)
        d_logA = ln10 * A_sharp * jnp.cos(arg)
        d_logomega = -ln10 * A_sharp * omega_sharp_Hz * jnp.sin(arg) * freq
        d_phi = -A_sharp * jnp.sin(arg)
        dF = jnp.stack([d_logA, d_logomega, d_phi], axis=-1)
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)
