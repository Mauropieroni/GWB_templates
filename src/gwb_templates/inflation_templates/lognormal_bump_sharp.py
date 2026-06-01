r"""
Lognormal-bump envelope modulated by a sharp-feature oscillation.

Two parametrizations:

* :class:`LognormalBumpSharp` — linear ``A_sharp``, ``omega_sharp_Hz``.
* :class:`LognormalBumpSharpLog` — log-scaled amplitude and frequency.

.. math::

    \Omega_{\rm GW} h^2(f) =
        \mathrm{LognormalBump}(f; A, f_*, \sigma)\,
        \bigl[1 + A_{\rm sharp}\cos(\omega f + \theta)\bigr]

Reference: arXiv:2407.04356.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


def _lognormal_bump_envelope(
    frequency: ArrayLike,
    log_amplitude: ArrayLike,
    log_pivot: ArrayLike,
    log_width: ArrayLike,
) -> jax.Array:
    """Internal pure-JAX lognormal bump envelope."""
    amplitude = 10.0**log_amplitude
    pivot = 10.0**log_pivot
    width = 10.0**log_width
    return amplitude * jnp.exp(-0.5 * (jnp.log10(frequency / pivot) / width) ** 2)


def _lognormal_bump_envelope_and_grad(
    frequency: ArrayLike,
    log_amplitude: ArrayLike,
    log_pivot: ArrayLike,
    log_width: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Return (envelope, d envelope / d[log_A, log_pivot, log_width])."""
    pivot = 10.0**log_pivot
    width = 10.0**log_width
    envelope = _lognormal_bump_envelope(frequency, log_amplitude, log_pivot, log_width)
    u = jnp.log10(jnp.asarray(frequency) / pivot)
    ln10 = jnp.log(10.0)
    d_logA = envelope * ln10
    d_logpiv = envelope * u / width**2
    d_logwid = envelope * u**2 * ln10 / width**2
    dE = jnp.stack([d_logA, d_logpiv, d_logwid], axis=-1)
    return envelope, dE


_ENVELOPE_LABELS = {
    "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
    "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
    "log_width": r"$\log_{10}\sigma$",
}
_ENVELOPE_PRIORS = {
    "log_amplitude": {"min": -20.0, "max": -5.0},
    "log_pivot": {"min": -5.0, "max": 0.0},
    "log_width": {"min": -2.0, "max": 1.0},
}


class LognormalBumpSharp(AnalyticTemplate):
    r"""
    Lognormal bump envelope multiplied by a linear sharp-feature modulation.

    Free parameters
    ---------------
    log_amplitude, log_pivot, log_width
        Envelope parameters.
    A_sharp
        Linear amplitude of the sharp oscillation.
    omega_sharp_Hz
        Angular frequency (Hz\ :sup:`-1`).
    phase_sharp
        Phase offset (radians).
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
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
                else "Lognormal Bump + Sharp Feature"
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
        log_width: ArrayLike,
        A_sharp: ArrayLike,
        omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        envelope = _lognormal_bump_envelope(
            frequency, log_amplitude, log_pivot, log_width
        )
        modulation = 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * frequency + phase_sharp)
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on lognormal-bump x sharp-feature."""
        freq = jnp.asarray(frequency)
        E, dE = _lognormal_bump_envelope_and_grad(freq, theta[0], theta[1], theta[2])
        A_sharp, omega_sharp_Hz, phase_sharp = theta[3], theta[4], theta[5]
        arg = omega_sharp_Hz * freq + phase_sharp
        F = 1.0 + A_sharp * jnp.cos(arg)
        d_A = jnp.cos(arg)
        d_omega = -A_sharp * jnp.sin(arg) * freq
        d_phi = -A_sharp * jnp.sin(arg)
        dF = jnp.stack([d_A, d_omega, d_phi], axis=-1)
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)


class LognormalBumpSharpLog(AnalyticTemplate):
    r"""
    Lognormal bump envelope multiplied by a log-parametrized sharp-feature
    modulation.

    Free parameters
    ---------------
    log_amplitude, log_pivot, log_width
        Envelope parameters.
    log_A_sharp
        :math:`\log_{10}` amplitude of the sharp oscillation.
    log_omega_sharp_Hz
        :math:`\log_{10}` angular frequency.
    phase_sharp
        Phase offset (radians).
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
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
                else "Lognormal Bump + Sharp Feature (log params)"
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
        log_width: ArrayLike,
        log_A_sharp: ArrayLike,
        log_omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        envelope = _lognormal_bump_envelope(
            frequency, log_amplitude, log_pivot, log_width
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
        """Analytic Jacobian via product rule on lognormal-bump x log sharp-feature."""
        freq = jnp.asarray(frequency)
        E, dE = _lognormal_bump_envelope_and_grad(freq, theta[0], theta[1], theta[2])
        log_A_sharp, log_omega_sharp_Hz, phase_sharp = theta[3], theta[4], theta[5]
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
