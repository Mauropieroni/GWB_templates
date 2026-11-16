r"""
Double-peak envelope modulated by a sharp-feature oscillation.

Two parametrizations:

* :class:`DoublePeakSharp` — linear ``A_sharp``, ``omega_sharp_Hz``.
* :class:`DoublePeakSharpLog` — log-scaled amplitude and frequency for wide-range
  priors.

The template is the product of the double-peak spectrum and a sharp-feature modulation:

.. math::

    \Omega_{\rm GW} h^2(f) =
        \mathrm{DoublePeak}(f; \dots)\,
        \bigl[1 + A_{\rm sharp}\cos(\omega f + \theta)\bigr]

Both classes hold a :class:`DoublePeak` envelope and a :class:`SharpFeature` (or
:class:`SharpFeatureLog`) modulation instance, and combine them via the product rule.

Reference: arXiv:2407.04356.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.inflation_templates.double_peak import DoublePeak
from gwb_templates.inflation_templates.sharp_feature import (
    SharpFeature,
    SharpFeatureLog,
)
from gwb_templates.template import AnalyticTemplate

_ENVELOPE_LABELS = DoublePeak.DEFAULT_PARAMETER_LABELS
_ENVELOPE_PRIORS = DoublePeak.DEFAULT_PRIOR_BY_PARAM


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

    DEFAULT_C1: ClassVar[jax.Array] = DoublePeak.DEFAULT_C1
    DEFAULT_TILT_P: ClassVar[jax.Array] = DoublePeak.DEFAULT_TILT_P

    DEFAULT_MODEL_NAME: ClassVar[str] = "double_peak_sharp"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Double Peak + Sharp Feature"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        **_ENVELOPE_LABELS,
        "A_sharp": r"$A_{\rm s}$",
        "omega_sharp_Hz": r"$\omega_{\rm s}\,[\mathrm{Hz}^{-1}]$",
        "phase_sharp": r"$\phi_{\rm s}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        **_ENVELOPE_PRIORS,
        "A_sharp": {"min": -1.0, "max": 1.0},
        "omega_sharp_Hz": {"min": 0.0, "max": 1e5},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    }

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
        c1: jax.Array = DEFAULT_C1,
        tilt_p: jax.Array = DEFAULT_TILT_P,
        **kwargs: Any,
    ) -> None:
        # Sub-template instances; reused on every call. Both are pure JAX,
        # so this is safe to construct once at init time.
        self._envelope = DoublePeak(c1=c1, tilt_p=tilt_p)
        self._modulation = SharpFeature()
        super().__init__(**kwargs)

    @property
    def c1(self) -> jax.Array:
        return self._envelope.c1

    @property
    def tilt_p(self) -> jax.Array:
        return self._envelope.tilt_p

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_pivot: jax.Array,
        beta: jax.Array,
        k1: jax.Array,
        k2: jax.Array,
        rho: jax.Array,
        gamma: jax.Array,
        A_sharp: jax.Array,
        omega_sharp_Hz: jax.Array,
        phase_sharp: jax.Array,
    ) -> jax.Array:
        """Evaluate the double-peak x sharp-feature product at ``frequency``."""
        envelope = self._envelope.omega_gw_h2(
            frequency, log_amplitude, log_pivot, beta, k1, k2, rho, gamma
        )
        modulation = self._modulation.omega_gw_h2(
            frequency, A_sharp, omega_sharp_Hz, phase_sharp
        )
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on double-peak x sharp-feature."""
        E = self._envelope.omega_gw_h2(frequency, *theta[:7])
        dE = self._envelope._grad_theta_omega_gw_h2_analytical(frequency, theta[:7])
        F = self._modulation.omega_gw_h2(frequency, theta[7], theta[8], theta[9])
        dF = self._modulation._grad_theta_omega_gw_h2_analytical(frequency, theta[7:10])
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)


class DoublePeakSharpLog(AnalyticTemplate):
    r"""
    Double-peak envelope multiplied by a log-parametrized sharp-feature modulation.
    Identical physics to :class:`DoublePeakSharp`.

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

    DEFAULT_C1: ClassVar[jax.Array] = DoublePeak.DEFAULT_C1
    DEFAULT_TILT_P: ClassVar[jax.Array] = DoublePeak.DEFAULT_TILT_P

    DEFAULT_MODEL_NAME: ClassVar[str] = "double_peak_sharp_log"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Double Peak + Sharp Feature (log params)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        **_ENVELOPE_LABELS,
        "log_A_sharp": r"$\log_{10}A_{\rm s}$",
        "log_omega_sharp_Hz": r"$\log_{10}(\omega_{\rm s}/\mathrm{Hz}^{-1})$",
        "phase_sharp": r"$\phi_{\rm s}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        **_ENVELOPE_PRIORS,
        "log_A_sharp": {"min": -3.0, "max": 0.0},
        "log_omega_sharp_Hz": {"min": 0.0, "max": 5.0},
        "phase_sharp": {"min": -3.14159, "max": 3.14159},
    }

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
        c1: jax.Array = DEFAULT_C1,
        tilt_p: jax.Array = DEFAULT_TILT_P,
        **kwargs: Any,
    ) -> None:
        # Sub-template instances; reused on every call. Both are pure JAX,
        # so this is safe to construct once at init time.
        self._envelope = DoublePeak(c1=c1, tilt_p=tilt_p)
        self._modulation = SharpFeatureLog()
        super().__init__(**kwargs)

    @property
    def c1(self) -> jax.Array:
        return self._envelope.c1

    @property
    def tilt_p(self) -> jax.Array:
        return self._envelope.tilt_p

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_pivot: jax.Array,
        beta: jax.Array,
        k1: jax.Array,
        k2: jax.Array,
        rho: jax.Array,
        gamma: jax.Array,
        log_A_sharp: jax.Array,
        log_omega_sharp_Hz: jax.Array,
        phase_sharp: jax.Array,
    ) -> jax.Array:
        """Evaluate the double-peak x log sharp-feature product at ``frequency``."""
        envelope = self._envelope.omega_gw_h2(
            frequency, log_amplitude, log_pivot, beta, k1, k2, rho, gamma
        )
        modulation = self._modulation.omega_gw_h2(
            frequency, log_A_sharp, log_omega_sharp_Hz, phase_sharp
        )
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on double-peak x log sharp-feature."""
        E = self._envelope.omega_gw_h2(frequency, *theta[:7])
        dE = self._envelope._grad_theta_omega_gw_h2_analytical(frequency, theta[:7])
        F = self._modulation.omega_gw_h2(frequency, theta[7], theta[8], theta[9])
        dF = self._modulation._grad_theta_omega_gw_h2_analytical(frequency, theta[7:10])
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)
