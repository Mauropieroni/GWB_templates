r"""
Lognormal-bump envelope modulated by a sharp-feature oscillation.

Two parametrizations:

* :class:`LognormalBumpSharp` — linear ``A_sharp``, ``omega_sharp_Hz``.
* :class:`LognormalBumpSharpLog` — log-scaled amplitude and frequency.

.. math::

    \Omega_{\rm GW} h^2(f) =
        \mathrm{LognormalBump}(f; A, f_*, \sigma)\,
        \bigl[1 + A_{\rm sharp}\cos(\omega f + \theta)\bigr]

Both classes hold a :class:`LognormalBump` envelope and a :class:`SharpFeature` (or
:class:`SharpFeatureLog`) modulation instance, and combine them via the product rule.

Reference: arXiv:2407.04356.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.lognormal_bump import LognormalBump
from gwb_templates.inflation_templates.sharp_feature import (
    SharpFeature,
    SharpFeatureLog,
)
from gwb_templates.template import AnalyticTemplate

_ENVELOPE_LABELS = LognormalBump.DEFAULT_PARAMETER_LABELS
_ENVELOPE_PRIORS = LognormalBump.DEFAULT_PRIOR_BY_PARAM


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

    DEFAULT_MODEL_NAME: ClassVar[str] = "lognormal_bump_sharp"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Lognormal Bump + Sharp Feature"
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

    def __init__(self, **kwargs: Any) -> None:
        # Sub-template instances; reused on every call. Both are pure JAX,
        # so this is safe to construct once at init time.
        self._envelope = LognormalBump()
        self._modulation = SharpFeature()
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_pivot: jax.Array,
        log_width: jax.Array,
        A_sharp: jax.Array,
        omega_sharp_Hz: jax.Array,
        phase_sharp: jax.Array,
    ) -> jax.Array:
        """Evaluate the lognormal bump x sharp-feature product at ``frequency``."""
        envelope = self._envelope.omega_gw_h2(
            frequency, log_amplitude, log_pivot, log_width
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
        """Analytic Jacobian via product rule on lognormal-bump x sharp-feature."""
        E = self._envelope.omega_gw_h2(frequency, theta[0], theta[1], theta[2])
        dE = self._envelope._grad_theta_omega_gw_h2_analytical(frequency, theta[:3])
        F = self._modulation.omega_gw_h2(frequency, theta[3], theta[4], theta[5])
        dF = self._modulation._grad_theta_omega_gw_h2_analytical(frequency, theta[3:6])
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)


class LognormalBumpSharpLog(AnalyticTemplate):
    r"""
    Lognormal bump envelope multiplied by a log-parametrized sharp-feature modulation.

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

    DEFAULT_MODEL_NAME: ClassVar[str] = "lognormal_bump_sharp_log"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Lognormal Bump + Sharp Feature (log params)"
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

    def __init__(self, **kwargs: Any) -> None:
        # Sub-template instances; reused on every call. Both are pure JAX,
        # so this is safe to construct once at init time.
        self._envelope = LognormalBump()
        self._modulation = SharpFeatureLog()
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_pivot: jax.Array,
        log_width: jax.Array,
        log_A_sharp: jax.Array,
        log_omega_sharp_Hz: jax.Array,
        phase_sharp: jax.Array,
    ) -> jax.Array:
        """Evaluate the lognormal bump x log sharp-feature product at ``frequency``."""
        envelope = self._envelope.omega_gw_h2(
            frequency, log_amplitude, log_pivot, log_width
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
        """Analytic Jacobian via product rule on lognormal-bump x log sharp-feature."""
        E = self._envelope.omega_gw_h2(frequency, theta[0], theta[1], theta[2])
        dE = self._envelope._grad_theta_omega_gw_h2_analytical(frequency, theta[:3])
        F = self._modulation.omega_gw_h2(frequency, theta[3], theta[4], theta[5])
        dF = self._modulation._grad_theta_omega_gw_h2_analytical(frequency, theta[3:6])
        return jnp.concatenate([dE * F[..., None], E[..., None] * dF], axis=-1)
