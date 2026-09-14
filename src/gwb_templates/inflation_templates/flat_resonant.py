r"""
Flat (amplitude-only) envelope modulated by a resonant-feature oscillation.

Two parametrizations:

* :class:`FlatResonant` — linear ``A_resonant``, ``omega_resonant``.
* :class:`FlatResonantLog` — log-scaled amplitude and frequency.

.. math::

    \Omega_{\rm GW} h^2(f) = 10^{A}\,
        \mathrm{ResonantFeature}(f; A_r, \omega_r, \phi_r)

Like :mod:`gwb_templates.inflation_templates.resonant_feature`, these templates rely on
the precomputed table ``data/Resonant_coefficients.npz`` and therefore inherit from
:class:`~gwb_templates.template.NumericalTemplate`.

References:
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline and physics
  interpretation). arXiv:0907.2916 (Flauger, McAllister, Pajer, Westphal & Xu — original
  resonant oscillatory power-spectrum template from axion monodromy).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from interpax import Interpolator1D

from gwb_templates.inflation_templates.resonant_feature import (
    _build_resonant_interpolators,
    _resonant_feature_grad_lin,
    _resonant_feature_impl,
)
from gwb_templates.template import DifferentiationBackend, NumericalTemplate


class FlatResonant(NumericalTemplate):
    r"""
    Flat amplitude multiplied by a linear resonant-feature modulation.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` overall amplitude.
    A_resonant
        Linear amplitude of the resonant modulation.
    omega_resonant
        Resonant oscillation frequency.
    phase_resonant
        Phase offset (radians).
    """

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
        r"""
@article{Flauger:2009ab,
    author = "Flauger, Raphael and McAllister, Liam and Pajer, Enrico and Westphal,
        Alexander and Xu, Gang",
    title = "{Oscillations in the CMB from Axion Monodromy Inflation}",
    eprint = "0907.2916",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "SLAC-PUB-14821",
    doi = "10.1088/1475-7516/2010/06/009",
    journal = "JCAP",
    volume = "06",
    pages = "009",
    year = "2010"
}
""",
    )

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    DEFAULT_MODEL_NAME: ClassVar[str] = "flat_resonant"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Flat + Resonant Feature"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "A_resonant": r"$A_{\rm r}$",
        "omega_resonant": r"$\omega_{\rm r}$",
        "phase_resonant": r"$\phi_{\rm r}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "A_resonant": {"min": 0.0, "max": 1.0},
        "omega_resonant": {"min": 1e-3, "max": 100.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    }

    def setup(self) -> None:
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        A_resonant: jax.Array,
        omega_resonant: jax.Array,
        phase_resonant: jax.Array,
    ) -> jax.Array:
        envelope = 10.0**log_amplitude
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian via product rule on flat envelope x resonant modulation."""
        log_amplitude = theta[0]
        A_resonant, omega_resonant, phase_resonant = theta[1], theta[2], theta[3]
        envelope = 10.0**log_amplitude
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        d_log_amplitude = envelope * modulation * jnp.log(10.0)
        grad_modulation = _resonant_feature_grad_lin(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        return jnp.concatenate(
            [d_log_amplitude[..., None], envelope * grad_modulation], axis=-1
        )


class FlatResonantLog(NumericalTemplate):
    r"""
    Flat amplitude multiplied by a log-parametrized resonant-feature modulation.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` overall amplitude.
    log_A_resonant
        :math:`\log_{10}` resonant amplitude.
    log_omega_resonant
        :math:`\log_{10}` resonant oscillation frequency.
    phase_resonant
        Phase offset (radians).
    """

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
        r"""
@article{Flauger:2009ab,
    author = "Flauger, Raphael and McAllister, Liam and Pajer, Enrico and Westphal,
        Alexander and Xu, Gang",
    title = "{Oscillations in the CMB from Axion Monodromy Inflation}",
    eprint = "0907.2916",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "SLAC-PUB-14821",
    doi = "10.1088/1475-7516/2010/06/009",
    journal = "JCAP",
    volume = "06",
    pages = "009",
    year = "2010"
}
""",
    )

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    DEFAULT_MODEL_NAME: ClassVar[str] = "flat_resonant_log"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Flat + Resonant Feature (log params)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_A_resonant": r"$\log_{10}A_{\rm r}$",
        "log_omega_resonant": r"$\log_{10}\omega_{\rm r}$",
        "phase_resonant": r"$\phi_{\rm r}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_A_resonant": {"min": -3.0, "max": 0.0},
        "log_omega_resonant": {"min": -3.0, "max": 2.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    }

    def setup(self) -> None:
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_A_resonant: jax.Array,
        log_omega_resonant: jax.Array,
        phase_resonant: jax.Array,
    ) -> jax.Array:
        envelope = 10.0**log_amplitude
        A_resonant = 10.0**log_A_resonant
        omega_resonant = 10.0**log_omega_resonant
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        return envelope * modulation

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: envelope chain rule x log resonant-feature chain rule."""
        log_amplitude = theta[0]
        log_A_resonant = theta[1]
        log_omega_resonant = theta[2]
        phase_resonant = theta[3]
        A_resonant = 10.0**log_A_resonant
        omega_resonant = 10.0**log_omega_resonant
        envelope = 10.0**log_amplitude
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        ln10 = jnp.log(10.0)
        d_log_amplitude = envelope * modulation * ln10

        grad_lin = _resonant_feature_grad_lin(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        d_logA = grad_lin[..., 0] * A_resonant * ln10
        d_logomega = grad_lin[..., 1] * omega_resonant * ln10
        d_phi = grad_lin[..., 2]
        grad_modulation_log = jnp.stack([d_logA, d_logomega, d_phi], axis=-1)

        return jnp.concatenate(
            [d_log_amplitude[..., None], envelope * grad_modulation_log], axis=-1
        )
