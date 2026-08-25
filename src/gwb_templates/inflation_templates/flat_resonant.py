r"""
Flat (amplitude-only) envelope modulated by a resonant-feature oscillation.

Two parametrizations:

* :class:`FlatResonant` — linear ``A_resonant``, ``omega_resonant``.
* :class:`FlatResonantLog` — log-scaled amplitude and frequency.

.. math::

    \Omega_{\rm GW} h^2(f) = 10^{A}\,
        \mathrm{ResonantFeature}(f; A_r, \omega_r, \phi_r)

Like :mod:`gwb_templates.inflation_templates.resonant_feature`, these
templates rely on the precomputed table
``data/Resonant_coefficients.npz`` and therefore inherit from
:class:`~gwb_templates.template.NumericalTemplate`.

References:
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline
  and physics interpretation).
  arXiv:0907.2916 (Flauger, McAllister, Pajer, Westphal & Xu — original
  resonant oscillatory power-spectrum template from axion monodromy).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.typing as jtp
from interpax import Interpolator1D

from gwb_templates.inflation_templates.resonant_feature import (
    _build_resonant_interpolators,
    _resonant_feature_impl,
)
from gwb_templates.template import NumericalTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


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
    differentiation_backend: ClassVar[str] = "autodiff"  # type: ignore[assignment]

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
            "A_resonant": r"$A_{\rm r}$",
            "omega_resonant": r"$\omega_{\rm r}$",
            "phase_resonant": r"$\phi_{\rm r}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "A_resonant": {"min": 0.0, "max": 1.0},
            "omega_resonant": {"min": 1e-3, "max": 100.0},
            "phase_resonant": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Flat + Resonant Feature"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_amplitude: ArrayLike,
        A_resonant: ArrayLike,
        omega_resonant: ArrayLike,
        phase_resonant: ArrayLike,
    ) -> jax.Array:
        envelope = 10.0**log_amplitude
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        return envelope * modulation

    # NOTE: see ResonantFeature — autodiff is preserved as the default
    # backend because the precomputed coefficient-slope tables differ from
    # interpax's autodiff slope of the value tables.


class FlatResonantLog(NumericalTemplate):
    r"""
    Flat amplitude multiplied by a log-parametrized resonant-feature
    modulation.

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
    differentiation_backend: ClassVar[str] = "autodiff"  # type: ignore[assignment]

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
            "log_A_resonant": r"$\log_{10}A_{\rm r}$",
            "log_omega_resonant": r"$\log_{10}\omega_{\rm r}$",
            "phase_resonant": r"$\phi_{\rm r}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_A_resonant": {"min": -3.0, "max": 0.0},
            "log_omega_resonant": {"min": -3.0, "max": 2.0},
            "phase_resonant": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Flat + Resonant Feature (log params)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_amplitude: ArrayLike,
        log_A_resonant: ArrayLike,
        log_omega_resonant: ArrayLike,
        phase_resonant: ArrayLike,
    ) -> jax.Array:
        envelope = 10.0**log_amplitude
        A_resonant = 10.0**log_A_resonant
        omega_resonant = 10.0**log_omega_resonant
        modulation = _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
        return envelope * modulation

    # NOTE: see ResonantFeature — autodiff preserved.
