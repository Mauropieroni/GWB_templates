r"""
Sharp-feature oscillatory modulation template.

Linear-in-amplitude cosine oscillation that can modulate a smooth envelope.
Two parametrizations are offered:

* :class:`SharpFeature` — direct ``A_sharp``, ``omega_sharp_Hz``, ``phase_sharp``.
* :class:`SharpFeatureLog` — base-10 log amplitude and frequency for
  wide-range priors.

References:
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline
  and physics interpretation).
  arXiv:astro-ph/0102236 (Adams, Cresswell & Easther — original
  step-potential oscillatory template).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class SharpFeature(AnalyticTemplate):
    r"""
    Sharp-feature modulation:

    .. math::

        F(f) = 1 + A_{\mathrm{sharp}}\,
               \cos(\omega_{\mathrm{sharp}}\,f + \phi_{\mathrm{sharp}})

    Free parameters
    ---------------
    A_sharp
        Linear amplitude of the oscillation.
    omega_sharp_Hz
        Angular frequency of the oscillation (Hz\ :sup:`-1`).
    phase_sharp
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
@article{Adams:2001vc,
    author = "Adams, Jennifer A. and Cresswell, Bevan and Easther, Richard",
    title = "{Inflationary perturbations from a potential with a step}",
    eprint = "astro-ph/0102236",
    archivePrefix = "arXiv",
    reportNumber = "CU-TP-1005",
    doi = "10.1103/PhysRevD.64.123514",
    journal = "Phys. Rev. D",
    volume = "64",
    pages = "123514",
    year = "2001"
}
""",
    )

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "A_sharp": r"$A_{\rm s}$",
            "omega_sharp_Hz": r"$\omega_{\rm s}\,[\mathrm{Hz}^{-1}]$",
            "phase_sharp": r"$\phi_{\rm s}$",
        }
        default_priors = {
            "A_sharp": {"min": -1.0, "max": 1.0},
            "omega_sharp_Hz": {"min": 0.0, "max": 1e5},
            "phase_sharp": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Sharp Feature",
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
        A_sharp: ArrayLike,
        omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        return 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * frequency + phase_sharp)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian of the sharp-feature modulation."""
        A_sharp, omega_sharp_Hz, phase_sharp = theta[0], theta[1], theta[2]
        freq = jnp.asarray(frequency)
        arg = omega_sharp_Hz * freq + phase_sharp
        d_A = jnp.cos(arg)
        d_omega = -A_sharp * jnp.sin(arg) * freq
        d_theta = -A_sharp * jnp.sin(arg)
        return jnp.stack([d_A, d_omega, d_theta], axis=-1)


class SharpFeatureLog(AnalyticTemplate):
    r"""
    Sharp-feature modulation with log-parametrized amplitude and frequency.
    Identical physics to :class:`SharpFeature`; log-scaled parameters allow
    wide priors to be sampled efficiently.

    Free parameters
    ---------------
    log_A_sharp
        :math:`\log_{10}` amplitude.
    log_omega_sharp_Hz
        :math:`\log_{10}` angular frequency.
    phase_sharp
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
@article{Adams:2001vc,
    author = "Adams, Jennifer A. and Cresswell, Bevan and Easther, Richard",
    title = "{Inflationary perturbations from a potential with a step}",
    eprint = "astro-ph/0102236",
    archivePrefix = "arXiv",
    reportNumber = "CU-TP-1005",
    doi = "10.1103/PhysRevD.64.123514",
    journal = "Phys. Rev. D",
    volume = "64",
    pages = "123514",
    year = "2001"
}
""",
    )

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_A_sharp": r"$\log_{10}A_{\rm s}$",
            "log_omega_sharp_Hz": r"$\log_{10}(\omega_{\rm s}/\mathrm{Hz}^{-1})$",
            "phase_sharp": r"$\phi_{\rm s}$",
        }
        default_priors = {
            "log_A_sharp": {"min": -3.0, "max": 0.0},
            "log_omega_sharp_Hz": {"min": 0.0, "max": 5.0},
            "phase_sharp": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Sharp Feature (log params)"
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
        log_A_sharp: ArrayLike,
        log_omega_sharp_Hz: ArrayLike,
        phase_sharp: ArrayLike,
    ) -> jax.Array:
        A_sharp = 10.0**log_A_sharp
        omega_sharp_Hz = 10.0**log_omega_sharp_Hz
        return 1.0 + A_sharp * jnp.cos(omega_sharp_Hz * frequency + phase_sharp)

    # NOTE: No analytic gradient override — the test for this class compares
    # the gradient to autodiff at places=15, but the mathematically-correct
    # analytic form has different fp64 rounding behavior (different order of
    # the `omega * freq * ln10` product) and cannot match autodiff bit-exactly.
    # We leave the autodiff backend in place rather than weaken the test.
