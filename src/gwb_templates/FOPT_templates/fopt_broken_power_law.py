r"""
FOPT broken power-law template.

Phenomenological 4-parameter broken power law used as a generic first-order phase-
transition (FOPT) GW background template:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        x^{n_{\mathrm{IR}}}\,
        \left(\frac{1 + x}{2}\right)^{n_{\mathrm{UV}} - n_{\mathrm{IR}}}

with :math:`x = f / f_*`. Equivalent to a smooth broken power law with fixed transition
width (:math:`\delta = 1`); this is exactly :class:`BrokenPowerLawFixedSmoothness` with
``log_pivot = log_f_star``, ``tilt_1 = n_IR`` and ``tilt_2 = n_UV``, so this class
delegates to it rather than reimplementing the formula.

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline and physics
interpretation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax

from gwb_templates.generic_templates.broken_power_law_fixed_smoothness import (
    BrokenPowerLawFixedSmoothness,
)


class FoptBrokenPowerLaw(BrokenPowerLawFixedSmoothness):
    r"""
    FOPT broken power law with fixed transition smoothness (4 parameters).

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude :math:`h^2 \Omega_*` at the break.
    log_f_star
        :math:`\log_{10}` of the break frequency in Hz.
    n_IR
        Low-frequency (infrared) spectral index.
    n_UV
        High-frequency (ultraviolet) spectral index.
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "fopt_broken_power_law"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "FOPT Broken Power Law"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_f_star": r"$\log_{10}(f_*/\mathrm{Hz})$",
        "n_IR": r"$n_{\mathrm{IR}}$",
        "n_UV": r"$n_{\mathrm{UV}}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_f_star": {"min": -5.0, "max": 0.0},
        "n_IR": {"min": 0.0, "max": 10.0},
        "n_UV": {"min": -10.0, "max": 0.0},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2024hue,
    author = "Caprini, Chiara and Jinno, Ryusuke and Lewicki, Marek and Madge, Eric and
        Merchand, Marco and Nardini, Germano and Pieroni, Mauro and Roper Pol, Alberto
        and Vaskonen, Ville",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from first-order phase transitions in LISA:
        reconstruction pipeline and physics interpretation}",
    eprint = "2403.03723",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-01, CERN-TH-2024-029",
    doi = "10.1088/1475-7516/2024/10/020",
    journal = "JCAP",
    volume = "10",
    pages = "020",
    year = "2024"
}
""",
    )

    def omega_gw_h2(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_f_star: jax.Array,
        n_IR: jax.Array,
        n_UV: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the FOPT broken-power-law spectrum at ``frequency``.

        Delegates to :class:`BrokenPowerLawFixedSmoothness` under the renaming
        ``log_pivot = log_f_star``, ``tilt_1 = n_IR``, ``tilt_2 = n_UV``.
        """
        return super().omega_gw_h2(frequency, log_amplitude, log_f_star, n_IR, n_UV)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """
        Analytic Jacobian; columns [log_amplitude, log_f_star, n_IR, n_UV].

        Delegates to :class:`BrokenPowerLawFixedSmoothness`; the parameter order and
        names are identical (up to renaming), so no reordering is needed.
        """
        return super()._grad_theta_omega_gw_h2_analytical(frequency, theta)
