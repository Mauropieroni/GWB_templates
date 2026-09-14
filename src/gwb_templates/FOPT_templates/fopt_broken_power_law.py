r"""
FOPT broken power-law template.

Phenomenological 4-parameter broken power law used as a generic first-order phase-
transition (FOPT) GW background template:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        x^{n_{\mathrm{IR}}}\,
        \left(\frac{1 + x}{2}\right)^{n_{\mathrm{UV}} - n_{\mathrm{IR}}}

with :math:`x = f / f_*`. Equivalent to a smooth broken power law with fixed transition
width (:math:`\delta = 1`).

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline and physics
interpretation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class FoptBrokenPowerLaw(AnalyticTemplate):
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

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_f_star: jax.Array,
        n_IR: jax.Array,
        n_UV: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the FOPT broken-power-law spectrum at ``frequency``.
        """
        x = frequency / 10.0**log_f_star
        return 10.0**log_amplitude * x**n_IR * (0.5 * (1.0 + x)) ** (n_UV - n_IR)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, log_f_star, n_IR, n_UV]."""
        log_amplitude, log_f_star, n_IR, n_UV = theta[0], theta[1], theta[2], theta[3]
        x = frequency / 10.0**log_f_star
        H = 0.5 * (1.0 + x)
        p = n_UV - n_IR
        model = FoptBrokenPowerLaw.omega_gw_h2(
            self, frequency, log_amplitude, log_f_star, n_IR, n_UV
        )
        ln10 = jnp.log(10.0)
        d_logA = model * ln10
        d_logfstar = -model * ln10 * (n_IR + 0.5 * p * x / H)
        d_nIR = model * (jnp.log(x) - jnp.log(H))
        d_nUV = model * jnp.log(H)
        return jnp.stack([d_logA, d_logfstar, d_nIR, d_nUV], axis=-1)
