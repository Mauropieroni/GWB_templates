r"""
FOPT broken power law (old / 2-parameter) template.

Implements the broken power law from Eq. 14 of arXiv:1512.06239:

.. math::

    \Omega_\mathrm{GW} h^2(f) = \Omega_* h^2\, x^{3}
        \left(\frac{7}{4 + 3 x^{2}}\right)^{7/2}, \qquad x = f / f_*.

A two-parameter special case of the smooth BPL with fixed :math:`n_1 = 3`,
:math:`n_2 = -4`, :math:`\delta = 1/2`, written so that ``log_amplitude`` and
``log_f_star`` are the log10 of the peak amplitude and peak frequency respectively.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class FoptBrokenPowerLawOld(AnalyticTemplate):
    r"""
    Two-parameter FOPT broken power law from arXiv:1512.06239.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` of the peak amplitude :math:`h^2 \Omega_*`.
    log_f_star
        :math:`\log_{10}` of the peak frequency in Hz.
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "fopt_broken_power_law_old"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "FOPT Broken Power Law (old)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_f_star": r"$\log_{10}(f_*/\mathrm{Hz})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_f_star": {"min": -5.0, "max": 0.0},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2015zlo,
    author = "Caprini, Chiara and others",
    title = "{Science with the space-based interferometer eLISA. II: Gravitational waves
        from cosmological phase transitions}",
    eprint = "1512.06239",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESY-15-246",
    doi = "10.1088/1475-7516/2016/04/001",
    journal = "JCAP",
    volume = "04",
    pages = "001",
    year = "2016"
}
""",
    )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_f_star: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the 2-parameter FOPT broken power-law spectrum at
        ``frequency``.
        """
        x = frequency / 10.0**log_f_star
        return 10.0**log_amplitude * x**3.0 * (7.0 / (4.0 + 3.0 * x**2.0)) ** 3.5

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the 2-parameter FOPT broken power-law spectrum
        with respect to ``(log_amplitude, log_f_star)``.

        Returns an array of shape ``frequency.shape + (2,)``.
        """
        log_amplitude, log_f_star = theta[0], theta[1]
        x = frequency / 10.0**log_f_star
        model = 10.0**log_amplitude * x**3.0 * (7.0 / (4.0 + 3.0 * x**2.0)) ** 3.5
        ln10 = jnp.log(10.0)

        d_logA = model * ln10
        # d/d(log_f_star): x ∝ 10^(-log_f_star), so d ln(x)/d log_f_star = -ln10.
        # d ln(model)/d log_f_star = 3 * (-ln10)
        #     + 3.5 * (-1) * (6 x^2 / (4 + 3 x^2)) * (-ln10)
        d_logf = model * ln10 * (-3.0 + 21.0 * x**2.0 / (4.0 + 3.0 * x**2.0))

        return jnp.stack([d_logA, d_logf], axis=-1)
