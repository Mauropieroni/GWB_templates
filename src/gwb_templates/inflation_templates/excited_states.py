r"""
Excited-states inflationary GWB template.

Models the GWB produced by excited states during inflation, with a power-law rise and a
sharp Heaviside cutoff at :math:`x_{\mathrm{cut}} = 2\,\gamma_{\rm ES}` where
:math:`x = 0.5 f\, 10^{\log\omega_{\rm ES}}`:

.. math::

    \Omega_{\rm GW} h^2(f) = \frac{10^A}{0.052}\, x^{-3}
        \bigl(1 - (x/(2\gamma_{\rm ES}))^2\bigr)^2\,
        \bigl(\sin x - 4\sin^2(x/2)/x\bigr)^2\,
        \Theta(2\gamma_{\rm ES} - x).

References:
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline and physics
  interpretation). arXiv:2111.14664 (Fumagalli, Palma, Renaux-Petel, Sypsas, Witkowski &
  Zenteno — original excited-states derivation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class ExcitedStates(AnalyticTemplate):
    r"""
    Excited-states inflationary GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` overall amplitude.
    log_gamma_ES
        :math:`\log_{10}\gamma_{\rm ES}`. Sets the location of the Heaviside cutoff.
    log_omega_ES
        :math:`\log_{10}\omega_{\rm ES}`. Sets the oscillation frequency.
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
@article{Fumagalli:2021mpc,
    author = "Fumagalli, Jacopo and Palma, Gonzalo A. and Renaux-Petel, S\'ebastien and
        Sypsas, Spyros and Witkowski, Lukas T. and Zenteno, Cristobal",
    title = "{Primordial gravitational waves from excited states}",
    eprint = "2111.14664",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "IFT-UAM/CSIC-21-140",
    doi = "10.1007/JHEP03(2022)196",
    journal = "JHEP",
    volume = "03",
    pages = "196",
    year = "2022"
}
""",
    )

    DEFAULT_MODEL_NAME: ClassVar[str] = "excited_states"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Excited States"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_gamma_ES": r"$\log_{10}(\gamma_{\rm ES})$",
        "log_omega_ES": r"$\log_{10}(\omega_{\rm ES}\,\mathrm{Hz})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_gamma_ES": {"min": -3.0, "max": 3.0},
        "log_omega_ES": {"min": 0.0, "max": 12.0},
    }

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_gamma_ES: jax.Array,
        log_omega_ES: jax.Array,
    ) -> jax.Array:
        amplitude = 10.0**log_amplitude
        gamma_ES = 10.0**log_gamma_ES

        x = 0.5 * frequency * 10.0**log_omega_ES
        x_cut = 2.0 * gamma_ES

        factor_1 = amplitude / 0.052 * x ** (-3)
        factor_2 = (1.0 - (x / (2.0 * gamma_ES)) ** 2) ** 2
        factor_3 = (jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x) ** 2

        return factor_1 * factor_2 * factor_3 * jnp.heaviside(x_cut - x, 1.0)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian of ``omega_gw_h2`` w.r.t. theta.

        Heaviside boundary delta is ignored (consistent with ``jnp.heaviside``).
        """
        log_amplitude, log_gamma_ES, log_omega_ES = theta[0], theta[1], theta[2]

        amplitude = 10.0**log_amplitude
        gamma_ES = 10.0**log_gamma_ES

        x = 0.5 * frequency * 10.0**log_omega_ES
        x_cut = 2.0 * gamma_ES
        u = x / x_cut

        H = jnp.heaviside(x_cut - x, 1.0)
        factor_1 = amplitude / 0.052 * x ** (-3)
        factor_2 = (1.0 - u**2) ** 2
        g = jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x
        factor_3 = g**2
        model = factor_1 * factor_2 * factor_3 * H

        ln10 = jnp.log(10.0)

        d_logA = model * ln10
        d_lgam = factor_1 * factor_3 * H * 4.0 * ln10 * u**2 * (1.0 - u**2)
        dg_dx = jnp.cos(x) - 2.0 * jnp.sin(x) / x + 2.0 * (1.0 - jnp.cos(x)) / x**2
        d_lomega = H * (
            factor_1 * (-3.0 * ln10) * factor_2 * factor_3
            + factor_1 * (-4.0 * ln10 * u**2 * (1.0 - u**2)) * factor_3
            + factor_1 * factor_2 * (2.0 * g * dg_dx * x * ln10)
        )

        return jnp.stack([d_logA, d_lgam, d_lomega], axis=-1)
