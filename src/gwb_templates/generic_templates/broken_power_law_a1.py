r"""
Broken power-law with direct smoothness parameter ``a_1`` (5 parameters).

Used as the base spectral template for bubble-collision GW spectra. The
smoothness ``a_1`` is a direct (non-log) parameter for transparent
physical interpretation:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha}\,
        x^{n_1}\,
        \left(\tfrac{1}{2} + \tfrac{1}{2} x^{a_1}\right)^{(n_2 - n_1)/a_1}

with :math:`x = f / f_b`.

Reference: arXiv:2403.03723.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class BrokenPowerLawA1(AnalyticTemplate):
    r"""
    Broken power law with a direct smoothness parameter ``a_1`` (5 parameters).

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude at the break frequency.
    log_f_b
        :math:`\log_{10}` of the break frequency in Hz.
    n_1
        Low-frequency spectral index.
    n_2
        High-frequency spectral index.
    a_1
        Transition smoothness.
    """

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
            "log_f_b": r"$\log_{10}(f_b/\mathrm{Hz})$",
            "n_1": r"$n_1$",
            "n_2": r"$n_2$",
            "a_1": r"$a_1$",
        }
        default_priors = {
            "log_amplitude": {
                "prior_type": "uniform",
                "minimum": -20.0,
                "maximum": -1.0,
            },
            "log_f_b": {
                "prior_type": "uniform",
                "minimum": -10.0,
                "maximum": 0.0,
            },
            "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Broken Power Law (a1)"
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
        log_f_b: ArrayLike,
        n_1: ArrayLike,
        n_2: ArrayLike,
        a_1: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the ``a_1``-parametrised broken power law at ``frequency``.
        """
        x = frequency / 10.0**log_f_b
        return (
            10.0**log_amplitude
            * x**n_1
            * (0.5 + 0.5 * x**a_1) ** ((n_2 - n_1) / a_1)
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the ``a_1``-parametrised broken power law.

        Closed-form derivatives w.r.t. ``(log_amplitude, log_f_b, n_1,
        n_2, a_1)``; see module docstring for the spectral shape.
        """
        log_amplitude, log_f_b, n_1, n_2, a_1 = theta
        x = frequency / 10.0**log_f_b
        xa = x**a_1
        ln10 = jnp.log(10.0)
        log_half_1_xa = jnp.log(0.5 * (1.0 + xa))
        model = self.omega_gw_h2(frequency, log_amplitude, log_f_b, n_1, n_2, a_1)

        d_logA = model * ln10
        d_logfb = model * ln10 * (-n_1 - n_2 * xa) / (1.0 + xa)
        d_n1 = model * (jnp.log(x) - log_half_1_xa / a_1)
        d_n2 = model * log_half_1_xa / a_1
        d_a1 = (
            model
            * (n_2 - n_1)
            / a_1**2
            / (1.0 + xa)
            * (a_1 * xa * jnp.log(x) - (1.0 + xa) * log_half_1_xa)
        )

        return jnp.stack([d_logA, d_logfb, d_n1, d_n2, d_a1], axis=-1)
