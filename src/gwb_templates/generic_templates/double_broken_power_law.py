r"""
Double broken power-law (DBPL) template for first-order phase transitions.

Generic 8-parameter spectral shape used as a phenomenological model for
GW sources from first-order phase transitions (sound waves, turbulence,
bubble collisions). The amplitude is defined at the second break
frequency :math:`f_2` for stable normalisation:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = N \cdot \Omega_* h^2\,
        (f/f_1)^{n_1}\,
        (1 + (f/f_1)^{a_1})^{(n_2 - n_1)/a_1}\,
        (1 + (f/f_2)^{a_2})^{(n_3 - n_2)/a_2}

where :math:`N` normalises the spectrum so that
:math:`\Omega_{\mathrm{GW}} h^2(f_2) = \Omega_* h^2`.

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


class DoubleBrokenPowerLaw(AnalyticTemplate):
    r"""
    Double broken power law (8 parameters), normalised at :math:`f_2`.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude at :math:`f_2`.
    log_f_1
        :math:`\log_{10}` of the first break frequency in Hz.
    log_f_2
        :math:`\log_{10}` of the second break frequency in Hz.
    n_1, n_2, n_3
        Spectral indices in the three regimes.
    a_1, a_2
        Smoothness parameters at the two breaks.
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
            "log_f_1": r"$\log_{10}(f_1/\mathrm{Hz})$",
            "log_f_2": r"$\log_{10}(f_2/\mathrm{Hz})$",
            "n_1": r"$n_1$",
            "n_2": r"$n_2$",
            "n_3": r"$n_3$",
            "a_1": r"$a_1$",
            "a_2": r"$a_2$",
        }
        default_priors = {
            "log_amplitude": {
                "prior_type": "uniform",
                "minimum": -20.0,
                "maximum": -1.0,
            },
            "log_f_1": {
                "prior_type": "uniform",
                "minimum": -10.0,
                "maximum": 0.0,
            },
            "log_f_2": {
                "prior_type": "uniform",
                "minimum": -10.0,
                "maximum": 0.0,
            },
            "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_3": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
            "a_2": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Double Broken Power Law"
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
        log_f_1: ArrayLike,
        log_f_2: ArrayLike,
        n_1: ArrayLike,
        n_2: ArrayLike,
        n_3: ArrayLike,
        a_1: ArrayLike,
        a_2: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the double-broken-power-law spectrum at ``frequency``.
        """
        amplitude = 10.0**log_amplitude
        ratio = 10.0 ** (log_f_1 - log_f_2)  # f_1 / f_2
        x_1 = frequency / 10.0**log_f_1
        x_2 = frequency / 10.0**log_f_2

        # normalisation so that Omega(f_2) = amplitude
        norm = (
            ratio**n_1
            * (1.0 + ratio ** (-a_1)) ** ((n_1 - n_2) / a_1)
            * 2.0 ** ((n_2 - n_3) / a_2)
        )

        return (
            norm
            * amplitude
            * x_1**n_1
            * (1.0 + x_1**a_1) ** ((n_2 - n_1) / a_1)
            * (1.0 + x_2**a_2) ** ((n_3 - n_2) / a_2)
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the double broken power law.

        Closed-form derivatives w.r.t. the 8 parameters
        ``(log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2)``;
        see the source for the full expressions.
        """
        log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2 = theta
        x_1 = frequency / 10.0**log_f_1
        x_2 = frequency / 10.0**log_f_2
        r_12 = 10.0 ** (log_f_1 - log_f_2)
        ln10 = jnp.log(10.0)
        model = self.omega_gw_h2(
            frequency, log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2
        )

        x1a1 = x_1**a_1
        x2a1 = x_2**a_1
        x2a2 = x_2**a_2
        r12a1 = r_12**a_1

        d_logA = model * ln10

        d_logf1 = (
            model * ln10 * (n_1 - n_2) * (x2a1 - 1.0) / ((1.0 + r12a1) * (1.0 + x1a1))
        )

        num_f2 = (
            -n_1 * r12a1
            - (n_3 + (n_1 + n_3) * r12a1) * x2a2
            + n_2 * (r12a1 * x2a2 - 1.0)
        )
        d_logf2 = model * ln10 * num_f2 / ((1.0 + r12a1) * (1.0 + x2a2))

        d_n1 = model * jnp.log((x1a1 + x2a1) / (1.0 + x1a1)) / a_1

        d_n2 = model * (
            -jnp.log(0.5 * (1.0 + x2a2)) / a_2
            + jnp.log((r12a1 + x2a1) / (1.0 + r12a1)) / a_1
        )

        d_n3 = model * jnp.log(0.5 * (1.0 + x2a2)) / a_2

        term_r = 1.0 + r12a1
        term_1 = 1.0 + x1a1
        prefactor_a1 = (n_2 - n_1) / a_1**2 / term_r / term_1
        add1 = a_1 * jnp.log(r_12)
        add2 = term_r * term_1 * jnp.log(term_r / r12a1)
        add3 = -term_r * jnp.log(term_1)
        add4 = x2a1 * (a_1 * jnp.log(x_1) - jnp.log(term_1))
        add5 = x1a1 * (a_1 * jnp.log(x_2) - jnp.log(term_1))
        d_a1 = model * prefactor_a1 * (add1 + add2 + add3 + add4 + add5)

        prefactor_a2 = (n_3 - n_2) / a_2**2 / (1.0 + x2a2)
        d_a2 = (
            model
            * prefactor_a2
            * (
                a_2 * x2a2 * jnp.log(x_2)
                - (1.0 + x2a2) * jnp.log(0.5 * (1.0 + x2a2))
            )
        )

        return jnp.stack(
            [d_logA, d_logf1, d_logf2, d_n1, d_n2, d_n3, d_a1, d_a2], axis=-1
        )
