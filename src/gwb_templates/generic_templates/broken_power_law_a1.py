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

    #: TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

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
