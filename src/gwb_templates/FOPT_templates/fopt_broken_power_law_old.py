r"""
FOPT broken power law (old / 2-parameter) template.

Implements the broken power law from Eq. 14 of arXiv:1512.06239:

.. math::

    \Omega(f) = A\,x^{3}
        \left(\frac{7}{4 + 3 x^{2}}\right)^{7/2}, \qquad x = f / f_*.

A two-parameter special case of the smooth BPL with fixed
:math:`n_1 = 3`, :math:`n_2 = -4`, :math:`\delta = 1/2`, written so that
``log_amplitude`` and ``log_f_star`` are the log10 of the peak amplitude
and peak frequency respectively.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


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

    # TODO: cite (arXiv:1512.06239)
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
            "log_f_star": r"$\log_{10}(f_*/\mathrm{Hz})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_f_star": {"min": -5.0, "max": 0.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "FOPT Broken Power Law (old)"
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
        log_f_star: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the 2-parameter FOPT broken power-law spectrum at
        ``frequency``.
        """
        x = frequency / 10.0**log_f_star
        return (
            10.0**log_amplitude
            * x**3.0
            * (7.0 / (4.0 + 3.0 * x**2.0)) ** 3.5
        )
