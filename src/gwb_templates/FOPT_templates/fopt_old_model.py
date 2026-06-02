r"""
Legacy 2-parameter broken power-law template (arXiv:1512.06239).

Near-duplicate of :mod:`fopt_broken_power_law_old`. Kept for backwards
compatibility while we decide which variant to drop. The only material
difference vs. :class:`FoptBrokenPowerLawOld` is the second parameter
name (``log_pivot`` here vs. ``log_f_star`` there).

TODO: merge with :class:`FoptBrokenPowerLawOld`.
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
    Legacy two-parameter broken power law from arXiv:1512.06239.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` of the peak amplitude :math:`h^2 \Omega_*`.
    log_pivot
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
            "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_pivot": {"min": -5.0, "max": 0.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Broken Power Law (old)"
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
        log_pivot: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate the legacy broken power-law spectrum."""
        x = frequency / 10.0**log_pivot
        return 10.0**log_amplitude * x**3.0 * (7.0 / (4.0 + 3.0 * x**2.0)) ** 3.5
