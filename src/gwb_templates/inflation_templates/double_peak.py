r"""
Double-peak inflationary GWB template.

Combines a power-law rise with a Heaviside cutoff (first peak) and a
log-normal with an erfc modulation (second peak). Useful for modelling
PBH-related spectra.

Reference: arXiv:2407.04356 (GW from inflation in LISA: reconstruction
pipeline and physics interpretation).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class DoublePeak(AnalyticTemplate):
    r"""
    Double-peak inflationary spectrum.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` of the overall amplitude.
    log_pivot
        :math:`\log_{10}` of the pivot frequency :math:`f_*`.
    beta
        Relative weight of the first (power-law) peak.
    k1
        Position of the first peak relative to the pivot.
    k2
        Position of the second (log-normal) peak relative to the pivot.
    rho
        Width of the log-normal second peak.
    gamma
        Asymmetry parameter of the second peak (erfc modulation).

    Configuration
    -------------
    c1
        Fixed shape constant for the first peak. Defaults to
        :math:`\sqrt{2/3}`.
    tilt_p
        Fixed UV spectral index of the first peak. Defaults to 2.5.
    """

    #: Default shape constant for the first peak: sqrt(2/3).
    DEFAULT_C1: ClassVar[float] = math.sqrt(2.0 / 3.0)
    DEFAULT_TILT_P: ClassVar[float] = 2.5

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        c1: float = DEFAULT_C1,
        tilt_p: float = DEFAULT_TILT_P,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.c1: float = float(c1)
        self.tilt_p: float = float(tilt_p)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
            "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
            "beta": r"$\beta$",
            "k1": r"$\kappa_1$",
            "k2": r"$\kappa_2$",
            "rho": r"$\rho$",
            "gamma": r"$\gamma$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_pivot": {"min": -5.0, "max": 0.0},
            "beta": {"min": 0.0, "max": 10.0},
            "k1": {"min": 0.1, "max": 10.0},
            "k2": {"min": 0.1, "max": 10.0},
            "rho": {"min": 0.01, "max": 5.0},
            "gamma": {"min": -5.0, "max": 5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Double Peak",
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
        beta: ArrayLike,
        k1: ArrayLike,
        k2: ArrayLike,
        rho: ArrayLike,
        gamma: ArrayLike,
    ) -> jax.Array:
        amplitude = 10.0**log_amplitude
        pivot = 10.0**log_pivot

        x = frequency / pivot
        x1 = x / k1
        c1_k1 = self.c1 / k1
        log10x2 = jnp.log10(x / k2)

        bump_factor = jnp.abs((c1_k1 - x1) / (c1_k1 - 1.0)) ** (
            self.tilt_p * (c1_k1 - 1.0)
        )
        first_term = (
            beta * x1**self.tilt_p * bump_factor * jnp.heaviside(c1_k1 - x1, 1.0)
        )

        log_normal = jnp.exp(-0.5 * (log10x2 / rho) ** 2)
        erfc_term = jax.scipy.special.erfc(gamma * log10x2)
        second_term = log_normal * erfc_term

        return amplitude * (first_term + second_term)
