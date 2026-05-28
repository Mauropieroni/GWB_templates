r"""
Galactic compact binary foreground template (old / 6-parameter form).

Implements Eq. 14 of Mangiagli et al. 2020 (arXiv:1803.01944):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,f^{2/3}\,
        \exp\bigl(-f^{\alpha} - \beta f \sin(\kappa f)\bigr)\,
        \bigl(1 + \tanh(\gamma (f_k - f))\bigr)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class GalacticBinariesOld(AnalyticTemplate):
    r"""
    Galactic binary foreground (Mangiagli et al. 2020, 6-parameter form).

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude.
    alpha
        Exponential cutoff exponent.
    beta
        Sinusoidal modulation amplitude in the exponential.
    kappa
        Sinusoidal modulation frequency in the exponential.
    gamma
        Width of the ``tanh`` cutoff.
    fk
        Knee frequency (Hz) of the ``tanh`` cutoff.
    """

    # TODO: cite
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
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "kappa": r"$\kappa$",
            "gamma": r"$\gamma$",
            "fk": r"$f_k\,[\mathrm{Hz}]$",
        }
        default_priors = {
            "log_amplitude": {"min": -14.0, "max": -5.0},
            "alpha": {"min": 0.01, "max": 5.0},
            "beta": {"min": 0.0, "max": 1000.0},
            "kappa": {"min": 0.0, "max": 1000.0},
            "gamma": {"min": 0.0, "max": 5000.0},
            "fk": {"min": 1e-5, "max": 1e-2},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Galactic Binaries (old)"
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
        alpha: ArrayLike,
        beta: ArrayLike,
        kappa: ArrayLike,
        gamma: ArrayLike,
        fk: ArrayLike,
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        return (
            10.0**log_amplitude
            * fvec ** (2.0 / 3.0)
            * jnp.exp(-(fvec**alpha) - beta * fvec * jnp.sin(kappa * fvec))
            * (1.0 + jnp.tanh(gamma * (fk - fvec)))
        )


class GalacticBinariesOldA(AnalyticTemplate):
    r"""
    Old galactic binary foreground — amplitude-only variant.

    Shape parameters are fixed at the fiducial values from
    arXiv:1803.01944 (Table 1) but can be overridden via constructor kwargs.
    """

    DEFAULT_ALPHA: ClassVar[float] = 0.138
    DEFAULT_BETA: ClassVar[float] = 221.0
    DEFAULT_KAPPA: ClassVar[float] = 521.0
    DEFAULT_GAMMA: ClassVar[float] = 1680.0
    DEFAULT_FK: ClassVar[float] = 0.00113

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        kappa: float = DEFAULT_KAPPA,
        gamma: float = DEFAULT_GAMMA,
        fk: float = DEFAULT_FK,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.kappa: float = float(kappa)
        self.gamma: float = float(gamma)
        self.fk: float = float(fk)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
        }
        default_priors = {
            "log_amplitude": {"min": -14.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Galactic Binaries old (amplitude only)"
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
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        return (
            10.0**log_amplitude
            * fvec ** (2.0 / 3.0)
            * jnp.exp(
                -(fvec**self.alpha) - self.beta * fvec * jnp.sin(self.kappa * fvec)
            )
            * (1.0 + jnp.tanh(self.gamma * (self.fk - fvec)))
        )
