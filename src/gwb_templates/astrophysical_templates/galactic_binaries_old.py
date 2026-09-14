r"""
Galactic compact binary foreground template (old / 6-parameter form).

Implements Eq. 14 of Robson, Cornish & Liu 2019 (arXiv:1803.01944):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,f^{2/3}\,
        \exp\bigl(-f^{\alpha} - \beta f \sin(\kappa f)\bigr)\,
        \bigl(1 + \tanh(\gamma (f_k - f))\bigr)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class GalacticBinariesOld(AnalyticTemplate):
    r"""
    Galactic binary foreground (Robson, Cornish & Liu 2019, 6-parameter form).

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

    DEFAULT_MODEL_NAME: ClassVar[str] = "gb_old"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Galactic Binaries (old)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
        "alpha": r"$\alpha$",
        "beta": r"$\beta$",
        "kappa": r"$\kappa$",
        "gamma": r"$\gamma$",
        "fk": r"$f_k\,[\mathrm{Hz}]$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -14.0, "max": -5.0},
        "alpha": {"min": 0.01, "max": 5.0},
        "beta": {"min": 0.0, "max": 1000.0},
        "kappa": {"min": 0.0, "max": 1000.0},
        "gamma": {"min": 0.0, "max": 5000.0},
        "fk": {"min": 1e-5, "max": 1e-2},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Robson:2018ifk,
    author = "Robson, Travis and Cornish, Neil J. and Liu, Chang",
    title = "{The construction and use of LISA sensitivity curves}",
    eprint = "1803.01944",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1088/1361-6382/ab1101",
    journal = "Class. Quant. Grav.",
    volume = "36",
    number = "10",
    pages = "105011",
    year = "2019"
}
""",
    )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        alpha: jax.Array,
        beta: jax.Array,
        kappa: jax.Array,
        gamma: jax.Array,
        fk: jax.Array,
    ) -> jax.Array:

        pl_term = frequency ** (2.0 / 3.0)
        stochasticity_term = jnp.exp(
            -(frequency**alpha) - beta * frequency * jnp.sin(kappa * frequency)
        )
        tanh_term = 1.0 + jnp.tanh(gamma * (fk - frequency))
        return 10.0**log_amplitude * pl_term * stochasticity_term * tanh_term

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, alpha, beta, kappa, gamma, fk]."""
        alpha, beta, kappa, gamma, fk = theta[1], theta[2], theta[3], theta[4], theta[5]
        t = jnp.tanh(gamma * (fk - frequency))
        model = GalacticBinariesOld.omega_gw_h2(self, frequency, *theta)
        ln10 = jnp.log(10.0)
        d_logA = model * ln10
        d_alpha = model * (-(frequency**alpha) * jnp.log(frequency))
        d_beta = model * (-frequency * jnp.sin(kappa * frequency))
        d_kappa = model * (-beta * frequency**2 * jnp.cos(kappa * frequency))
        d_gamma = model * (1.0 - t) * (fk - frequency)
        d_fk = model * (1.0 - t) * gamma
        return jnp.stack([d_logA, d_alpha, d_beta, d_kappa, d_gamma, d_fk], axis=-1)


class GalacticBinariesOldA(GalacticBinariesOld):
    r"""
    Old galactic binary foreground — amplitude-only variant.

    Free parameter: ``log_amplitude`` only. Inherits :attr:`bibtex_entries` and the
    underlying spectral shape from :class:`GalacticBinariesOld`, fixing ``alpha``,
    ``beta``, ``kappa``, ``gamma``, ``fk`` at the fiducial values from arXiv:1803.01944
    (Table 1) (or values passed at construction).
    """

    DEFAULT_ALPHA: ClassVar[jax.Array] = jnp.array(0.138)
    DEFAULT_BETA: ClassVar[jax.Array] = jnp.array(221.0)
    DEFAULT_KAPPA: ClassVar[jax.Array] = jnp.array(521.0)
    DEFAULT_GAMMA: ClassVar[jax.Array] = jnp.array(1680.0)
    DEFAULT_FK: ClassVar[jax.Array] = jnp.array(0.00113)

    DEFAULT_MODEL_NAME: ClassVar[str] = "gb_old_a"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Galactic Binaries old (amplitude only)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -14.0, "max": -5.0},
    }

    def __init__(
        self,
        alpha: jax.Array = DEFAULT_ALPHA,
        beta: jax.Array = DEFAULT_BETA,
        kappa: jax.Array = DEFAULT_KAPPA,
        gamma: jax.Array = DEFAULT_GAMMA,
        fk: jax.Array = DEFAULT_FK,
        **kwargs: Any,
    ) -> None:
        self.alpha: jax.Array = alpha
        self.beta: jax.Array = beta
        self.kappa: jax.Array = kappa
        self.gamma: jax.Array = gamma
        self.fk: jax.Array = fk
        super().__init__(**kwargs)

    def omega_gw_h2(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
    ) -> jax.Array:
        return super().omega_gw_h2(
            frequency,
            log_amplitude,
            self.alpha,
            self.beta,
            self.kappa,
            self.gamma,
            self.fk,
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        full_theta = jnp.array(
            [theta[0], self.alpha, self.beta, self.kappa, self.gamma, self.fk]
        )
        return super()._grad_theta_omega_gw_h2_analytical(frequency, full_theta)[:, :1]
