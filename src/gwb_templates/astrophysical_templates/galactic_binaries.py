r"""
Galactic compact binary foreground template.

Implements Eq. 6 of Karnesis et al. 2021 (arXiv:2103.14598):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,f^{2/3}\,
        \exp\bigl(-(f/f_{r1})^{\alpha}\bigr)\,
        \tfrac{1}{2}\bigl(1 + \tanh(-(f - f_{rk})/f_{r2})\bigr)

A helper :func:`galactic_pars` computes the observation-dependent fiducial shape
parameters from the Karnesis fit table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


def galactic_pars(
    Tobs_yrs: float = 4.0, snr: float = 7.0, links: int = 6
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute fiducial shape parameters for the galactic WD binary foreground.

    Fit coefficients from Table 1 of Karnesis et al. 2021 (arXiv:2103.14598).

    Args:
        Tobs_yrs: Observation time in years (must be in [0.25, 10]).
        snr: Signal-to-noise ratio threshold (5 or 7).
        links: Number of LISA links (6 only).
    Returns:
        Tuple ``(Ampl, alpha, fr1, frk, fr2)`` — shape parameters in physical units
        (``fr*`` in Hz).
    """
    if snr not in (5.0, 7.0):
        raise ValueError(f"snr must be 5 or 7, got {snr}")
    if links != 6:
        raise ValueError("Only links=6 is supported")
    if not (0.25 <= Tobs_yrs <= 10.0):
        raise ValueError(
            f"Galaxy fit is valid for Tobs in [0.25, 10] years, got {Tobs_yrs}"
        )

    # Columns: Ampl, alpha, fr2, af1, bf1, afk, bfk
    _L6_snr5 = [1.14e-44, 1.66, 0.00059, -0.15, -2.78, -0.34, -2.55]
    _L6_snr7 = [1.15e-44, 1.56, 0.00067, -0.15, -2.72, -0.37, -2.49]

    Ampl, alpha, fr2, af1, bf1, afk, bfk = _L6_snr5 if snr == 5.0 else _L6_snr7

    log10_T = jnp.log10(Tobs_yrs)
    fr1 = 10.0 ** (af1 * log10_T + bf1)
    frk = 10.0 ** (afk * log10_T + bfk)

    return jnp.array(Ampl), jnp.array(alpha), fr1, frk, jnp.array(fr2)


class GalacticBinaries(AnalyticTemplate):
    r"""
    Galactic binary foreground (Karnesis et al. 2021).

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude.
    alpha
        Exponential cutoff exponent.
    log_fr1
        :math:`\log_{10} f_{r1}` (Hz). Exponential cutoff scale.
    log_frk
        :math:`\log_{10} f_{rk}` (Hz). Knee frequency of the
        ``tanh`` cutoff.
    log_fr2
        :math:`\log_{10} f_{r2}` (Hz). Width of the ``tanh`` cutoff.
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "gb"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Galactic Binaries"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
        "alpha": r"$\alpha$",
        "log_fr1": r"$\log_{10}(f_{r1}/\mathrm{Hz})$",
        "log_frk": r"$\log_{10}(f_{rk}/\mathrm{Hz})$",
        "log_fr2": r"$\log_{10}(f_{r2}/\mathrm{Hz})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -14.0, "max": -5.0},
        "alpha": {"min": 0.1, "max": 10.0},
        "log_fr1": {"min": -4.5, "max": -2.0},
        "log_frk": {"min": -4.5, "max": -2.0},
        "log_fr2": {"min": -5.0, "max": -2.5},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Karnesis:2021tsh,
    author = "Karnesis, Nikolaos and Babak, Stanislav and Pieroni, Mauro and Cornish,
        Neil and Littenberg, Tyson",
    title = "{Characterization of the stochastic signal originating from compact binary
        populations as measured by LISA}",
    eprint = "2103.14598",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.104.043019",
    journal = "Phys. Rev. D",
    volume = "104",
    number = "4",
    pages = "043019",
    year = "2021"
}
""",
    )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        alpha: jax.Array,
        log_fr1: jax.Array,
        log_frk: jax.Array,
        log_fr2: jax.Array,
    ) -> jax.Array:
        fr1 = 10.0**log_fr1
        frk = 10.0**log_frk
        fr2 = 10.0**log_fr2
        return (
            10.0**log_amplitude
            * frequency ** (2.0 / 3.0)
            * jnp.exp(-((frequency / fr1) ** alpha))
            * 0.5
            * (1.0 + jnp.tanh(-(frequency - frk) / fr2))
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, alpha, log_fr1, log_frk,
        log_fr2]."""
        alpha, log_fr1, log_frk, log_fr2 = theta[1], theta[2], theta[3], theta[4]
        fr1 = 10.0**log_fr1
        frk = 10.0**log_frk
        fr2 = 10.0**log_fr2
        u1 = (frequency / fr1) ** alpha
        t = jnp.tanh(-(frequency - frk) / fr2)
        model = GalacticBinaries.omega_gw_h2(self, frequency, *theta)
        ln10 = jnp.log(10.0)
        d_logA = model * ln10
        d_alpha = model * (-u1 * jnp.log(frequency / fr1))
        d_logfr1 = model * (alpha * ln10 * u1)
        d_logfrk = model * (1.0 - t) * frk * ln10 / fr2
        d_logfr2 = model * (1.0 - t) * (frequency - frk) * ln10 / fr2
        return jnp.stack([d_logA, d_alpha, d_logfr1, d_logfrk, d_logfr2], axis=-1)


class GalacticBinariesA(GalacticBinaries):
    r"""
    Galactic binary foreground — amplitude-only variant.

    Free parameter: ``log_amplitude`` only. Inherits :attr:`bibtex_entries` and the
    underlying spectral shape from :class:`GalacticBinaries`, fixing ``alpha``,
    ``log_fr1``, ``log_frk``, ``log_fr2`` at the Karnesis fiducials for ``Tobs_yrs=4``,
    ``snr=7``, ``links=6`` (or values passed at construction).
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "gb_a"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Galactic Binaries (amplitude only)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -14.0, "max": -5.0},
    }

    def __init__(
        self,
        Tobs_yrs: float = 4.0,
        snr: float = 7.0,
        links: int = 6,
        *,
        alpha: jax.Array | None = None,
        log_fr1: jax.Array | None = None,
        log_frk: jax.Array | None = None,
        log_fr2: jax.Array | None = None,
        **kwargs: Any,
    ) -> None:
        _, alpha_fid, fr1_fid, frk_fid, fr2_fid = galactic_pars(Tobs_yrs, snr, links)
        self.alpha: jax.Array = alpha if alpha is not None else alpha_fid
        self.log_fr1: jax.Array = log_fr1 if log_fr1 is not None else jnp.log10(fr1_fid)
        self.log_frk: jax.Array = log_frk if log_frk is not None else jnp.log10(frk_fid)
        self.log_fr2: jax.Array = log_fr2 if log_fr2 is not None else jnp.log10(fr2_fid)

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
            self.log_fr1,
            self.log_frk,
            self.log_fr2,
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        full_theta = jnp.array(
            [theta[0], self.alpha, self.log_fr1, self.log_frk, self.log_fr2]
        )
        return super()._grad_theta_omega_gw_h2_analytical(frequency, full_theta)[:, :1]
