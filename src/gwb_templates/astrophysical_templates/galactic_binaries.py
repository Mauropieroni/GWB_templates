r"""
Galactic compact binary foreground template.

Implements Eq. 6 of Karnesis et al. 2021 (arXiv:2103.14598):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,f^{2/3}\,
        \exp\bigl(-(f/f_{r1})^{\alpha}\bigr)\,
        \tfrac{1}{2}\bigl(1 + \tanh(-(f - f_{rk})/f_{r2})\bigr)

A helper :func:`galactic_pars` computes the observation-dependent
fiducial shape parameters from the Karnesis fit table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


def galactic_pars(
    Tobs_yrs: float = 4.0, snr: float = 7.0, links: int = 6
) -> tuple[float, float, float, float, float]:
    """
    Compute fiducial shape parameters for the galactic WD binary foreground.

    Fit coefficients from Table 1 of Karnesis et al. 2021 (arXiv:2103.14598).

    Args:
        Tobs_yrs: Observation time in years (must be in [0.25, 10]).
        snr: Signal-to-noise ratio threshold (5 or 7).
        links: Number of LISA links (6 only).
    Returns:
        Tuple ``(Ampl, alpha, fr1, frk, fr2)`` — shape parameters in
        physical units (``fr*`` in Hz).
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

    log10_T = math.log10(Tobs_yrs)
    fr1 = 10.0 ** (af1 * log10_T + bf1)
    frk = 10.0 ** (afk * log10_T + bfk)

    return Ampl, alpha, fr1, frk, fr2


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
            "log_fr1": r"$\log_{10}(f_{r1}/\mathrm{Hz})$",
            "log_frk": r"$\log_{10}(f_{rk}/\mathrm{Hz})$",
            "log_fr2": r"$\log_{10}(f_{r2}/\mathrm{Hz})$",
        }
        default_priors = {
            "log_amplitude": {"min": -14.0, "max": -5.0},
            "alpha": {"min": 0.1, "max": 10.0},
            "log_fr1": {"min": -4.5, "max": -2.0},
            "log_frk": {"min": -4.5, "max": -2.0},
            "log_fr2": {"min": -5.0, "max": -2.5},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Galactic Binaries"
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
        log_fr1: ArrayLike,
        log_frk: ArrayLike,
        log_fr2: ArrayLike,
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        fr1 = 10.0**log_fr1
        frk = 10.0**log_frk
        fr2 = 10.0**log_fr2
        return (
            10.0**log_amplitude
            * fvec ** (2.0 / 3.0)
            * jnp.exp(-((fvec / fr1) ** alpha))
            * 0.5
            * (1.0 + jnp.tanh(-(fvec - frk) / fr2))
        )


class GalacticBinariesA(AnalyticTemplate):
    r"""
    Galactic binary foreground — amplitude-only variant.

    Shape parameters default to the Karnesis fiducials at
    ``Tobs_yrs=4``, ``snr=7``, ``links=6``, but can be overridden via
    constructor kwargs.
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        Tobs_yrs: float = 4.0,
        snr: float = 7.0,
        links: int = 6,
        *,
        alpha: float | None = None,
        log_fr1: float | None = None,
        log_frk: float | None = None,
        log_fr2: float | None = None,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        _, alpha_fid, fr1_fid, frk_fid, fr2_fid = galactic_pars(Tobs_yrs, snr, links)
        self.alpha: float = float(alpha) if alpha is not None else float(alpha_fid)
        self.log_fr1: float = (
            float(log_fr1) if log_fr1 is not None else math.log10(fr1_fid)
        )
        self.log_frk: float = (
            float(log_frk) if log_frk is not None else math.log10(frk_fid)
        )
        self.log_fr2: float = (
            float(log_fr2) if log_fr2 is not None else math.log10(fr2_fid)
        )

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
                else "Galactic Binaries (amplitude only)"
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
        fr1 = 10.0**self.log_fr1
        frk = 10.0**self.log_frk
        fr2 = 10.0**self.log_fr2
        return (
            10.0**log_amplitude
            * fvec ** (2.0 / 3.0)
            * jnp.exp(-((fvec / fr1) ** self.alpha))
            * 0.5
            * (1.0 + jnp.tanh(-(fvec - frk) / fr2))
        )
