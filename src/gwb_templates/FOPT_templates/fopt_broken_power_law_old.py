r"""
FOPT broken power law (old / 2-parameter) template.

Implements the broken power law from Eq. 14 of arXiv:1512.06239:

.. math::

    \Omega_\mathrm{GW} h^2(f) = \Omega_* h^2\, x^{3}
        \left(\frac{7}{4 + 3 x^{2}}\right)^{7/2}, \qquad x = f / f_*.

A two-parameter special case of the smooth BPL with fixed :math:`n_1 = 3`,
:math:`n_2 = -4`, :math:`\delta = 1/2`, written so that ``log_amplitude`` and
``log_f_star`` are the log10 of the peak amplitude and peak frequency respectively.

This is a reparametrisation of
:class:`~gwb_templates.generic_templates.broken_power_law_a1.BrokenPowerLawA1` with
``n_1 = 3``, ``n_2 = -4``, ``a_1 = 2`` (equivalently, :class:`BrokenPowerLaw` with
``delta = 1/2``): the formula's maximum sits not at the break frequency ``f_b`` of that
generic parametrisation but at :math:`f_b \sqrt{3}/2`, with peak value scaled by a fixed
constant relative to the amplitude at :math:`f_b`. Inverting that exactly gives the
peak-frequency/peak-amplitude convention used here; see ``_LOG_FREQ_RATIO`` and
``_LOG_AMPLITUDE_RATIO`` below. This class therefore holds a :class:`BrokenPowerLawA1`
instance and delegates to it instead of reimplementing the closed form.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.broken_power_law_a1 import BrokenPowerLawA1
from gwb_templates.template import AnalyticTemplate

# The n_1=3, n_2=-4, a_1=2 formula peaks not at its own break frequency f_b but at
# f_b * sqrt(3)/2, with peak value Omega(f_b) * (9/16) * (7/6)**3.5. Inverting gives
# the maps from this class's (peak-frequency, peak-amplitude) convention to
# BrokenPowerLawA1's (break-frequency, amplitude-at-break) convention:
#   f_b = f_star * 2/sqrt(3),  Omega_b = Omega_star * (16/9) * (6/7)**3.5
_LOG_FREQ_RATIO = jnp.log10(2.0 / jnp.sqrt(3.0))  # log10(f_b / f_star)
_LOG_AMPLITUDE_RATIO = jnp.log10(9.0 / 16.0) + 3.5 * jnp.log10(
    7.0 / 6.0
)  # log10(Omega_b / Omega_star)


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

    #: Fixed spectral exponents (n_1, n_2, a_1) in the BrokenPowerLawA1 parametrisation.
    SPECTRAL_INDEX_IR: ClassVar[jax.Array] = jnp.array(3.0)
    SPECTRAL_INDEX_UV: ClassVar[jax.Array] = jnp.array(-4.0)
    TRANSITION_SMOOTHNESS: ClassVar[jax.Array] = jnp.array(2.0)
    #: log10(f_b/f_star) and log10(Omega_b/Omega_star); see module docstring/comments.
    LOG_FREQ_RATIO: ClassVar[jax.Array] = _LOG_FREQ_RATIO
    LOG_AMPLITUDE_RATIO: ClassVar[jax.Array] = _LOG_AMPLITUDE_RATIO

    DEFAULT_MODEL_NAME: ClassVar[str] = "fopt_broken_power_law_old"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "FOPT Broken Power Law (old)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_f_star": r"$\log_{10}(f_*/\mathrm{Hz})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_f_star": {"min": -5.0, "max": 0.0},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2015zlo,
    author = "Caprini, Chiara and others",
    title = "{Science with the space-based interferometer eLISA. II: Gravitational waves
        from cosmological phase transitions}",
    eprint = "1512.06239",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESY-15-246",
    doi = "10.1088/1475-7516/2016/04/001",
    journal = "JCAP",
    volume = "04",
    pages = "001",
    year = "2016"
}
""",
    )

    def __init__(self, **kwargs: Any) -> None:
        # Spectral shape, reparametrised from (peak-amplitude, peak-frequency) to
        # BrokenPowerLawA1's (amplitude-at-break, break-frequency) convention.
        self._bpl_a1 = BrokenPowerLawA1()
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_f_star: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the 2-parameter FOPT broken power-law spectrum at ``frequency``.
        """
        log_f_b = log_f_star + self.LOG_FREQ_RATIO
        log_amplitude_b = log_amplitude + self.LOG_AMPLITUDE_RATIO
        return self._bpl_a1.omega_gw_h2(
            frequency,
            log_amplitude_b,
            log_f_b,
            self.SPECTRAL_INDEX_IR,
            self.SPECTRAL_INDEX_UV,
            self.TRANSITION_SMOOTHNESS,
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian w.r.t. ``(log_amplitude, log_f_star)``; shape
        ``frequency.shape + (2,)``.

        The reparametrisation to :class:`BrokenPowerLawA1`'s ``(log_amplitude,
        log_f_b)`` is an additive shift by fixed constants, so both partials pass
        straight through the held sub-template's Jacobian unchanged.
        """
        log_amplitude, log_f_star = theta[0], theta[1]
        log_f_b = log_f_star + self.LOG_FREQ_RATIO
        log_amplitude_b = log_amplitude + self.LOG_AMPLITUDE_RATIO
        theta_bpl = jnp.stack(
            [
                log_amplitude_b,
                log_f_b,
                self.SPECTRAL_INDEX_IR,
                self.SPECTRAL_INDEX_UV,
                self.TRANSITION_SMOOTHNESS,
            ]
        )
        J_full = self._bpl_a1._grad_theta_omega_gw_h2_analytical(frequency, theta_bpl)
        return J_full[..., :2]
