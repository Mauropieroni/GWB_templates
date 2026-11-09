r"""
Broken power-law template with fixed transition smoothness (:math:`\delta = 1`).

Four-parameter version of :class:`BrokenPowerLaw` obtained by fixing
``log_transition = 0``. Parameters: log amplitude, log break frequency, low-frequency
tilt :math:`n_1`, high-frequency tilt :math:`n_2`.

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        x^{n_1}\,
        \left(\tfrac{1+x}{2}\right)^{n_2 - n_1},
    \qquad x = f / f_*.

This is exactly :class:`BrokenPowerLaw` with ``log_transition = 0``, so this class holds
a :class:`BrokenPowerLaw` instance and delegates to it instead of reimplementing the
closed form.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from gwb_templates.generic_templates.broken_power_law import BrokenPowerLaw
from gwb_templates.template import AnalyticTemplate, FrozenParametersMixin


class BrokenPowerLawFixedSmoothness(FrozenParametersMixin, AnalyticTemplate):
    r"""
    Broken power law with fixed transition shape (:math:`\delta = 1`).

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude at the break frequency.
    log_pivot
        :math:`\log_{10}` of the break frequency in Hz.
    tilt_1
        Low-frequency spectral index.
    tilt_2
        High-frequency spectral index.
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "broken_power_law_fixed_smoothness"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Broken Power Law (fixed smoothness)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
        "tilt_1": r"$n_1$",
        "tilt_2": r"$n_2$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "tilt_1": {"min": -10.0, "max": 10.0},
        "tilt_2": {"min": -10.0, "max": 10.0},
    }

    #: TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    # This class is BrokenPowerLaw with log_transition fixed to 0.
    PARENT = BrokenPowerLaw
    FROZEN: ClassVar[Mapping[str, Any]] = {"log_transition": jnp.array(0.0)}
