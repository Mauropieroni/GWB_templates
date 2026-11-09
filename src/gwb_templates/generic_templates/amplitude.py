"""
Amplitude-only (flat) spectrum template.

Single-parameter model for a frequency-independent GWB:

.. math::

    \\Omega_{\\mathrm{GW}} h^2(f) = 10^{\\alpha}

This is exactly :class:`~gwb_templates.generic_templates.power_law.PowerLaw` with
``tilt = 0``, so this class holds a :class:`PowerLaw` instance and delegates to it
instead of reimplementing the closed form. Used as a simple baseline model or as a
building block for composite spectra.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate, FrozenParametersMixin


class Amplitude(FrozenParametersMixin, AnalyticTemplate):
    r"""
    Flat (amplitude-only) GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the (frequency-independent) amplitude.
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "amplitude"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Amplitude"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
    }

    #: TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    # The flat spectrum is PowerLaw with tilt fixed to zero.
    PARENT = PowerLaw
    FROZEN: ClassVar[Mapping[str, Any]] = {"tilt": jnp.array(0.0)}
