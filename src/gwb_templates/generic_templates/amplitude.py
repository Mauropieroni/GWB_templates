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

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate


class Amplitude(AnalyticTemplate):
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

    def __init__(self, **kwargs: Any) -> None:
        # Shape delegate; the flat spectrum is PowerLaw with tilt fixed to zero.
        self._power_law = PowerLaw()
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the flat spectrum at ``frequency``.

        Args:
            frequency: Frequency value(s) in Hz.
            log_amplitude: :math:`\log_{10}` amplitude.

        Returns:
            Spectrum :math:`\Omega_{\mathrm{GW}} h^2(f)` at each input frequency.
        """
        return self._power_law.omega_gw_h2(frequency, log_amplitude, jnp.array(0.0))

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the flat spectrum, via :class:`PowerLaw`'s own Jacobian
        with ``tilt = 0`` (the ``tilt`` column is dropped, not reweighted).
        """
        (log_amplitude,) = theta
        theta_pl = jnp.stack([log_amplitude, jnp.array(0.0)])
        J = self._power_law._grad_theta_omega_gw_h2_analytical(frequency, theta_pl)
        return J[..., :1]
