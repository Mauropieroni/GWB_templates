"""
Amplitude-only (flat) spectrum template.

Single-parameter model for a frequency-independent GWB:

.. math::

    \\Omega_{\\mathrm{GW}} h^2(f) = 10^{\\alpha}

Used as a simple baseline model or as a building block for composite
spectra.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class Amplitude(AnalyticTemplate):
    r"""
    Flat (amplitude-only) GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the (frequency-independent) amplitude.
    """

    #: TODO: cite
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
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Amplitude",
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
        r"""
        Evaluate the flat spectrum at ``frequency``.

        Args:
            frequency: Frequency value(s) in Hz.
            log_amplitude: :math:`\log_{10}` amplitude.

        Returns:
            Spectrum :math:`\Omega_{\mathrm{GW}} h^2(f)` at each input
            frequency.
        """
        return 10.0**log_amplitude * jnp.ones_like(frequency)
