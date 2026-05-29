r"""
Extragalactic compact binary merger foreground template.

Power-law spectrum with the 2/3 spectral index expected from GW emission
during the inspiral phase of compact binary mergers integrated over redshift.

Two variants are provided:

* :class:`ExtragalacticSobbhBns` — 2-parameter model (log amplitude + tilt).
* :class:`ExtragalacticSobbhBnsA` — 1-parameter amplitude-only model with
  tilt fixed at 2/3.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jtp.ArrayLike


class ExtragalacticSobbhBns(AnalyticTemplate):
    r"""
    Extragalactic SOBBH+BNS foreground with free spectral index.

    .. math::

        \Omega_{\mathrm{GW}} h^2(f) =
            10^{\log_{10} A}\,(f / f_{\mathrm{ref}})^{\alpha}

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude at the reference frequency.
    tilt
        Spectral index.

    Configuration
    -------------
    ref_freq
        Reference frequency (Hz). Defaults to 1 mHz.
    """

    #: Default reference frequency in Hz.
    DEFAULT_REF_FREQ: ClassVar[float] = 1e-3

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        ref_freq: float = DEFAULT_REF_FREQ,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.ref_freq: float = float(ref_freq)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
            "tilt": r"$\alpha_{\rm EG}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "tilt": {"min": -3.0, "max": 5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Extragalactic SOBBH+BNS"
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
        frequency: Array,
        log_amplitude: Array,
        tilt: Array,
    ) -> Array:
        x = jnp.asarray(frequency) / self.ref_freq
        return 10.0**log_amplitude * x**tilt


class ExtragalacticSobbhBnsA(AnalyticTemplate):
    r"""
    Extragalactic SOBBH+BNS foreground with tilt fixed to 2/3.

    Free parameter: ``log_amplitude`` only.
    """

    DEFAULT_REF_FREQ: ClassVar[float] = 1e-3
    DEFAULT_TILT: ClassVar[float] = 2.0 / 3.0

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        ref_freq: float = DEFAULT_REF_FREQ,
        tilt: float = DEFAULT_TILT,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.ref_freq: float = float(ref_freq)
        self.tilt: float = float(tilt)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Extragalactic SOBBH+BNS (amplitude only)"
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
        frequency: Array,
        log_amplitude: Array,
    ) -> Array:
        x = jnp.asarray(frequency) / self.ref_freq
        return 10.0**log_amplitude * x**self.tilt
