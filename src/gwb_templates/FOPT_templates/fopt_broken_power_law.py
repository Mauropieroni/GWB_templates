r"""
FOPT broken power-law template.

Phenomenological 4-parameter broken power law used as a generic
first-order phase-transition (FOPT) GW background template:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha}\,
        x^{n_{\mathrm{IR}}}\,
        \left(\frac{1 + x}{2}\right)^{n_{\mathrm{UV}} - n_{\mathrm{IR}}}

with :math:`x = f / f_*`. Equivalent to a smooth broken power law with
fixed transition width (:math:`\delta = 1`).

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction
pipeline and physics interpretation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class FoptBrokenPowerLaw(AnalyticTemplate):
    r"""
    FOPT broken power law with fixed transition smoothness (4 parameters).

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude :math:`h^2 \Omega_*` at the break.
    log_f_star
        :math:`\log_{10}` of the break frequency in Hz.
    n_IR
        Low-frequency (infrared) spectral index.
    n_UV
        High-frequency (ultraviolet) spectral index.
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
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
            "log_f_star": r"$\log_{10}(f_*/\mathrm{Hz})$",
            "n_IR": r"$n_{\mathrm{IR}}$",
            "n_UV": r"$n_{\mathrm{UV}}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_f_star": {"min": -5.0, "max": 0.0},
            "n_IR": {"min": 0.0, "max": 10.0},
            "n_UV": {"min": -10.0, "max": 0.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "FOPT Broken Power Law"
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
        log_f_star: ArrayLike,
        n_IR: ArrayLike,
        n_UV: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the FOPT broken-power-law spectrum at ``frequency``.
        """
        x = frequency / 10.0**log_f_star
        return (
            10.0**log_amplitude
            * x**n_IR
            * (0.5 * (1.0 + x)) ** (n_UV - n_IR)
        )
