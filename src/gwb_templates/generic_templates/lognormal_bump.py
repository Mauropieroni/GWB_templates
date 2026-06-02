r"""
Log-normal bump spectrum template.

Three-parameter Gaussian in :math:`\log_{10}`-frequency space:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha}\,
        \exp\!\left[-\tfrac{1}{2}\left(\frac{\log_{10}(f/f_*)}{\sigma}\right)^2\right].

Used as a standalone phenomenological model and as an envelope for
combined inflation templates.
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


class LognormalBump(AnalyticTemplate):
    r"""
    Log-normal bump GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` peak amplitude.
    log_pivot
        :math:`\log_{10}` of the peak frequency in Hz.
    log_width
        :math:`\log_{10}\sigma`, the width of the bump in
        :math:`\log_{10}`-frequency.
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
            "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
            "log_width": r"$\log_{10}\sigma$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_pivot": {"min": -5.0, "max": 0.0},
            "log_width": {"min": -2.0, "max": 1.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(model_label if model_label is not None else "Lognormal Bump"),
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
        log_pivot: Array,
        log_width: Array,
    ) -> jax.Array:
        r"""
        Evaluate the log-normal bump spectrum at ``frequency``.
        """
        pivot = 10.0**log_pivot
        width = 10.0**log_width
        return 10.0**log_amplitude * jnp.exp(
            -0.5 * (jnp.log10(frequency / pivot) / width) ** 2
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: Array,
        theta: Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the log-normal bump.

        With :math:`u = \log_{10}(f/f_*)` and
        :math:`\sigma = 10^{\log_{10}\sigma}`:

        - :math:`\partial/\partial(\log_{10}A) = \text{model}\cdot\ln 10`
        - :math:`\partial/\partial(\log_{10}f_*) = \text{model}\cdot u/\sigma^2`
        - :math:`\partial/\partial(\log_{10}\sigma)
        = \text{model}\cdot u^2 \ln 10/\sigma^2`
        """
        log_amplitude, log_pivot, log_width = theta
        pivot = 10.0**log_pivot
        width = 10.0**log_width
        model = self.omega_gw_h2(frequency, log_amplitude, log_pivot, log_width)
        u = jnp.log10(frequency / pivot)

        d_log_A = model * jnp.log(10.0)
        d_log_piv = model * u / width**2
        d_log_wid = model * u**2 * jnp.log(10.0) / width**2

        return jnp.stack([d_log_A, d_log_piv, d_log_wid], axis=-1)
