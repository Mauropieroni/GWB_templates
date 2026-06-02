r"""
Excited-states inflationary GWB template.

Models the GWB produced by excited states during inflation, with a power-law
rise and a sharp Heaviside cutoff at :math:`x_{\mathrm{cut}} = 2\,\gamma_{\rm ES}`
where :math:`x = 0.5 f\, 10^{\log\omega_{\rm ES}}`:

.. math::

    \Omega_{\rm GW} h^2(f) = \frac{10^A}{0.052}\, x^{-3}
        \bigl(1 - (x/(2\gamma_{\rm ES}))^2\bigr)^2\,
        \bigl(\sin x - 4\sin^2(x/2)/x\bigr)^2\,
        \Theta(2\gamma_{\rm ES} - x).

Reference: arXiv:2407.04356.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class ExcitedStates(AnalyticTemplate):
    r"""
    Excited-states inflationary GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` overall amplitude.
    log_gamma_ES
        :math:`\log_{10}\gamma_{\rm ES}`. Sets the location of the Heaviside
        cutoff.
    log_omega_ES
        :math:`\log_{10}\omega_{\rm ES}`. Sets the oscillation frequency.
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
            "log_gamma_ES": r"$\log_{10}(\gamma_{\rm ES})$",
            "log_omega_ES": r"$\log_{10}(\omega_{\rm ES}\,\mathrm{Hz})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_gamma_ES": {"min": -3.0, "max": 3.0},
            "log_omega_ES": {"min": 0.0, "max": 12.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Excited States",
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
        log_gamma_ES: ArrayLike,
        log_omega_ES: ArrayLike,
    ) -> jax.Array:
        amplitude = 10.0**log_amplitude
        gamma_ES = 10.0**log_gamma_ES

        x = 0.5 * jnp.asarray(frequency) * 10.0**log_omega_ES
        x_cut = 2.0 * gamma_ES

        factor_1 = amplitude / 0.052 * x ** (-3)
        factor_2 = (1.0 - (x / (2.0 * gamma_ES)) ** 2) ** 2
        factor_3 = (jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x) ** 2

        return factor_1 * factor_2 * factor_3 * jnp.heaviside(x_cut - x, 1.0)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian of ``omega_gw_h2`` w.r.t. theta.

        Heaviside boundary delta is ignored (consistent with ``jnp.heaviside``).
        """
        log_amplitude, log_gamma_ES, log_omega_ES = theta[0], theta[1], theta[2]

        amplitude = 10.0**log_amplitude
        gamma_ES = 10.0**log_gamma_ES

        fvec = jnp.asarray(frequency)
        x = 0.5 * fvec * 10.0**log_omega_ES
        x_cut = 2.0 * gamma_ES
        u = x / x_cut

        H = jnp.heaviside(x_cut - x, 1.0)
        factor_1 = amplitude / 0.052 * x ** (-3)
        factor_2 = (1.0 - u**2) ** 2
        g = jnp.sin(x) - 4.0 * jnp.sin(x / 2.0) ** 2 / x
        factor_3 = g**2
        model = factor_1 * factor_2 * factor_3 * H

        ln10 = jnp.log(10.0)

        d_logA = model * ln10
        d_lgam = factor_1 * factor_3 * H * 4.0 * ln10 * u**2 * (1.0 - u**2)
        dg_dx = jnp.cos(x) - 2.0 * jnp.sin(x) / x + 2.0 * (1.0 - jnp.cos(x)) / x**2
        d_lomega = H * (
            factor_1 * (-3.0 * ln10) * factor_2 * factor_3
            + factor_1 * (-4.0 * ln10 * u**2 * (1.0 - u**2)) * factor_3
            + factor_1 * factor_2 * (2.0 * g * dg_dx * x * ln10)
        )

        return jnp.stack([d_logA, d_lgam, d_lomega], axis=-1)
