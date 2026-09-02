r"""
Broken power-law template with fixed transition smoothness
(:math:`\delta = 1`).

Four-parameter version of :class:`BrokenPowerLaw` obtained by fixing
``log_transition = 0``. Parameters: log amplitude, log break frequency,
low-frequency tilt :math:`n_1`, high-frequency tilt :math:`n_2`.

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        x^{n_1}\,
        \left(\tfrac{1+x}{2}\right)^{n_2 - n_1},
    \qquad x = f / f_*.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class BrokenPowerLawFixedSmoothness(AnalyticTemplate):
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
            "tilt_1": r"$n_1$",
            "tilt_2": r"$n_2$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_pivot": {"min": -5.0, "max": 0.0},
            "tilt_1": {"min": -10.0, "max": 10.0},
            "tilt_2": {"min": -10.0, "max": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Broken Power Law (fixed smoothness)"
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
        log_pivot: ArrayLike,
        tilt_1: ArrayLike,
        tilt_2: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the fixed-smoothness broken power law at ``frequency``.
        """
        x = frequency / 10.0**log_pivot
        return (
            10.0**log_amplitude
            * x**tilt_1
            * (0.5 * (1.0 + x)) ** (tilt_2 - tilt_1)
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian. Special case of the 5-param BPL with
        :math:`\delta = 1` (so :math:`t = x`); the
        :math:`\partial/\partial\log_{10}\delta` column is dropped.
        """
        log_amplitude, log_pivot, tilt_1, tilt_2 = theta
        x = frequency / 10.0**log_pivot
        t = x  # delta = 1
        model = self.omega_gw_h2(
            frequency, log_amplitude, log_pivot, tilt_1, tilt_2
        )
        ln10 = jnp.log(10.0)

        d_logA = model * ln10
        d_logpiv = model * ln10 * (-tilt_1 - tilt_2 * t) / (1.0 + t)
        d_t1 = model * (jnp.log(x) + (jnp.log(2.0) - jnp.log(1.0 + t)))
        d_t2 = model * (jnp.log(1.0 + t) - jnp.log(2.0))

        return jnp.stack([d_logA, d_logpiv, d_t1, d_t2], axis=-1)
