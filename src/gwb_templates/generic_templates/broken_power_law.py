r"""
Smooth broken-power-law (BPL) spectrum template.

General-purpose 5-parameter smooth BPL:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha}\,
        \frac{x^{n_1}}{\left(\tfrac{1}{2}(1 + x^{1/\delta})\right)^{(n_1 - n_2)\delta}}

with :math:`x = f / f_*` and :math:`\delta = 10^{\log_{10}\delta}`. At low
frequencies the tilt approaches :math:`n_1`; at high frequencies
:math:`n_2`. Setting ``log_transition = 0`` (:math:`\delta = 1`) recovers
:class:`BrokenPowerLawFixedSmoothness`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class BrokenPowerLaw(AnalyticTemplate):
    r"""
    Smooth broken power law with a tunable transition width (5 parameters).

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
    log_transition
        :math:`\log_{10}\delta`, controlling the transition sharpness.
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
            "log_transition": r"$\log_{10}\delta$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "log_pivot": {"min": -5.0, "max": 0.0},
            "tilt_1": {"min": -10.0, "max": 10.0},
            "tilt_2": {"min": -10.0, "max": 10.0},
            "log_transition": {"min": -3.0, "max": 3.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Broken Power Law"
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
        log_transition: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the smooth broken-power-law spectrum at ``frequency``.
        """
        x = frequency / 10.0**log_pivot
        delta = 10.0**log_transition
        return (
            10.0**log_amplitude
            * x**tilt_1
            / (0.5 * (1.0 + x ** (1.0 / delta))) ** ((tilt_1 - tilt_2) * delta)
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the smooth broken power law.

        Let :math:`x = f/f_*`, :math:`\delta = 10^{\log_{10}\delta}`,
        :math:`t = x^{1/\delta}`. See module docstring for full expressions.
        """
        log_amplitude, log_pivot, tilt_1, tilt_2, log_transition = theta
        x = frequency / 10.0**log_pivot
        delta = 10.0**log_transition
        t = x ** (1.0 / delta)
        model = self.omega_gw_h2(
            frequency, log_amplitude, log_pivot, tilt_1, tilt_2, log_transition
        )
        ln10 = jnp.log(10.0)

        d_logA = model * ln10
        d_logpiv = model * ln10 * (-tilt_1 - tilt_2 * t) / (1.0 + t)
        d_t1 = model * (jnp.log(x) + delta * (jnp.log(2.0) - jnp.log(1.0 + t)))
        d_t2 = model * delta * (jnp.log(1.0 + t) - jnp.log(2.0))
        d_logtrans = (
            model
            * ln10
            * (tilt_1 - tilt_2)
            * (t * jnp.log(x) / (1.0 + t) - delta * jnp.log(0.5 * (1.0 + t)))
        )

        return jnp.stack([d_logA, d_logpiv, d_t1, d_t2, d_logtrans], axis=-1)
