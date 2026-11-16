r"""
Smooth broken-power-law (BPL) spectrum template.

General-purpose 5-parameter smooth BPL:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        \frac{x^{n_1}}{\left(\tfrac{1}{2}(1 + x^{1/\delta})\right)^{(n_1 - n_2)\delta}}

with :math:`x = f / f_*` and :math:`\delta = 10^{\log_{10}\delta}`. At low
frequencies the tilt approaches :math:`n_1`; at high frequencies
:math:`n_2`. Setting ``log_transition = 0`` (:math:`\delta = 1`) recovers
:class:`BrokenPowerLawFixedSmoothness`.

Numerical evaluation
--------------------
The spectrum is evaluated in log space,

.. math::

    \ln \Omega_{\mathrm{GW}} h^2 = \ln \Omega_* h^2 + n_1 \ln x
        - (n_1 - n_2)\,\delta\,\ln\!\left[\tfrac{1}{2}\big(1 + e^{\ln x/\delta}\big)\right],

with :math:`\ln(1 + e^{y})` computed as ``logaddexp(0, y)``. Forming
:math:`x^{1/\delta}` directly overflows double precision for sharp transitions:
across the LISA band this happens for ``log_transition`` below about -1.9, roughly
10% of the default prior, and the spectrum comes out as exactly 0 or ``inf``. The
log-space form is finite for every :math:`\delta`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_LN_10 = math.log(10.0)
_LN_2 = math.log(2.0)


def _log_terms(
    frequency: ArrayLike, log_pivot: ArrayLike, log_transition: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Return :math:`\ln x`, :math:`\delta` and :math:`\ln[\tfrac12(1 + x^{1/\delta})]`.

    The last term is computed as ``logaddexp(0, ln x / delta) - ln 2``, so it stays
    finite for any :math:`\delta`.
    """
    log_x = jnp.log(frequency) - log_pivot * _LN_10
    delta = 10.0**log_transition
    log_denominator = jnp.logaddexp(0.0, log_x / delta) - _LN_2
    return log_x, delta, log_denominator


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

        Computed in log space (see the module docstring), so a sharp transition
        does not overflow.
        """
        log_x, delta, log_denominator = _log_terms(frequency, log_pivot, log_transition)
        return jnp.exp(
            log_amplitude * _LN_10
            + tilt_1 * log_x
            - (tilt_1 - tilt_2) * delta * log_denominator
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the smooth broken power law.

        Let :math:`x = f/f_*`, :math:`\delta = 10^{\log_{10}\delta}`,
        :math:`t = x^{1/\delta}`, :math:`D = \ln[\tfrac12(1 + t)]` and
        :math:`u = t/(1 + t)`. Then, with :math:`\Omega` the model,

        .. math::

            \begin{aligned}
            \partial_{\log_{10}\Omega_*}\Omega &= \Omega \ln 10, \\
            \partial_{\log_{10} f_*}\Omega &= -\Omega \ln 10\,[n_1 (1 - u) + n_2 u], \\
            \partial_{n_1}\Omega &= \Omega\,(\ln x - \delta D), \\
            \partial_{n_2}\Omega &= \Omega\,\delta D, \\
            \partial_{\log_{10}\delta}\Omega &= \Omega \ln 10\,(n_1 - n_2)(u \ln x - \delta D).
            \end{aligned}

        :math:`u` is ``sigmoid(ln x / delta)`` and :math:`D` comes from
        :func:`_log_terms`, so :math:`t` is never formed and the Jacobian stays
        finite for sharp transitions.
        """
        log_amplitude, log_pivot, tilt_1, tilt_2, log_transition = theta
        log_x, delta, log_denominator = _log_terms(frequency, log_pivot, log_transition)
        upper = jax.nn.sigmoid(log_x / delta)
        model = self.omega_gw_h2(
            frequency, log_amplitude, log_pivot, tilt_1, tilt_2, log_transition
        )

        d_logA = model * _LN_10
        d_logpiv = -model * _LN_10 * (tilt_1 * (1.0 - upper) + tilt_2 * upper)
        d_t1 = model * (log_x - delta * log_denominator)
        d_t2 = model * delta * log_denominator
        d_logtrans = (
            model
            * _LN_10
            * (tilt_1 - tilt_2)
            * (upper * log_x - delta * log_denominator)
        )

        return jnp.stack([d_logA, d_logpiv, d_t1, d_t2, d_logtrans], axis=-1)
