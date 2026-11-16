r"""
Smooth broken-power-law (BPL) spectrum template.

General-purpose 5-parameter smooth BPL:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = \Omega_* h^2\,
        \frac{x^{n_1}}{\left(\tfrac{1}{2}(1 + x^{1/\delta})\right)^{(n_1 - n_2)\delta}}

with :math:`x = f / f_*` and :math:`\delta = 10^{\log_{10}\delta}`. At low frequencies
the tilt approaches :math:`n_1`; at high frequencies :math:`n_2`. Setting
``log_transition = 0`` (:math:`\delta = 1`) recovers
:class:`BrokenPowerLawFixedSmoothness`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


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

    DEFAULT_MODEL_NAME: ClassVar[str] = "broken_power_law"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Broken Power Law"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
        "log_pivot": r"$\log_{10}(f_*/\mathrm{Hz})$",
        "tilt_1": r"$n_1$",
        "tilt_2": r"$n_2$",
        "log_transition": r"$\log_{10}\delta$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_pivot": {"min": -5.0, "max": 0.0},
        "tilt_1": {"min": -10.0, "max": 10.0},
        "tilt_2": {"min": -10.0, "max": 10.0},
        "log_transition": {"min": -3.0, "max": 3.0},
    }

    #: TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_pivot: jax.Array,
        tilt_1: jax.Array,
        tilt_2: jax.Array,
        log_transition: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the smooth broken-power-law spectrum at ``frequency``.
        """
        log_x = jnp.log(frequency / 10.0**log_pivot)
        delta = 10.0**log_transition
        # log(1 + x**(1/delta)), via logaddexp rather than forming x**(1/delta)
        # directly: for sharp transitions (delta << 1) that power overflows to inf
        # long before this log does, which would otherwise poison the Jacobian with
        # 0 * inf / inf - inf cancellations (see _grad_theta_omega_gw_h2_analytical).
        log1p_t = jnp.logaddexp(0.0, log_x / delta)
        return 10.0**log_amplitude * jnp.exp(
            tilt_1 * log_x - (tilt_1 - tilt_2) * delta * (log1p_t - jnp.log(2.0))
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the smooth broken power law.

        Let :math:`x = f/f_*`, :math:`\delta = 10^{\log_{10}\delta}`,
        :math:`t = x^{1/\delta}`. Written via :math:`\log(1 + t)` (``jnp.logaddexp``)
        and :math:`t/(1+t) = \mathrm{sigmoid}(\log x/\delta)` rather than forming
        :math:`t` itself, which overflows to ``inf`` for sharp transitions
        (:math:`\delta \ll 1`) and poisons every term below with ``0 * inf``.
        """
        log_amplitude, log_pivot, tilt_1, tilt_2, log_transition = theta
        log_x = jnp.log(frequency / 10.0**log_pivot)
        delta = 10.0**log_transition
        u = log_x / delta
        log1p_t = jnp.logaddexp(0.0, u)
        sig = jax.nn.sigmoid(u)  # t / (1 + t)
        model = self.omega_gw_h2(
            frequency, log_amplitude, log_pivot, tilt_1, tilt_2, log_transition
        )
        ln10 = jnp.log(10.0)
        log2 = jnp.log(2.0)

        d_logA = model * ln10
        d_logpiv = model * ln10 * (-tilt_1 * (1.0 - sig) - tilt_2 * sig)
        d_t1 = model * (log_x + delta * (log2 - log1p_t))
        d_t2 = model * delta * (log1p_t - log2)
        d_logtrans = (
            model * ln10 * (tilt_1 - tilt_2) * (sig * log_x - delta * (log1p_t - log2))
        )

        return jnp.stack([d_logA, d_logpiv, d_t1, d_t2, d_logtrans], axis=-1)
