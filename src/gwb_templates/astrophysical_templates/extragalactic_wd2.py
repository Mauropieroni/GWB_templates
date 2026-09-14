r"""
Alternative extragalactic white-dwarf binary foreground template.

Double-transition broken power law with a fixed 2/3 mid-slope:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,(f/f_0)^{2/3}
        \bigl(1 + (f_{\mathrm{low}}/f)^{\delta}\bigr)^{(\alpha_{\rm low} - 2/3)/\delta}
        \bigl(1 + (f/f_{\mathrm{high}})^{\delta}\bigr)^{(\alpha_{\rm high}
        - 2/3)/\delta}

with fixed :math:`f_0 = 10^{-3}` Hz, mid-slope 2/3, and :math:`\delta = 2`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class ExtragalacticWd2(AnalyticTemplate):
    r"""
    Alternative extragalactic WD-binary foreground (5-parameter form).

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude at ``f0``.
    f_low
        Low-frequency knee (Hz).
    f_high
        High-frequency knee (Hz).
    alpha_low
        Low-frequency slope.
    alpha_high
        High-frequency slope.

    Configuration
    -------------
    f0
        Reference frequency (Hz), default 1 mHz.
    mid_slope
        Spectral index between the two knees, default 2/3.
    delta
        Transition width, default 2.
    """

    DEFAULT_F0: ClassVar[jax.Array] = jnp.array(1e-3)
    DEFAULT_MID_SLOPE: ClassVar[jax.Array] = jnp.array(2.0 / 3.0)
    DEFAULT_DELTA: ClassVar[jax.Array] = jnp.array(2.0)

    DEFAULT_MODEL_NAME: ClassVar[str] = "ewd2"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Extragalactic WD Binaries (alt.)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$",
        "f_low": r"$f_{\rm low}$",
        "f_high": r"$f_{\rm high}$",
        "alpha_low": r"$\alpha_{\rm low}$",
        "alpha_high": r"$\alpha_{\rm high}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "f_low": {"min": 1e-5, "max": 0.1},
        "f_high": {"min": 1e-4, "max": 1.0},
        "alpha_low": {"min": -5.0, "max": 5.0},
        "alpha_high": {"min": -5.0, "max": 5.0},
    }

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        f0: jax.Array = DEFAULT_F0,
        mid_slope: jax.Array = DEFAULT_MID_SLOPE,
        delta: jax.Array = DEFAULT_DELTA,
        **kwargs: Any,
    ) -> None:
        self.f0: jax.Array = f0
        self.mid_slope: jax.Array = mid_slope
        self.delta: jax.Array = delta
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        f_low: jax.Array,
        f_high: jax.Array,
        alpha_low: jax.Array,
        alpha_high: jax.Array,
    ) -> jax.Array:
        L = 1.0 + (f_low / frequency) ** self.delta
        H = 1.0 + (frequency / f_high) ** self.delta
        low_term = L ** ((self.mid_slope - alpha_low) / self.delta)
        high_term = H ** ((alpha_high - self.mid_slope) / self.delta)
        return (
            10.0**log_amplitude
            * (frequency / self.f0) ** self.mid_slope
            * low_term
            * high_term
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, f_low, f_high, alpha_low,
        alpha_high]."""
        log_amplitude, f_low, f_high, alpha_low, alpha_high = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
        )
        L = 1.0 + (f_low / frequency) ** self.delta
        H = 1.0 + (frequency / f_high) ** self.delta
        model = ExtragalacticWd2.omega_gw_h2(
            self, frequency, log_amplitude, f_low, f_high, alpha_low, alpha_high
        )
        d_logA = model * jnp.log(10.0)
        d_flow = (
            model
            * (self.mid_slope - alpha_low)
            / f_low
            * (f_low / frequency) ** self.delta
            / L
        )
        d_fhigh = (
            model
            * (-(alpha_high - self.mid_slope))
            / f_high
            * (frequency / f_high) ** self.delta
            / H
        )
        d_alpha_low = -model * (1.0 / self.delta) * jnp.log(L)
        d_alpha_high = model * (1.0 / self.delta) * jnp.log(H)
        return jnp.stack([d_logA, d_flow, d_fhigh, d_alpha_low, d_alpha_high], axis=-1)


class ExtragalacticWd2A(ExtragalacticWd2):
    r"""
    Alternative extragalactic WD-binary foreground — amplitude-only variant.

    Free parameter: ``log_amplitude`` only. Inherits :attr:`bibtex_entries` and the
    underlying spectral shape from :class:`ExtragalacticWd2`, fixing ``f_low``,
    ``f_high``, ``alpha_low``, ``alpha_high`` at fiducial values (or values passed at
    construction); ``f0``, ``mid_slope`` and ``delta`` remain configurable as in the
    base class.
    """

    DEFAULT_F_LOW: ClassVar[jax.Array] = jnp.array(3e-4)
    DEFAULT_F_HIGH: ClassVar[jax.Array] = jnp.array(1.5e-2)
    DEFAULT_ALPHA_LOW: ClassVar[jax.Array] = jnp.array(0.0)
    DEFAULT_ALPHA_HIGH: ClassVar[jax.Array] = jnp.array(-5.5)

    DEFAULT_MODEL_NAME: ClassVar[str] = "ewd2_a"
    DEFAULT_MODEL_LABEL: ClassVar[str] = (
        "Extragalactic WD Binaries alt. (amplitude only)"
    )
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
    }

    def __init__(
        self,
        f_low: jax.Array = DEFAULT_F_LOW,
        f_high: jax.Array = DEFAULT_F_HIGH,
        alpha_low: jax.Array = DEFAULT_ALPHA_LOW,
        alpha_high: jax.Array = DEFAULT_ALPHA_HIGH,
        f0: jax.Array = ExtragalacticWd2.DEFAULT_F0,
        mid_slope: jax.Array = ExtragalacticWd2.DEFAULT_MID_SLOPE,
        delta: jax.Array = ExtragalacticWd2.DEFAULT_DELTA,
        **kwargs: Any,
    ) -> None:
        self.f_low: jax.Array = f_low
        self.f_high: jax.Array = f_high
        self.alpha_low: jax.Array = alpha_low
        self.alpha_high: jax.Array = alpha_high

        super().__init__(f0=f0, mid_slope=mid_slope, delta=delta, **kwargs)

    def omega_gw_h2(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
    ) -> jax.Array:
        return super().omega_gw_h2(
            frequency,
            log_amplitude,
            self.f_low,
            self.f_high,
            self.alpha_low,
            self.alpha_high,
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        full_theta = jnp.array(
            [theta[0], self.f_low, self.f_high, self.alpha_low, self.alpha_high]
        )
        return super()._grad_theta_omega_gw_h2_analytical(frequency, full_theta)[:, :1]
