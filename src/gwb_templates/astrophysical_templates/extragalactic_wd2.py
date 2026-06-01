r"""
Alternative extragalactic white-dwarf binary foreground template.

Double-transition broken power law with a fixed 2/3 mid-slope:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,(f/f_0)^{2/3}
        \bigl(1 + (f_{\mathrm{low}}/f)^{\delta}\bigr)^{(\alpha_{\rm low} - 2/3)/\delta}
        \bigl(1 + (f/f_{\mathrm{high}})^{\delta}\bigr)^{(\alpha_{\rm high} - 2/3)/\delta}

with fixed :math:`f_0 = 10^{-3}` Hz, mid-slope 2/3, and :math:`\delta = 2`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


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

    DEFAULT_F0: ClassVar[float] = 1e-3
    DEFAULT_MID_SLOPE: ClassVar[float] = 2.0 / 3.0
    DEFAULT_DELTA: ClassVar[float] = 2.0

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        f0: float = DEFAULT_F0,
        mid_slope: float = DEFAULT_MID_SLOPE,
        delta: float = DEFAULT_DELTA,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.f0: float = float(f0)
        self.mid_slope: float = float(mid_slope)
        self.delta: float = float(delta)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$",
            "f_low": r"$f_{\rm low}$",
            "f_high": r"$f_{\rm high}$",
            "alpha_low": r"$\alpha_{\rm low}$",
            "alpha_high": r"$\alpha_{\rm high}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "f_low": {"min": 1e-5, "max": 0.1},
            "f_high": {"min": 1e-4, "max": 1.0},
            "alpha_low": {"min": -5.0, "max": 5.0},
            "alpha_high": {"min": -5.0, "max": 5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Extragalactic WD Binaries (alt.)"
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
        f_low: ArrayLike,
        f_high: ArrayLike,
        alpha_low: ArrayLike,
        alpha_high: ArrayLike,
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        L = 1.0 + (f_low / fvec) ** self.delta
        H = 1.0 + (fvec / f_high) ** self.delta
        low_term = L ** ((alpha_low - self.mid_slope) / self.delta)
        high_term = H ** ((alpha_high - self.mid_slope) / self.delta)
        return (
            10.0**log_amplitude
            * (fvec / self.f0) ** self.mid_slope
            * low_term
            * high_term
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, f_low, f_high, alpha_low, alpha_high]."""
        log_amplitude, f_low, f_high, alpha_low, alpha_high = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
        )
        fvec = jnp.asarray(frequency)
        L = 1.0 + (f_low / fvec) ** self.delta
        H = 1.0 + (fvec / f_high) ** self.delta
        model = self.omega_gw_h2(
            fvec, log_amplitude, f_low, f_high, alpha_low, alpha_high
        )
        d_logA = model * jnp.log(10.0)
        d_flow = (
            model * (alpha_low - self.mid_slope) / f_low * (f_low / fvec) ** self.delta / L
        )
        d_fhigh = (
            model
            * (-(alpha_high - self.mid_slope))
            / f_high
            * (fvec / f_high) ** self.delta
            / H
        )
        d_alpha_low = model * (1.0 / self.delta) * jnp.log(L)
        d_alpha_high = model * (1.0 / self.delta) * jnp.log(H)
        return jnp.stack(
            [d_logA, d_flow, d_fhigh, d_alpha_low, d_alpha_high], axis=-1
        )


class ExtragalacticWd2A(AnalyticTemplate):
    r"""
    Alternative extragalactic WD-binary foreground — amplitude-only variant.

    Shape parameters are fixed at fiducial values but can be overridden
    via constructor kwargs.
    """

    DEFAULT_F0: ClassVar[float] = 1e-3
    DEFAULT_MID_SLOPE: ClassVar[float] = 2.0 / 3.0
    DEFAULT_DELTA: ClassVar[float] = 2.0
    DEFAULT_F_LOW: ClassVar[float] = 3e-4
    DEFAULT_F_HIGH: ClassVar[float] = 1.5e-2
    DEFAULT_ALPHA_LOW: ClassVar[float] = 0.0
    DEFAULT_ALPHA_HIGH: ClassVar[float] = -5.5

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        f_low: float = DEFAULT_F_LOW,
        f_high: float = DEFAULT_F_HIGH,
        alpha_low: float = DEFAULT_ALPHA_LOW,
        alpha_high: float = DEFAULT_ALPHA_HIGH,
        f0: float = DEFAULT_F0,
        mid_slope: float = DEFAULT_MID_SLOPE,
        delta: float = DEFAULT_DELTA,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.f_low: float = float(f_low)
        self.f_high: float = float(f_high)
        self.alpha_low: float = float(alpha_low)
        self.alpha_high: float = float(alpha_high)
        self.f0: float = float(f0)
        self.mid_slope: float = float(mid_slope)
        self.delta: float = float(delta)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Extragalactic WD Binaries alt. (amplitude only)"
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
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        L = 1.0 + (self.f_low / fvec) ** self.delta
        H = 1.0 + (fvec / self.f_high) ** self.delta
        low_term = L ** ((self.alpha_low - self.mid_slope) / self.delta)
        high_term = H ** ((self.alpha_high - self.mid_slope) / self.delta)
        return (
            10.0**log_amplitude
            * (fvec / self.f0) ** self.mid_slope
            * low_term
            * high_term
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; single column d/d(log_amplitude)."""
        model = self.omega_gw_h2(jnp.asarray(frequency), theta[0])
        return (model * jnp.log(10.0))[..., None]
