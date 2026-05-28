r"""
Resonant-feature oscillatory modulation template (arXiv:1407.4034).

Log-space oscillation. The coefficients ``C0, C1, C2, theta1, theta2`` and
their derivatives are pre-computed on a grid of ``omega_resonant`` values
stored in ``data/Resonant_coefficients.npz`` and retrieved at runtime via
``interpax`` 1-D linear interpolators (no extrapolation).

Two parametrizations are provided:

* :class:`ResonantFeature` — linear ``A_resonant`` / ``omega_resonant``.
* :class:`ResonantFeatureLog` — log10-scaled amplitude and frequency.

The numerical-table dependency on the hot path is why these classes inherit
from :class:`~gwb_templates.template.NumericalTemplate`.

References:
  arXiv:1407.4034 (Flauger, Pajer & Paban)
  arXiv:2407.04356
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np
from interpax import Interpolator1D

from gwb_templates.template import NumericalTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "Resonant_coefficients.npz",
)

_INTERP_KEYS: tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "theta1",
    "theta2",
    "C0p",
    "C1p",
    "C2p",
    "theta1p",
    "theta2p",
)


def _build_resonant_interpolators() -> dict[str, Interpolator1D]:
    """Load the precomputed coefficient table and build linear interpolators."""
    with open(_DATA_PATH, "rb") as fh:
        raw = np.load(fh)
        omega_grid = jnp.array(raw["omega"])
        return {
            key: Interpolator1D(
                omega_grid, jnp.array(raw[key]), method="linear", extrap=False
            )
            for key in _INTERP_KEYS
        }


def _resonant_feature_impl(
    frequency: ArrayLike,
    A_resonant: ArrayLike,
    omega_resonant: ArrayLike,
    phase_resonant: ArrayLike,
    interps: dict[str, Interpolator1D],
) -> jax.Array:
    """Shared closed-form expression for the resonant modulation."""
    C0 = interps["C0"](omega_resonant)
    C1 = interps["C1"](omega_resonant)
    C2 = interps["C2"](omega_resonant)
    theta1 = interps["theta1"](omega_resonant)
    theta2 = interps["theta2"](omega_resonant)

    x = jnp.log(frequency)
    arg1 = omega_resonant * x + theta1 + phase_resonant
    arg2 = 2.0 * omega_resonant * x + theta2 + 2.0 * phase_resonant

    denom = 1.0 + A_resonant**2 * C0
    Omega1 = (A_resonant * C1) / denom * jnp.cos(arg1)
    Omega2 = (A_resonant**2 * C2) / denom * jnp.cos(arg2)

    return 1.0 + Omega1 + Omega2 + 1e-30


class ResonantFeature(NumericalTemplate):
    r"""
    Resonant-feature modulation (linear amplitude / frequency).

    Coefficients :math:`C_0, C_1, C_2, \theta_1, \theta_2` are interpolated
    from a pre-computed grid as a function of ``omega_resonant``.

    Free parameters
    ---------------
    A_resonant
        Oscillation amplitude.
    omega_resonant
        Oscillation frequency in log-space (dimensionless).
    phase_resonant
        Phase offset (radians).
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    # interpax-based interpolators are JAX-traceable so we can keep autodiff.
    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"  # type: ignore[assignment]

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "A_resonant": r"$A_{\rm r}$",
            "omega_resonant": r"$\omega_{\rm r}$",
            "phase_resonant": r"$\phi_{\rm r}$",
        }
        default_priors = {
            "A_resonant": {"min": 0.0, "max": 1.0},
            "omega_resonant": {"min": 1e-3, "max": 100.0},
            "phase_resonant": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Resonant Feature",
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        """Load the precomputed coefficient table and build interpolators."""
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        A_resonant: ArrayLike,
        omega_resonant: ArrayLike,
        phase_resonant: ArrayLike,
    ) -> jax.Array:
        return _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )


class ResonantFeatureLog(NumericalTemplate):
    r"""
    Resonant-feature modulation with log-parametrized amplitude and frequency.
    Identical physics to :class:`ResonantFeature`.

    Free parameters
    ---------------
    log_A_resonant
        :math:`\log_{10}` amplitude.
    log_omega_resonant
        :math:`\log_{10}` oscillation frequency.
    phase_resonant
        Phase offset (radians).
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"  # type: ignore[assignment]

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_A_resonant": r"$\log_{10}A_{\rm r}$",
            "log_omega_resonant": r"$\log_{10}\omega_{\rm r}$",
            "phase_resonant": r"$\phi_{\rm r}$",
        }
        default_priors = {
            "log_A_resonant": {"min": -3.0, "max": 0.0},
            "log_omega_resonant": {"min": -3.0, "max": 2.0},
            "phase_resonant": {"min": -3.14159, "max": 3.14159},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Resonant Feature (log params)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        """Load the precomputed coefficient table and build interpolators."""
        self._interps: dict[str, Interpolator1D] = _build_resonant_interpolators()

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_A_resonant: ArrayLike,
        log_omega_resonant: ArrayLike,
        phase_resonant: ArrayLike,
    ) -> jax.Array:
        A_resonant = 10.0**log_A_resonant
        omega_resonant = 10.0**log_omega_resonant
        return _resonant_feature_impl(
            frequency, A_resonant, omega_resonant, phase_resonant, self._interps
        )
