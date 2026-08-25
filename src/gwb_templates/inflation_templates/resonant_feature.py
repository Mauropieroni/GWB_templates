r"""
Resonant-feature oscillatory modulation template (arXiv:1002.0833).

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
  arXiv:1002.0833 (Flauger & Pajer — resonant non-Gaussianity)
  arXiv:0907.2916 (Flauger, McAllister, Pajer, Westphal & Xu — original
  resonant oscillatory power-spectrum template from axion monodromy)
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline
  and physics interpretation)
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


def _resonant_feature_grad_lin(
    frequency: ArrayLike,
    A_resonant: ArrayLike,
    omega_resonant: ArrayLike,
    phase_resonant: ArrayLike,
    interps: dict[str, Interpolator1D],
) -> jax.Array:
    """Analytic Jacobian of the linear resonant-feature w.r.t. (A, omega, phase)."""
    C0 = interps["C0"](omega_resonant)
    C1 = interps["C1"](omega_resonant)
    C2 = interps["C2"](omega_resonant)
    theta1 = interps["theta1"](omega_resonant)
    theta2 = interps["theta2"](omega_resonant)
    C0p = interps["C0p"](omega_resonant)
    C1p = interps["C1p"](omega_resonant)
    C2p = interps["C2p"](omega_resonant)
    theta1p = interps["theta1p"](omega_resonant)
    theta2p = interps["theta2p"](omega_resonant)

    x = jnp.log(jnp.asarray(frequency))
    arg1 = omega_resonant * x + theta1 + phase_resonant
    arg2 = 2.0 * omega_resonant * x + theta2 + 2.0 * phase_resonant
    denom = 1.0 + A_resonant**2 * C0
    denom2 = denom**2

    amp11 = (1.0 - A_resonant**2 * C0) * C1 / denom2
    amp12 = 2.0 * A_resonant * C2 / denom2
    d_A = amp11 * jnp.cos(arg1) + amp12 * jnp.cos(arg2)

    amp21 = (
        A_resonant
        * (C1p + A_resonant**2 * C0 * C1p - A_resonant**2 * C0p * C1)
        / denom2
    )
    amp22 = (
        A_resonant**2
        * (C2p + A_resonant**2 * C0 * C2p - A_resonant**2 * C0p * C2)
        / denom2
    )
    amp23 = -A_resonant * C1 / denom * (x + theta1p)
    amp24 = -(A_resonant**2) * C2 / denom * (2.0 * x + theta2p)
    d_omega = (
        amp21 * jnp.cos(arg1)
        + amp22 * jnp.cos(arg2)
        + amp23 * jnp.sin(arg1)
        + amp24 * jnp.sin(arg2)
    )

    amp41 = A_resonant * C1 / denom
    amp42 = 2.0 * A_resonant**2 * C2 / denom
    d_phi = -(amp41 * jnp.sin(arg1) + amp42 * jnp.sin(arg2))

    return jnp.stack([d_A, d_omega, d_phi], axis=-1)


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

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Flauger:2010ja,
    author = "Flauger, Raphael and Pajer, Enrico",
    title = "{Resonant Non-Gaussianity}",
    eprint = "1002.0833",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1088/1475-7516/2011/01/017",
    journal = "JCAP",
    volume = "01",
    pages = "017",
    year = "2011"
}
""",
        r"""
@article{LISACosmologyWorkingGroup:2024hsc,
    author = "Braglia, Matteo and others",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from inflation in LISA: reconstruction pipeline and
        physics interpretation}",
    eprint = "2407.04356",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-03, CERN-TH-2024-072",
    doi = "10.1088/1475-7516/2024/11/032",
    journal = "JCAP",
    volume = "11",
    pages = "032",
    year = "2024"
}
""",
        r"""
@article{Flauger:2009ab,
    author = "Flauger, Raphael and McAllister, Liam and Pajer, Enrico and Westphal,
        Alexander and Xu, Gang",
    title = "{Oscillations in the CMB from Axion Monodromy Inflation}",
    eprint = "0907.2916",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "SLAC-PUB-14821",
    doi = "10.1088/1475-7516/2010/06/009",
    journal = "JCAP",
    volume = "06",
    pages = "009",
    year = "2010"
}
""",
    )

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

    # NOTE: An analytic gradient using the precomputed C0p/C1p/... slope
    # tables is provided as `_resonant_feature_grad_lin` for callers who want
    # to use it directly. We do NOT install it as the
    # `_grad_theta_omega_gw_h2_analytical` override because those tables are
    # not bit-identical to interpax's autodiff slope of the value tables, and
    # the registered gradient test compares to autodiff at places=15. The
    # autodiff path through the interpax interpolators is fully traceable.


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

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Flauger:2010ja,
    author = "Flauger, Raphael and Pajer, Enrico",
    title = "{Resonant Non-Gaussianity}",
    eprint = "1002.0833",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1088/1475-7516/2011/01/017",
    journal = "JCAP",
    volume = "01",
    pages = "017",
    year = "2011"
}
""",
        r"""
@article{LISACosmologyWorkingGroup:2024hsc,
    author = "Braglia, Matteo and others",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from inflation in LISA: reconstruction pipeline and
        physics interpretation}",
    eprint = "2407.04356",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-03, CERN-TH-2024-072",
    doi = "10.1088/1475-7516/2024/11/032",
    journal = "JCAP",
    volume = "11",
    pages = "032",
    year = "2024"
}
""",
        r"""
@article{Flauger:2009ab,
    author = "Flauger, Raphael and McAllister, Liam and Pajer, Enrico and Westphal,
        Alexander and Xu, Gang",
    title = "{Oscillations in the CMB from Axion Monodromy Inflation}",
    eprint = "0907.2916",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "SLAC-PUB-14821",
    doi = "10.1088/1475-7516/2010/06/009",
    journal = "JCAP",
    volume = "06",
    pages = "009",
    year = "2010"
}
""",
    )

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

    # NOTE: see ResonantFeature — autodiff is preserved as the default
    # backend because the precomputed coefficient-slope tables differ from
    # interpax's autodiff slope of the value tables.
