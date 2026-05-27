"""
Resonant-feature oscillatory modulation template.

Log-space oscillation following arXiv:1407.4034.  Coefficients are
pre-computed on a grid of omega_log values (stored in data/Resonant_coefficients.npz)
and retrieved via linear interpolators (interpax.interp1d callables, no extrapolation).

References:
  arXiv:1407.4034 (Flauger, Pajer & Paban: resonant features in primordial
                   power spectra)
  arXiv:2407.04356 (GW from inflation in LISA: reconstruction pipeline and
                   physics interpretation)
"""

import os
import numpy as np
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp
from interpax import Interpolator1D

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

# ── Load precomputed coefficient grid ────────────────────────────────────────

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "Resonant_coefficients.npz",
)

with open(_DATA_PATH, "rb") as _f:
    _raw = np.load(_f)
    _omega_grid = jnp.array(_raw["omega"])

    # Build one callable interpolator per coefficient / derivative grid.
    # Data is cached inside each object; extrap=False → NaN outside the grid.
    _resonant_interps: dict = {
        r: Interpolator1D(
            _omega_grid, jnp.array(_raw[r]), method="linear", extrap=False
        )
        for r in (
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
    }


def get_coefficients(omega_resonant: ArrayLike) -> list:
    """
    Retrieve the resonant coefficients for a given omega_resonant.

    Args:
        omega_resonant: Oscillation frequency parameter.
    Returns:
        list [C0, C1, C2, theta1, theta2] of interpolated coefficient values.
    """
    return [
        _resonant_interps[r](omega_resonant)
        for r in ("C0", "C1", "C2", "theta1", "theta2")
    ]

    # ── Template function ─────────────────────────────────────────────────────────


def resonant_feature(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Resonant-feature modulation (arXiv:1407.4034).

    Coefficients C0, C1, C2, theta1, theta2 are interpolated from a
    pre-computed grid as a function of omega_resonant.

    Args:
        freq: Frequency grid [Hz].
        pars: [A_resonant, omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq,).
    """
    A_resonant, omega_resonant, phase_resonant = pars[0], pars[1], pars[2]

    C0, C1, C2, theta1, theta2 = get_coefficients(omega_resonant)

    x = jnp.log(freq)
    arg1 = omega_resonant * x + theta1 + phase_resonant
    arg2 = 2.0 * omega_resonant * x + theta2 + 2.0 * phase_resonant

    denom = 1.0 + A_resonant**2 * C0
    Omega1 = (A_resonant * C1) / denom * jnp.cos(arg1)
    Omega2 = (A_resonant**2 * C2) / denom * jnp.cos(arg2)

    return 1.0 + Omega1 + Omega2 + 1e-30


def d1resonant_feature(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``resonant_feature`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [A_resonant, omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(A_resonant)     — via amplitude-coefficient derivatives
            d/d(omega_resonant) — via precomputed coefficient slopes (C0p … theta2p)
            d/d(phase_resonant)   — phase shift
    """
    A_resonant, omega_resonant, phase_resonant = pars[0], pars[1], pars[2]

    C0, C1, C2, theta1, theta2 = get_coefficients(omega_resonant)

    # Derivatives of coefficients w.r.t. omega_resonant (precomputed in the pkl)
    C0p = _resonant_interps["C0p"](omega_resonant)
    C1p = _resonant_interps["C1p"](omega_resonant)
    C2p = _resonant_interps["C2p"](omega_resonant)
    theta1p = _resonant_interps["theta1p"](omega_resonant)
    theta2p = _resonant_interps["theta2p"](omega_resonant)

    x = jnp.log(freq)
    arg1 = omega_resonant * x + theta1 + phase_resonant
    arg2 = 2.0 * omega_resonant * x + theta2 + 2.0 * phase_resonant
    denom = 1.0 + A_resonant**2 * C0
    denom2 = denom**2

    # d/d(A_resonant)
    amp11 = (1.0 - A_resonant**2 * C0) * C1 / denom2
    amp12 = 2.0 * A_resonant * C2 / denom2
    d_A = amp11 * jnp.cos(arg1) + amp12 * jnp.cos(arg2)

    # d/d(omega_resonant)
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

    # d/d(phase_resonant)
    amp41 = A_resonant * C1 / denom
    amp42 = 2.0 * A_resonant**2 * C2 / denom
    d_phi = -(amp41 * jnp.sin(arg1) + amp42 * jnp.sin(arg2))

    return jnp.stack([d_A, d_omega, d_phi], axis=1)


resonant_feature_model = ut.Signal_model(
    "resonant_feature",
    resonant_feature,
    dtemplate=d1resonant_feature,
    model_label="Resonant Feature",
    parameter_names=["A_resonant", "omega_resonant", "phase_resonant"],
    parameter_labels=[
        r"$A_{\rm r}$",
        r"$\omega_{\rm r}$",
        r"$\phi_{\rm r}$",
    ],
    prior={
        "A_resonant": {"min": 0.0, "max": 1.0},
        "omega_resonant": {"min": 1e-3, "max": 100.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    },
)


# ── Log parametrization ───────────────────────────────────────────────────────


def resonant_feature_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Resonant-feature modulation with log-parametrized amplitude and frequency.

    Identical physics to ``resonant_feature``; uses log10-scaled parameters
    so that wide prior ranges can be sampled efficiently.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 A_resonant, log10 omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_A_resonant, log_omega_resonant, phase_resonant = pars[0], pars[1], pars[2]
    A_resonant = 10.0**log_A_resonant
    omega_resonant = 10.0**log_omega_resonant

    C0, C1, C2, theta1, theta2 = get_coefficients(omega_resonant)

    x = jnp.log(freq)
    arg1 = omega_resonant * x + theta1 + phase_resonant
    arg2 = 2.0 * omega_resonant * x + theta2 + 2.0 * phase_resonant

    denom = 1.0 + A_resonant**2 * C0
    Omega1 = (A_resonant * C1) / denom * jnp.cos(arg1)
    Omega2 = (A_resonant**2 * C2) / denom * jnp.cos(arg2)

    return 1.0 + Omega1 + Omega2 + 1e-30


def d1resonant_feature_log(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``resonant_feature_log`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 A_resonant, log10 omega_resonant, phase_resonant].
    Returns:
        jax.Array of shape (N_freq, 3):
            d/d(log_A_resonant)    = ln(10)*A_resonant  * d/d(A_resonant)
            d/d(log_omega_resonant) = ln(10)*omega_resonant * d/d(omega_resonant)
            d/d(phase_resonant)      = d/d(phase_resonant)
    """
    log_A_resonant, log_omega_resonant, phase_resonant = pars[0], pars[1], pars[2]
    A_resonant = 10.0**log_A_resonant
    omega_resonant = 10.0**log_omega_resonant

    J_lin = d1resonant_feature(
        freq, jnp.array([A_resonant, omega_resonant, phase_resonant])
    )
    ln10 = jnp.log(10.0)
    factors = jnp.array([ln10 * A_resonant, ln10 * omega_resonant, 1.0])
    return J_lin * factors[None, :]


resonant_feature_log_model = ut.Signal_model(
    "resonant_feature_log",
    resonant_feature_log,
    dtemplate=d1resonant_feature_log,
    model_label="Resonant Feature (log params)",
    parameter_names=["log_A_resonant", "log_omega_resonant", "phase_resonant"],
    parameter_labels=[
        r"$\log_{10}A_{\rm r}$",
        r"$\log_{10}\omega_{\rm r}$",
        r"$\phi_{\rm r}$",
    ],
    prior={
        "log_A_resonant": {"min": -3.0, "max": 0.0},
        "log_omega_resonant": {"min": -3.0, "max": 2.0},
        "phase_resonant": {"min": -3.14159, "max": 3.14159},
    },
)
