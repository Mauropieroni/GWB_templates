"""
JAX-compatible helper functions for phase transition GW templates.

All functions are differentiable via ``jax.jacfwd``.
"""

import jax
import jax.numpy as jnp

# Temperature thresholds (GeV), in decreasing order — shared by both g_star tables.
_T_THRESHOLDS = [
    173.3,
    125.6,
    91.2,
    80.4,
    4.19,
    1.777,
    1.29,
    0.16,
    0.1396,
    0.135,
    0.1057,
    0.0008,
    0.000511,
]

# Effective d.o.f. values for each interval;
# last entry is the default (T below all thresholds).
_G_ENERGY = [
    106.75,
    96.25,
    95.25,
    92.25,
    86.25,
    75.75,
    72.25,
    61.75,
    17.25,
    15.25,
    14.25,
    10.75,
    6.863,
    3.363,
]
_G_ENTROPY = [
    106.75,
    96.25,
    95.25,
    92.25,
    86.25,
    75.75,
    72.25,
    61.75,
    17.25,
    15.25,
    14.25,
    10.75,
    7.409,
    3.909,
]


def g_star_energy(T: float | jax.Array) -> jax.Array:
    """
    Effective number of energetic degrees of freedom as a piecewise-constant
    function of temperature T (in GeV).

    Based on the standard model particle content with thresholds at:
    QCD transition (~0.16 GeV), electroweak symmetry breaking, and particle
    mass thresholds.

    Args:
        T: Temperature in GeV.
    Returns:
        Effective number of energetic relativistic degrees of freedom at T.
    """
    return jnp.select(
        [T > t for t in _T_THRESHOLDS], _G_ENERGY[:-1], default=_G_ENERGY[-1]
    )


def g_star_entropy(T: float | jax.Array) -> jax.Array:
    """
    Effective number of entropic degrees of freedom as a piecewise-constant
    function of temperature T (in GeV).

    Args:
        T: Temperature in GeV.
    Returns:
        Effective number of entropic relativistic degrees of freedom at T.
    """
    return jnp.select(
        [T > t for t in _T_THRESHOLDS], _G_ENTROPY[:-1], default=_G_ENTROPY[-1]
    )


def a_hubble(T: float | jax.Array) -> jax.Array:
    """
    Hubble rate at the transition temperature, redshifted to today (in Hz).

    a_H = 1.65e-5 * sqrt(g_e / 100) * (100 / g_s)^(1/3) * T / 100

    Args:
        T: Transition temperature in GeV.

    Returns:
        Redshifted Hubble rate in Hz.
    """
    g_e = g_star_energy(T)
    g_s = g_star_entropy(T)
    return 1.65e-5 * jnp.sqrt(g_e / 100.0) * (100.0 / g_s) ** (1.0 / 3.0) * T / 100.0


def redshift_omega(T: float | jax.Array) -> jax.Array:
    """
    Redshift factor for the gravitational wave energy density.

    h2FGW0 = 1.64e-5 * g_e / 100 * (100 / g_s)^(4/3)

    Args:
        T: Transition temperature in GeV.

    Returns:
        Dimensionless redshift factor h^2 F_GW0.
    """
    g_e = g_star_energy(T)
    g_s = g_star_entropy(T)
    return 1.64e-5 * g_e / 100.0 * (100.0 / g_s) ** (4.0 / 3.0)


def h_star_tau(K: float | jax.Array, R_H_star: float | jax.Array) -> jax.Array:
    """
    Hubble rate times the duration of the sound wave source.

    H_* tau = min(1, R_H_star / sqrt(0.75 * K))

    Based on:
    J. Ellis, M. Lewicki and J.M. No,
    "Gravitational waves from first-order cosmological phase transitions:
    lifetime of the sound wave source",
    JCAP 07 (2020), 050; [arXiv:2003.07360 [hep-ph]].

    Args:
        K: Fluid kinetic energy fraction.
        R_H_star: Bubble size times the Hubble rate.

    Returns:
        Hubble rate times sound wave source duration.
    """
    return jnp.minimum(1.0, R_H_star / jnp.sqrt(0.75 * K))
