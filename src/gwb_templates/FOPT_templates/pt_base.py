"""
JAX-compatible helper functions for phase transition GW templates.

Pure-function helpers shared by the FOPT template classes:

* Effective relativistic d.o.f. tables (``g_star_energy``,
  ``g_star_entropy``) and the cosmological redshift factors derived from
  them (``a_hubble``, ``redshift_omega``).
* The sound-wave source-duration helper ``h_star_tau``.
* Inline spectral shapes (``double_broken_power_law``,
  ``broken_power_law_a1``) that the PT templates compose with their
  physics-level prefactors. These are kept here rather than imported from
  ``generic_templates`` so the PT classes remain robust to refactors of
  the generic-template layer.

All functions are pure JAX and differentiable via ``jax.jacfwd`` /
``jax.grad``.
"""

import jax
import jax.numpy as jnp

# Temperature thresholds (GeV), in decreasing order — shared by both g_star tables.
# Cf. Tab. 2 of Husdal, Galaxies 4 (2016), 4, 78 [arXiv:1609.04979 [astro-ph.CO]].
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
# Cf. Tab. 2 of Husdal, Galaxies 4 (2016), 4, 78 [arXiv:1609.04979 [astro-ph.CO]].
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

    Based on Tab. 2 of
    L. Husdal, "On Effective Degrees of Freedom in the Early Universe",
    Galaxies 4 (2016) 4, 78; [arXiv:1609.04979 [astro-ph.CO]].

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

    Based on Tab. 2 of
    L. Husdal, "On Effective Degrees of Freedom in the Early Universe",
    Galaxies 4 (2016) 4, 78; [arXiv:1609.04979 [astro-ph.CO]].

    Args:
        T: Temperature in GeV.
    Returns:
        Effective number of entropic relativistic degrees of freedom at T.
    """
    return jnp.select(
        [T > t for t in _T_THRESHOLDS], _G_ENTROPY[:-1], default=_G_ENTROPY[-1]
    )


def a_hubble(T: float | jax.Array) -> float | jax.Array:
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


def double_broken_power_law(
    freq: jax.Array,
    log_amplitude: jax.Array,
    log_f_1: jax.Array,
    log_f_2: jax.Array,
    n_1: float | jax.Array,
    n_2: float | jax.Array,
    n_3: float | jax.Array,
    a_1: float | jax.Array,
    a_2: float | jax.Array,
) -> jax.Array:
    r"""
    Double broken power-law spectrum, normalised so that
    ``Omega(f_2) = 10**log_amplitude``.

    .. math::

        \Omega(f) = N\,A\,(f/f_1)^{n_1}\,
            \bigl(1 + (f/f_1)^{a_1}\bigr)^{(n_2 - n_1)/a_1}\,
            \bigl(1 + (f/f_2)^{a_2}\bigr)^{(n_3 - n_2)/a_2}

    with the normalisation

    .. math::

        N = (f_1/f_2)^{n_1}\,
            \bigl(1 + (f_1/f_2)^{-a_1}\bigr)^{(n_1 - n_2)/a_1}\,
            2^{(n_2 - n_3)/a_2}.

    Args:
        freq: Frequency grid (Hz).
        log_amplitude: log10 amplitude at f_2.
        log_f_1, log_f_2: log10 of the two break frequencies (Hz).
        n_1, n_2, n_3: Spectral indices in the three regimes.
        a_1, a_2: Transition smoothness parameters.
    """
    amplitude = 10.0**log_amplitude
    ratio = 10.0 ** (log_f_1 - log_f_2)  # f_1 / f_2
    x_1 = freq / 10.0**log_f_1
    x_2 = freq / 10.0**log_f_2

    norm = (
        ratio**n_1
        * (1.0 + ratio ** (-a_1)) ** ((n_1 - n_2) / a_1)
        * 2.0 ** ((n_2 - n_3) / a_2)
    )

    return (
        norm
        * amplitude
        * x_1**n_1
        * (1.0 + x_1**a_1) ** ((n_2 - n_1) / a_1)
        * (1.0 + x_2**a_2) ** ((n_3 - n_2) / a_2)
    )


def broken_power_law_a1(
    freq: jax.Array,
    log_amplitude: jax.Array,
    log_f_b: jax.Array,
    n_1: float | jax.Array,
    n_2: float | jax.Array,
    a_1: float | jax.Array,
) -> jax.Array:
    r"""
    Broken power law with direct smoothness parameter ``a_1``.

    .. math::

        \Omega(f) = 10^{\alpha}\,x^{n_1}\,
            \bigl(\tfrac{1}{2} + \tfrac{1}{2} x^{a_1}\bigr)^{(n_2 - n_1)/a_1}

    with :math:`x = f / f_b`.
    """
    x = freq / 10.0**log_f_b
    return 10.0**log_amplitude * x**n_1 * (0.5 + 0.5 * x**a_1) ** ((n_2 - n_1) / a_1)


def jac_double_broken_power_law_amp_freqs(
    freq: jax.Array,
    log_amplitude: jax.Array,
    log_f_1: jax.Array,
    log_f_2: jax.Array,
    n_1: float | jax.Array,
    n_2: float | jax.Array,
    n_3: float | jax.Array,
    a_1: float | jax.Array,
    a_2: float | jax.Array,
) -> jax.Array:
    """
    Analytic partials of :func:`double_broken_power_law` w.r.t. the three
    "outer" arguments ``(log_amplitude, log_f_1, log_f_2)``.

    Returns:
        jax.Array of shape ``freq.shape + (3,)``.
    """
    x_1 = freq / 10.0**log_f_1
    x_2 = freq / 10.0**log_f_2
    r_12 = 10.0 ** (log_f_1 - log_f_2)
    ln10 = jnp.log(10.0)
    model = double_broken_power_law(
        freq, log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2
    )

    x1a1 = x_1**a_1
    x2a1 = x_2**a_1
    x2a2 = x_2**a_2
    r12a1 = r_12**a_1

    d_logA = model * ln10
    d_logf1 = model * ln10 * (n_1 - n_2) * (x2a1 - 1.0) / ((1.0 + r12a1) * (1.0 + x1a1))
    num_f2 = (
        -n_1 * r12a1 - (n_3 + (n_1 + n_3) * r12a1) * x2a2 + n_2 * (r12a1 * x2a2 - 1.0)
    )
    d_logf2 = model * ln10 * num_f2 / ((1.0 + r12a1) * (1.0 + x2a2))

    return jnp.stack([d_logA, d_logf1, d_logf2], axis=-1)


def jac_broken_power_law_a1_amp_freq(
    freq: jax.Array,
    log_amplitude: jax.Array,
    log_f_b: jax.Array,
    n_1: float | jax.Array,
    n_2: float | jax.Array,
    a_1: float | jax.Array,
) -> jax.Array:
    """
    Analytic partials of :func:`broken_power_law_a1` w.r.t. the two "outer"
    arguments ``(log_amplitude, log_f_b)``.

    Returns:
        jax.Array of shape ``freq.shape + (2,)``.
    """
    x = freq / 10.0**log_f_b
    xa = x**a_1
    ln10 = jnp.log(10.0)
    model = broken_power_law_a1(freq, log_amplitude, log_f_b, n_1, n_2, a_1)

    d_logA = model * ln10
    d_logfb = model * ln10 * (-n_1 - n_2 * xa) / (1.0 + xa)

    return jnp.stack([d_logA, d_logfb], axis=-1)
