"""
Cosmic String Model I template (3 parameters).

Computes h^2 * Omega_GW(f) for a network of Nambu-Goto cosmic strings
following the analytic approach of arXiv:2405.03740.

Because scipy.special.hyp2f1 has no JAX counterpart, the computation is
performed in NumPy/SciPy.  A log-log spline is built on a coarse grid and
then evaluated on the full frequency array so that the overall cost stays
low while preserving accuracy.

Reference: arXiv:2405.03740 (GW from cosmic strings in LISA: reconstruction
pipeline and physics interpretation).
"""

from collections.abc import Callable, Sequence

import jax
import numpy as np
import jax.numpy as jnp
import scipy.special as sc

from gwb_templates import constants as ct
from gwb_templates import utils as ut

jax.config.update("jax_enable_x64", True)

# ── Model constants ───────────────────────────────────────────────────────────

_Gamma = 50.0
_mathcal_F = 0.1  # normalization factor (set to 0.1 for Model II)
# String scaling parameters (VOS model calibrated with simulations)
_c_tilde = 0.23
_xi_r = 0.2711082721253275
_xi_m = 0.6253627
_v_r = 0.6620623102615658
_v_m = 0.5825118
_nu_r = 0.5
_nu_m = 2.0 / 3.0

# Fit to SM degrees of freedom (end of p. 51 of arXiv:2405.03740)
_Delta_gr = np.array([0.4065, 0.4853, 0.7085, 1.0])
# Scale factors corresponding to the changes in degrees of freedom
_a0 = 1.0
_a_star_factor = np.array([5.8897e16, 7.2182e19, 7.5617e21])


# ── Helper functions ──────────────────────────────────────────────────────────


def _hyperharmonic(r: float, N: float | np.ndarray) -> float | np.ndarray:
    """
    Euler-Maclaurin approximation to the generalized harmonic sum.

    Implements Eq. B.2 of arXiv:2405.03740.

    Args:
        r: Power-law exponent of the harmonic series.
        N: Truncation order (scalar or array).
    Returns:
        Approximate value of sum_{n=1}^{N} n^{-r} at each N.
    """
    zeta_r = sc.zeta(r) if r >= 1.0 else sc.zetac(r) + 1.0
    return zeta_r + (N / (1.0 - r) + 0.5 - r / (12.0 * N)) / N**r


def _get_epsilon_r(Gmu: float, alpha: float) -> float:
    """
    Radiation-era loop size ratio eps_r (Eq. A.4 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        eps_r = alpha / (Gamma * Gmu).
    """
    # See Eq. A.4 of arXiv:2405.03740
    return alpha / _Gamma / Gmu


def _get_epsilon_m(Gmu: float, alpha: float) -> float:
    """
    Matter-era loop size ratio eps_m (Eq. A.4 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        eps_m = eps_r * xi_m / xi_r.
    """
    # See Eq. A.4 of arXiv:2405.03740
    return _get_epsilon_r(Gmu, alpha) * _xi_m / _xi_r


def _get_gamma_m(Gmu: float, alpha: float) -> float:
    """
    Matter-era loop-size enhancement factor gamma_m (Eq. A.4 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        gamma_m = 1 + 1 / eps_m.
    """
    # See Eq. A.4 of arXiv:2405.03740
    return 1.0 + 1.0 / _get_epsilon_m(Gmu, alpha)


def _get_beta_m(Gmu: float, alpha: float) -> float:
    """
    Matter-era shape factor beta_m (Eq. A.4 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        beta_m = (1 + 1/gamma_m) / eps_m.
    """
    # See Eq. A.4 of arXiv:2405.03740
    eps_m = _get_epsilon_m(Gmu, alpha)
    return (1.0 + 1.0 / _get_gamma_m(Gmu, alpha)) / eps_m


def _get_D(nu_i: float, Omega_i: float, Gmu: float) -> float:
    """
    Characteristic frequency scale D_i (Eq. A.4 of arXiv:2405.03740).

    Args:
        nu_i: Loop velocity scaling exponent for epoch i.
        Omega_i: Cosmological energy density parameter for epoch i.
        Gmu: String tension G*mu.
    Returns:
        D_i in Planck natural units.
    """
    # See Eq. A.4 of arXiv:2405.03740
    return 2.0 * ct.H0_eV * np.sqrt(Omega_i) / nu_i / _Gamma / Gmu


def _get_f_min_r(Gmu: float, alpha: float) -> float:
    """
    Min. frequency from radiation-era loops (after Eq. A.5 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        f_min_r in Planck natural units.
    """
    # Min freq from radiation-era loops; defined after Eq. A.5 of arXiv:2405.03740
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    return D_r * (ct.Omega_M / ct.Omega_R) / _get_epsilon_r(Gmu, alpha)


def _get_f_min_m(Gmu: float, alpha: float) -> float:
    """
    Min. frequency from matter-era loops (after Eq. A.5 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        f_min_m in Planck natural units.
    """
    # Min freq from matter-era loops; defined after Eq. A.5 of arXiv:2405.03740
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    return D_m / _get_epsilon_m(Gmu, alpha)


def _get_a0_star(Gmu: float, alpha: float) -> float:
    """
    Scale factor a0_* for DOF summation (after Eq. A.17 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        a0_star (dimensionless scale factor).
    """
    # a0_star as defined after Eq. A.17 of arXiv:2405.03740
    prefac = np.sqrt(2.0 * np.sqrt(ct.Omega_R) * ct.H0_SI * ct.t_planck_SI)
    return _a0 * prefac * np.sqrt(1.0 + _get_epsilon_r(Gmu, alpha)) / Gmu


def _get_an_star(Gmu: float, alpha: float, epoch: int) -> float:
    """
    Scale factor a_n* at each DOF step (after Eq. A.19 of arXiv:2405.03740).

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        epoch: DOF step index (0–4; 0 = no step, 4 = matter-radiation equality).
    Returns:
        a_n* (dimensionless scale factor for the given DOF step).
    """
    # a_n* defined after Eq. A.19 of arXiv:2405.03740 (p. 45)
    a0_star = _get_a0_star(Gmu, alpha)
    if epoch == 0:
        return a0_star
    if epoch < 4:
        return _a_star_factor[epoch - 1] * Gmu * a0_star
    if epoch == 4:
        return ct.a_eq * _a0
    raise ValueError(f"Invalid epoch {epoch}")


def _get_A_n(f_eV: np.ndarray, Gmu: float, alpha: float, epoch: int) -> np.ndarray:
    """
    Integration bound Script-A_n (after Eq. A.19 of arXiv:2405.03740).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        epoch: DOF step index (0–4).
    Returns:
        np.ndarray of Script-A_n values at each frequency.
    """
    # Script-A_n defined after Eq. A.19 of arXiv:2405.03740
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    a_star_n = _get_an_star(Gmu, alpha, epoch) / _a0
    return (D_r / a_star_n) / f_eV


def _get_tilde_A_rm(f_eV: np.ndarray, Gmu: float) -> np.ndarray:
    """
    Rad-to-matter transition bound tilde-A_rm (Eq. A.4 of arXiv:2405.03740).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
    Returns:
        np.ndarray of tilde-A_rm values at each frequency.
    """
    # See Eq. A.4 of arXiv:2405.03740
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    return (D_m * np.sqrt(ct.Omega_M / ct.Omega_R)) / f_eV


def _get_tilde_A_m(f_eV: np.ndarray, Gmu: float) -> np.ndarray:
    """
    Matter-era integration bound tilde-A_m (Eq. A.4 of arXiv:2405.03740).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
    Returns:
        np.ndarray of tilde-A_m values at each frequency.
    """
    # See Eq. A.4 of arXiv:2405.03740
    return _get_D(_nu_m, ct.Omega_M, Gmu) / f_eV


def _get_C_i(xi_i: float, v_i: float) -> float:
    """
    Spectral amplitude coefficient C_i (Eq. A.4 of arXiv:2405.03740).

    Args:
        xi_i: Loop correlation length for epoch i.
        v_i: Loop velocity for epoch i.
    Returns:
        C_i = c_tilde * F * v_i / (xi_i^3 * sqrt(2)).
    """
    # See Eq. A.4 of arXiv:2405.03740
    return _c_tilde * _mathcal_F * v_i / xi_i**3 / np.sqrt(2.0)


def _get_C_r_no_dof(Gmu: float, alpha: float) -> float:
    """
    Radiation-era amplitude prefactor C_r without DOF changes.

    Implements Eq. A.4 of arXiv:2405.03740.

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        C_r coefficient (without DOF ratio corrections).
    """
    # See Eq. A.4 of arXiv:2405.03740
    eps_r = _get_epsilon_r(Gmu, alpha)
    C_r = _get_C_i(_xi_r, _v_r)
    return 128.0 / 9.0 * np.pi * C_r * ct.Omega_R * (1.0 + eps_r) ** 1.5 / eps_r * Gmu


def _get_C_rm(f_eV: np.ndarray, Gmu: float, alpha: float) -> np.ndarray:
    """
    Rad-to-matter transition amplitude C_rm (Eq. A.4 of arXiv:2405.03740).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        np.ndarray of C_rm values at each frequency.
    """
    # See Eq. A.4 of arXiv:2405.03740
    eps_r = _get_epsilon_r(Gmu, alpha)
    return (
        32.0
        * np.sqrt(3.0)
        * np.pi
        * _get_C_i(_xi_r, _v_r)
        * (1.0 + eps_r) ** 1.5
        / eps_r
        * ct.H0_eV
        * (ct.Omega_M * ct.Omega_R) ** 0.75
        / _Gamma
    ) / f_eV


def _get_C_m(f_eV: np.ndarray, Gmu: float, alpha: float) -> np.ndarray:
    """
    Matter-era amplitude coefficient C_m (Eq. A.4 of arXiv:2405.03740).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
    Returns:
        np.ndarray of C_m values at each frequency.
    """
    # See Eq. A.4 of arXiv:2405.03740
    eps_m = _get_epsilon_m(Gmu, alpha)
    return (
        162.0
        * np.pi
        * _get_C_i(_xi_m, _v_m)
        * (1.0 + eps_m)
        / eps_m
        * (ct.H0_eV * ct.Omega_M / _Gamma) ** 2.0
        / Gmu
    ) / f_eV**2.0


# ── Euler–Maclaurin summation pieces (Eqs. A.7–A.14) ────────────────────────


def _I(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Integral I(a, b, A, N); Eq. A.7 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Power-law exponent for the (1 + A*n) denominator.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of I values at each element of A (and N if array).
    """
    # Eq. A.7 of arXiv:2405.03740
    return N ** (a + 1.0) / (a + 1.0) * sc.hyp2f1(a + 1.0, b, a + 2.0, -A * N)


def _D1(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    First-order Euler-Maclaurin correction D1; Eq. A.8 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Power-law exponent for the (1 + A*n) denominator.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of D1 values.
    """
    # Eq. A.8 of arXiv:2405.03740
    AN = A * N
    return N ** (a - 1.0) * (a + AN * (a - b)) / (1.0 + AN) ** (1.0 + b)


def _D3(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Third-order Euler-Maclaurin correction D3; Eq. A.9 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Power-law exponent for the (1 + A*n) denominator.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of D3 values.
    """
    # Eq. A.9 of arXiv:2405.03740
    AN1 = 1.0 + A * N
    pref = N**a / AN1**b
    t1 = a * (1.0 - a) * (2.0 - a) / N**3.0
    t2 = -(b * (1.0 + b) * (2.0 + b)) * (A / AN1) ** 3.0
    t3 = 3.0 * a * b * (1.0 - a + A * N * (2.0 - a + b)) * (A / (N * AN1) ** 2.0)
    return pref * (t1 + t2 + t3)


def _delta(
    fn: Callable, a: float, b: float | np.ndarray, A: np.ndarray, N: float | np.ndarray
) -> np.ndarray:
    """
    Difference fn(a,b,A,N) - fn(a,b,A,1); after Eq. A.10 of arXiv:2405.03740.

    Args:
        fn: One of the E-M integrand helpers (_I, _D1, _D3, _Il, _Dl1, _Dl3).
        a: Power-law exponent for the N dependence.
        b: Power-law exponent for the (1 + A*n) denominator.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of fn(a,b,A,N) - fn(a,b,A,1) at each frequency.
    """
    # Delta_F = F(a,b,A,N) - F(a,b,A,1); after Eq. A.10 of arXiv:2405.03740
    return fn(a, b, A, N) - fn(a, b, A, 1.0)


def _M(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Euler-Maclaurin sum M(a, b, A, N); Eq. A.10 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Power-law exponent for the (1 + A*n) denominator.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of M values at each frequency.
    """
    # Eq. A.10 of arXiv:2405.03740
    return (
        _delta(_I, a, b, A, N)
        + 0.5 / (1.0 + A) ** b
        + 0.5 * N**a / (1.0 + A * N) ** b
        + _delta(_D1, a, b, A, N) / 12.0
        - _delta(_D3, a, b, A, N) / 720.0
    )


def _Il(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Log-weighted integral Il(a, b, A, N); Eq. A.11 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Log argument coefficient.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of Il values at each frequency.
    """
    # Eq. A.11 of arXiv:2405.03740
    return (N ** (1.0 + a) * np.log(b * N / (1.0 + A * N)) - _I(a, 1.0, A, N)) / (
        1.0 + a
    )


def _Dl1(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Log-correction Dl1 (first-order E-M); Eq. A.12 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Log argument coefficient.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of Dl1 values.
    """
    # Eq. A.12 of arXiv:2405.03740
    inv = 1.0 / (1.0 + A * N)
    return N ** (a - 1.0) * (inv + a * np.log(b * N * inv))


def _Dl3(a: float, b: float, A: np.ndarray, N: float | np.ndarray) -> np.ndarray:
    """
    Log-correction Dl3 (third-order E-M); Eq. A.13 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Log argument coefficient.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of Dl3 values.
    """
    # Eq. A.13 of arXiv:2405.03740
    inv = 1.0 / (1.0 + A * N)
    pref = N ** (a - 3.0)
    t1 = 2.0 * inv**3.0
    t2 = 3.0 * (a - 2.0) * inv**2.0
    t3 = 3.0 * (2.0 - a) * (1.0 - a) * inv
    t4 = (2.0 - a) * (1.0 - a) * a * np.log(b * N * inv)
    return pref * (t1 + t2 + t3 + t4)


def _N_func(
    a: float, b: float | np.ndarray, A: np.ndarray, N: float | np.ndarray
) -> np.ndarray:
    """
    Log-weighted Euler-Maclaurin sum N(a, b, A, N); Eq. A.14 of arXiv:2405.03740.

    Args:
        a: Power-law exponent for the N dependence.
        b: Log argument coefficient.
        A: Integration bound array.
        N: Upper truncation limit (scalar or array).
    Returns:
        np.ndarray of N values at each frequency.
    """
    # Eq. A.14 of arXiv:2405.03740
    return (
        _delta(_Il, a, b, A, N)
        + 0.5 * np.log(b / (1.0 + A))
        + 0.5 * N**a * np.log(b * N / (1.0 + A * N))
        + _delta(_Dl1, a, b, A, N) / 12.0
        - _delta(_Dl3, a, b, A, N) / 720.0
    )


def _M_delta(
    q: float, A_next: np.ndarray, A_now: np.ndarray, N: float | np.ndarray
) -> np.ndarray:
    """
    M_delta piece for the DOF-step summation; Eq. A.19 of arXiv:2405.03740.

    Args:
        q: Harmonic power-law index.
        A_next: Upper integration bound for the next DOF step.
        A_now: Upper integration bound for the current DOF step.
        N: Euler-Maclaurin truncation order (scalar or array).
    Returns:
        np.ndarray of M(-q, 1.5, A_next, N) - M(-q, 1.5, A_now, N).
    """
    # Eq. A.19 of arXiv:2405.03740 (without the prefactor)
    return _M(q, 1.5, A_next, N) - _M(q, 1.5, A_now, N)


# ── Omega contributions ───────────────────────────────────────────────────────


def _Omega_r_dof(
    f_eV: np.ndarray, Gmu: float, alpha: float, q: float, N: np.ndarray
) -> np.ndarray:
    """
    GW energy density from radiation-era loops with SM DOF changes.

    Implements Eq. A.18 of arXiv:2405.03740.

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        q: Harmonic power-law index.
        N: Euler-Maclaurin truncation order (array, one per frequency bin).
    Returns:
        h^2 * Omega_GW contribution from radiation-era loops.
    """
    # Omega_r from Eq. A.18 of arXiv:2405.03740 (radiation era with SM DOF changes)
    Cr = _get_C_r_no_dof(Gmu, alpha) / sc.zeta(q)
    A_all = [_get_A_n(f_eV, Gmu, alpha, i) for i in range(len(_Delta_gr) + 1)]
    total = 0.0
    for i, dg in enumerate(_Delta_gr):
        A_now = np.sqrt(dg) * A_all[i]
        A_next = np.sqrt(dg) * A_all[i + 1]
        total += dg * _M_delta(-q, A_next, A_now, N)
    return Cr * total


def _Omega_rm(
    f_eV: np.ndarray, Gmu: float, alpha: float, q: float, N: np.ndarray
) -> np.ndarray:
    """
    GW energy density from radiation-to-matter transition loops.

    Implements Eq. A.15 of arXiv:2405.03740.

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        q: Harmonic power-law index.
        N: Euler-Maclaurin truncation order.
    Returns:
        h^2 * Omega_GW contribution from rad-to-matter transition loops.
    """
    # Omega_{r→m} from Eq. A.15 of arXiv:2405.03740 (radiation-to-matter transition)
    pref = _get_C_rm(f_eV, Gmu, alpha) / sc.zeta(q)
    A_rm = _get_tilde_A_rm(f_eV, Gmu)
    A_m = _get_tilde_A_m(f_eV, Gmu)
    r = ct.Omega_R / ct.Omega_M
    t1 = 2.0 * _M(1.0 - q, 0.5, A_rm, N)
    t2 = _M(1.0 - q, 1.5, A_rm, N)
    t3 = -2.0 * _M(1.0 - q, 0.5, A_m, N)
    t4 = -_M(1.0 - q, 1.5, A_m, N)
    return pref * ((t1 + t2) / r**0.25 + t3 + t4)


def _Omega_m(
    f_eV: np.ndarray, Gmu: float, alpha: float, q: float, N: np.ndarray
) -> np.ndarray:
    """
    GW energy density from matter-dominated loops.

    Implements Eq. A.15 of arXiv:2405.03740.

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        q: Harmonic power-law index.
        N: Euler-Maclaurin truncation order.
    Returns:
        h^2 * Omega_GW contribution from matter-era loops.
    """
    # Omega_m from Eq. A.15 of arXiv:2405.03740 (matter-dominated loops)
    pref = _get_C_m(f_eV, Gmu, alpha) / sc.zeta(q)
    A_m = _get_tilde_A_m(f_eV, Gmu)
    gamma_m = _get_gamma_m(Gmu, alpha)
    beta_m = _get_beta_m(Gmu, alpha)
    t1 = _M(2.0 - q, 1.0, A_m, N)
    t2 = _hyperharmonic(q - 1.0, N) / A_m
    t3 = -beta_m * _hyperharmonic(q - 2.0, N)
    t4 = 2.0 * _N_func(2.0 - q, gamma_m * A_m, A_m, N)
    return pref * (t1 + t2 + t3 + t4)


def _Omega_small_alpha(
    f_eV: np.ndarray, Gmu: float, alpha: float, q: float, N_m: np.ndarray
) -> np.ndarray:
    """
    Analytic GW energy density in the small-alpha limit.

    Uses the simplified formula after Eq. A.16 of arXiv:2405.03740,
    valid when alpha << Gamma * Gmu * xi_r (no DOF changes or UV cutoff).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter (must satisfy alpha << Gamma*Gmu*xi_r).
        q: Harmonic power-law index.
        N_m: Euler-Maclaurin truncation order for matter-era loops.
    Returns:
        h^2 * Omega_GW in the small-alpha approximation.
    """
    # Analytic Omega_GW from the equation after A.16 of arXiv:2405.03740
    # (small-alpha limit; no DOF changes or high-frequency cutoff)
    t1 = 64.0 * np.pi * Gmu * ct.Omega_R * _get_C_i(_xi_r, _v_r) / 3.0
    pref2 = (
        54.0
        * np.pi
        * ct.H0_eV
        * ct.Omega_M**1.5
        * _get_C_i(_xi_m, _v_m)
        / _get_epsilon_m(Gmu, alpha)
        / sc.zeta(q)
        / _Gamma
        / f_eV
    )
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    eps_m = _get_epsilon_m(Gmu, alpha)
    t2 = _hyperharmonic(q - 1.0, N_m)
    t3 = D_m * _hyperharmonic(q - 2.0, N_m) / eps_m / f_eV
    return t1 + pref2 * (t2 - t3)


# ── Full spectrum + log-interpolated wrapper ──────────────────────────────────


def _compute_spectrum(
    freq: np.ndarray, log_Gmu: float, log_alpha: float, q: float
) -> np.ndarray:
    """
    Evaluate h^2 * Omega_GW(freq) for Model I without log-interpolation.

    Args:
        freq: Frequency array [Hz].
        log_Gmu: log10 of the string tension.
        log_alpha: log10 of the loop-size parameter.
        q: Harmonic power-law index.
    Returns:
        np.ndarray of h^2 * Omega_GW values at each frequency.
    """
    f_eV = freq * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    Omega = np.zeros(len(freq))

    if alpha > _Gamma * Gmu * _xi_r:
        mask_r = f_eV >= f_min_r
        if mask_r.any():
            fn_r = f_eV[mask_r]
            N_r = np.floor(fn_r / f_min_r)
            Omega[mask_r] += _Omega_r_dof(fn_r, Gmu, alpha, q, N_r)

        mask_m = f_eV >= f_min_m
        if mask_m.any():
            fn_m = f_eV[mask_m]
            N_m = np.floor(fn_m / f_min_m)
            Omega[mask_m] += _Omega_rm(fn_m, Gmu, alpha, q, N_m)
            if alpha < 1e-3:
                Omega[mask_m] += _Omega_m(fn_m, Gmu, alpha, q, N_m)
    else:
        mask_m = f_eV > f_min_m
        if mask_m.any():
            fn_m = f_eV[mask_m]
            N_m = np.floor(fn_m / f_min_m)
            Omega[mask_m] += _Omega_small_alpha(fn_m, Gmu, alpha, q, N_m)

    return Omega * ct.h**2


# ── Public template function ──────────────────────────────────────────────────


def cosmic_string_model_i(freq: jnp.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Cosmic String Model I (arXiv:2405.03740).

    Computes h^2 * Omega_GW(f) for a Nambu-Goto cosmic string network using
    NumPy/SciPy (no JAX AD support; gradients are not available).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu, log10 alpha, q].
    Returns:
        jax.Array of h^2 * Omega_GW at each frequency (via log-log interpolation).
    """

    log_Gmu, log_alpha, q = pars[0], pars[1], pars[2]

    return ut.log_log_interpolate(freq, _compute_spectrum, log_Gmu, log_alpha, q)


cosmic_string_model_i_model = ut.Signal_model(
    "cosmic_string_model_i",
    cosmic_string_model_i,
    model_label="Cosmic String Model I",
    parameter_names=["log_Gmu", "log_alpha", "q"],
    parameter_labels=[
        r"$\log_{10}(G\mu)$",
        r"$\log_{10}\alpha$",
        r"$q$",
    ],
    prior={
        "log_Gmu": {"min": -12.0, "max": -6.0},
        "log_alpha": {"min": -3.0, "max": 0.0},
        "q": {"min": 1.01, "max": 2.0},
    },
)
