"""
Cosmic String Model I template (3 parameters).

Computes h^2 * Omega_GW(f) for a network of Nambu-Goto cosmic strings
following the analytic approach of arXiv:2405.03740.

The spectrum is computed entirely in JAX using ``jax.scipy.special.hyp2f1``
via a Pfaff transformation to handle the large-negative-argument regime that
arises in the Euler-Maclaurin summation. The template is fully JIT-compilable
and autodiff-compatible.

Reference: arXiv:2405.03740 (GW from cosmic strings in LISA: reconstruction
pipeline and physics interpretation).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.special as jsc

from gwb_templates import constants as ct
from gwb_templates.template import AnalyticTemplate
from gwb_templates.utils import log_log_interpolate

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
_Delta_gr = jnp.array([0.4065, 0.4853, 0.7085, 1.0])
# Scale factors corresponding to the changes in degrees of freedom
_a0 = 1.0
_a_star_factor = jnp.array([5.8897e16, 7.2182e19, 7.5617e21])


# ── Helper functions ──────────────────────────────────────────────────────────


def _hyperharmonic(r: jax.Array, N: jax.Array) -> jax.Array:
    """Euler-Maclaurin approximation to the generalized harmonic sum (Eq. B.2)."""
    zeta_r = jsc.zeta(r, 1.0)
    return zeta_r + (N / (1.0 - r) + 0.5 - r / (12.0 * N)) / N**r


def _get_epsilon_r(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Radiation-era loop size ratio eps_r (Eq. A.4)."""
    return jnp.asarray(alpha / _Gamma / Gmu)


def _get_epsilon_m(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Matter-era loop size ratio eps_m (Eq. A.4)."""
    return _get_epsilon_r(Gmu, alpha) * _xi_m / _xi_r


def _get_gamma_m(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Matter-era loop-size enhancement factor gamma_m (Eq. A.4)."""
    return 1.0 + 1.0 / _get_epsilon_m(Gmu, alpha)


def _get_beta_m(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Matter-era shape factor beta_m (Eq. A.4)."""
    eps_m = _get_epsilon_m(Gmu, alpha)
    return (1.0 + 1.0 / _get_gamma_m(Gmu, alpha)) / eps_m


def _get_D(nu_i: float, Omega_i: float, Gmu: jax.Array) -> jax.Array:
    """Characteristic frequency scale D_i (Eq. A.4)."""
    return jnp.asarray(2.0 * ct.H0_eV * jnp.sqrt(Omega_i) / nu_i / _Gamma / Gmu)


def _get_f_min_r(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Min. frequency from radiation-era loops (after Eq. A.5)."""
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    return D_r * (ct.Omega_M / ct.Omega_R) / _get_epsilon_r(Gmu, alpha)


def _get_f_min_m(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Min. frequency from matter-era loops (after Eq. A.5)."""
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    return D_m / _get_epsilon_m(Gmu, alpha)


def _get_a0_star(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Scale factor a0_* for DOF summation (after Eq. A.17)."""
    prefac = jnp.sqrt(2.0 * jnp.sqrt(ct.Omega_R) * ct.H0_SI * ct.t_planck_SI)
    return _a0 * prefac * jnp.sqrt(1.0 + _get_epsilon_r(Gmu, alpha)) / Gmu


def _get_an_star(Gmu: jax.Array, alpha: jax.Array, epoch: int) -> jax.Array:
    """Scale factor a_n* at each DOF step (after Eq. A.19)."""
    a0_star = _get_a0_star(Gmu, alpha)
    if epoch == 0:
        return a0_star
    if epoch < 4:
        return _a_star_factor[epoch - 1] * Gmu * a0_star
    if epoch == 4:
        return jnp.asarray(ct.a_eq * _a0)
    raise ValueError(f"Invalid epoch {epoch}")


def _get_A_n(
    f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array, epoch: int
) -> jax.Array:
    """Integration bound Script-A_n (after Eq. A.19)."""
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    a_star_n = _get_an_star(Gmu, alpha, epoch) / _a0
    return (D_r / a_star_n) / f_eV


def _get_tilde_A_rm(f_eV: jax.Array, Gmu: jax.Array) -> jax.Array:
    """Rad-to-matter transition bound tilde-A_rm (Eq. A.4)."""
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    return (D_m * jnp.sqrt(ct.Omega_M / ct.Omega_R)) / f_eV


def _get_tilde_A_m(f_eV: jax.Array, Gmu: jax.Array) -> jax.Array:
    """Matter-era integration bound tilde-A_m (Eq. A.4)."""
    return _get_D(_nu_m, ct.Omega_M, Gmu) / f_eV


def _get_C_i(xi_i: float, v_i: float) -> jax.Array:
    """Spectral amplitude coefficient C_i (Eq. A.4)."""
    return _c_tilde * _mathcal_F * v_i / xi_i**3 / jnp.sqrt(2.0)


def _get_C_r_no_dof(Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Radiation-era amplitude prefactor C_r without DOF changes (Eq. A.4)."""
    eps_r = _get_epsilon_r(Gmu, alpha)
    C_r = _get_C_i(_xi_r, _v_r)
    return 128.0 / 9.0 * jnp.pi * C_r * ct.Omega_R * (1.0 + eps_r) ** 1.5 / eps_r * Gmu


def _get_C_rm(f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Rad-to-matter transition amplitude C_rm (Eq. A.4)."""
    eps_r = _get_epsilon_r(Gmu, alpha)
    return (
        32.0
        * jnp.sqrt(3.0)
        * jnp.pi
        * _get_C_i(_xi_r, _v_r)
        * (1.0 + eps_r) ** 1.5
        / eps_r
        * ct.H0_eV
        * (ct.Omega_M * ct.Omega_R) ** 0.75
        / _Gamma
    ) / f_eV


def _get_C_m(f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array) -> jax.Array:
    """Matter-era amplitude coefficient C_m (Eq. A.4)."""
    eps_m = _get_epsilon_m(Gmu, alpha)
    return (
        162.0
        * jnp.pi
        * _get_C_i(_xi_m, _v_m)
        * (1.0 + eps_m)
        / eps_m
        * (ct.H0_eV * ct.Omega_M / _Gamma) ** 2.0
        / Gmu
    ) / f_eV**2.0


# ── Euler–Maclaurin summation pieces (Eqs. A.7–A.14) ────────────────────────


def _hyp2f1_neg(a: jax.Array, b: float, c: jax.Array, z: jax.Array) -> jax.Array:
    """hyp2f1 stable for z <= 0 via the Pfaff transformation.

    jax.scipy.special.hyp2f1 diverges for large negative z. The Pfaff
    identity maps z -> w = z/(z-1) in [0, 1) for all z <= 0, keeping the
    argument inside the unit disk:

        hyp2f1(a, b, c, z) = (1 - z)^{-a} * hyp2f1(a, c-b, c, z/(z-1))
    """
    w = z / (z - 1.0)
    return (1.0 - z) ** (-a) * jsc.hyp2f1(a, c - b, c, w)


def _I(a: jax.Array, b: float, A: jax.Array, N: jax.Array) -> jax.Array:
    """Integral I(a, b, A, N); Eq. A.7."""
    return N ** (a + 1.0) / (a + 1.0) * _hyp2f1_neg(a + 1.0, b, a + 2.0, -A * N)


def _D1(a: jax.Array, b: float | jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """First-order Euler-Maclaurin correction D1; Eq. A.8."""
    AN = A * N
    return N ** (a - 1.0) * (a + AN * (a - b)) / (1.0 + AN) ** (1.0 + b)


def _D3(a: jax.Array, b: float | jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """Third-order Euler-Maclaurin correction D3; Eq. A.9."""
    AN1 = 1.0 + A * N
    pref = N**a / AN1**b
    t1 = a * (1.0 - a) * (2.0 - a) / N**3.0
    t2 = -(b * (1.0 + b) * (2.0 + b)) * (A / AN1) ** 3.0
    t3 = 3.0 * a * b * (1.0 - a + A * N * (2.0 - a + b)) * (A / (N * AN1) ** 2.0)
    return pref * (t1 + t2 + t3)


def _delta(
    fn: Callable, a: jax.Array, b: float | jax.Array, A: jax.Array, N: jax.Array
) -> jax.Array:
    """Difference fn(a,b,A,N) - fn(a,b,A,1); after Eq. A.10."""
    return fn(a, b, A, N) - fn(a, b, A, 1.0)


def _M(a: jax.Array, b: float | jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """Euler-Maclaurin sum M(a, b, A, N); Eq. A.10."""
    return (
        _delta(_I, a, b, A, N)
        + 0.5 / (1.0 + A) ** b
        + 0.5 * N**a / (1.0 + A * N) ** b
        + _delta(_D1, a, b, A, N) / 12.0
        - _delta(_D3, a, b, A, N) / 720.0
    )


def _Il(a: jax.Array, b: jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """Log-weighted integral Il(a, b, A, N); Eq. A.11."""
    return (N ** (1.0 + a) * jnp.log(b * N / (1.0 + A * N)) - _I(a, 1.0, A, N)) / (
        1.0 + a
    )


def _Dl1(a: jax.Array, b: jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """Log-correction Dl1 (first-order E-M); Eq. A.12."""
    inv = 1.0 / (1.0 + A * N)
    return N ** (a - 1.0) * (inv + a * jnp.log(b * N * inv))


def _Dl3(a: jax.Array, b: jax.Array, A: jax.Array, N: jax.Array) -> jax.Array:
    """Log-correction Dl3 (third-order E-M); Eq. A.13."""
    inv = 1.0 / (1.0 + A * N)
    pref = N ** (a - 3.0)
    t1 = 2.0 * inv**3.0
    t2 = 3.0 * (a - 2.0) * inv**2.0
    t3 = 3.0 * (2.0 - a) * (1.0 - a) * inv
    t4 = (2.0 - a) * (1.0 - a) * a * jnp.log(b * N * inv)
    return pref * (t1 + t2 + t3 + t4)


def _N_func(
    a: jax.Array, b: float | jax.Array, A: jax.Array, N: jax.Array
) -> jax.Array:
    """Log-weighted Euler-Maclaurin sum N(a, b, A, N); Eq. A.14."""
    return (
        _delta(_Il, a, b, A, N)
        + 0.5 * jnp.log(b / (1.0 + A))
        + 0.5 * N**a * jnp.log(b * N / (1.0 + A * N))
        + _delta(_Dl1, a, b, A, N) / 12.0
        - _delta(_Dl3, a, b, A, N) / 720.0
    )


def _M_delta(
    q: jax.Array, A_next: jax.Array, A_now: jax.Array, N: jax.Array
) -> jax.Array:
    """M_delta piece for the DOF-step summation; Eq. A.19 (no prefactor)."""
    return _M(q, 1.5, A_next, N) - _M(q, 1.5, A_now, N)


# ── Omega contributions ───────────────────────────────────────────────────────


def _Omega_r_dof(
    f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array, q: jax.Array, N: jax.Array
) -> jax.Array:
    """GW energy density from radiation-era loops with SM DOF changes (Eq. A.18)."""
    Cr = _get_C_r_no_dof(Gmu, alpha) / jsc.zeta(q, 1.0)
    A_all = [_get_A_n(f_eV, Gmu, alpha, i) for i in range(len(_Delta_gr) + 1)]
    total = 0.0
    for i, dg in enumerate(_Delta_gr):
        A_now = jnp.sqrt(dg) * A_all[i]
        A_next = jnp.sqrt(dg) * A_all[i + 1]
        total += dg * _M_delta(-q, A_next, A_now, N)
    return Cr * total


def _Omega_rm(
    f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array, q: jax.Array, N: jax.Array
) -> jax.Array:
    """GW energy density from radiation-to-matter transition loops (Eq. A.15)."""
    pref = _get_C_rm(f_eV, Gmu, alpha) / jsc.zeta(q, 1.0)
    A_rm = _get_tilde_A_rm(f_eV, Gmu)
    A_m = _get_tilde_A_m(f_eV, Gmu)
    r = ct.Omega_R / ct.Omega_M
    t1 = 2.0 * _M(1.0 - q, 0.5, A_rm, N)
    t2 = _M(1.0 - q, 1.5, A_rm, N)
    t3 = -2.0 * _M(1.0 - q, 0.5, A_m, N)
    t4 = -_M(1.0 - q, 1.5, A_m, N)
    return pref * ((t1 + t2) / r**0.25 + t3 + t4)


def _Omega_m(
    f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array, q: jax.Array, N: jax.Array
) -> jax.Array:
    """GW energy density from matter-dominated loops (Eq. A.15)."""
    pref = _get_C_m(f_eV, Gmu, alpha) / jsc.zeta(q, 1.0)
    A_m = _get_tilde_A_m(f_eV, Gmu)
    gamma_m = _get_gamma_m(Gmu, alpha)
    beta_m = _get_beta_m(Gmu, alpha)
    t1 = _M(2.0 - q, 1.0, A_m, N)
    t2 = _hyperharmonic(q - 1.0, N) / A_m
    t3 = -beta_m * _hyperharmonic(q - 2.0, N)
    t4 = 2.0 * _N_func(2.0 - q, gamma_m * A_m, A_m, N)
    return pref * (t1 + t2 + t3 + t4)


def _Omega_small_alpha(
    f_eV: jax.Array, Gmu: jax.Array, alpha: jax.Array, q: jax.Array, N_m: jax.Array
) -> jax.Array:
    """Analytic GW energy density in the small-alpha limit (after Eq. A.16)."""
    t1 = 64.0 * jnp.pi * Gmu * ct.Omega_R * _get_C_i(_xi_r, _v_r) / 3.0
    pref2 = (
        54.0
        * jnp.pi
        * ct.H0_eV
        * ct.Omega_M**1.5
        * _get_C_i(_xi_m, _v_m)
        / _get_epsilon_m(Gmu, alpha)
        / jsc.zeta(q, 1.0)
        / _Gamma
        / f_eV
    )
    D_m = _get_D(_nu_m, ct.Omega_M, Gmu)
    eps_m = _get_epsilon_m(Gmu, alpha)
    t2 = _hyperharmonic(q - 1.0, N_m)
    t3 = D_m * _hyperharmonic(q - 2.0, N_m) / eps_m / f_eV
    return t1 + pref2 * (t2 - t3)


# ── Full spectrum (JAX-traceable) ────────────────────────────────────────────


def _compute_spectrum(
    freq: jax.Array, log_Gmu: jax.Array, log_alpha: jax.Array, q: jax.Array
) -> jax.Array:
    """Evaluate h^2 * Omega_GW(freq) for Model I (JAX-traceable)."""
    f_eV = freq * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    # N values clamped to ≥1 to avoid division-by-zero outside the valid range.
    N_r = jnp.maximum(1.0, jnp.floor(f_eV / f_min_r))
    N_m = jnp.maximum(1.0, jnp.floor(f_eV / f_min_m))

    def large_alpha_branch(_):
        omega_r = jnp.where(
            f_eV >= f_min_r,
            _Omega_r_dof(f_eV, Gmu, alpha, q, N_r),
            0.0,
        )
        omega_rm = jnp.where(
            f_eV >= f_min_m,
            _Omega_rm(f_eV, Gmu, alpha, q, N_m),
            0.0,
        )
        omega_m = jnp.where(
            f_eV >= f_min_m,
            _Omega_m(f_eV, Gmu, alpha, q, N_m),
            0.0,
        )
        # Matter-era loops only contribute in the small-alpha regime.
        omega_m_contrib = jnp.where(alpha < 1e-3, omega_m, jnp.zeros_like(f_eV))
        return omega_r + omega_rm + omega_m_contrib

    def small_alpha_branch(_):
        return jnp.where(
            f_eV > f_min_m,
            _Omega_small_alpha(f_eV, Gmu, alpha, q, N_m),
            0.0,
        )

    Omega = jax.lax.cond(
        alpha > _Gamma * Gmu * _xi_r,
        large_alpha_branch,
        small_alpha_branch,
        None,
    )

    return Omega * ct.h**2


# NB: For the moment we jit compile in this way, but it should be handled at a higher
# level (e.g. via a class method) if we want to support both JIT and non-JIT versions
# of the template.
_compute_spectrum_jit = jax.jit(_compute_spectrum)


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelI(AnalyticTemplate):
    r"""
    Cosmic String Model I: Nambu-Goto network (arXiv:2405.03740).

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`.
    log_alpha
        :math:`\log_{10}` of the loop-size parameter :math:`\alpha`.
    q
        Harmonic power-law index.

    The spectrum is computed directly in JAX via ``jax.scipy.special.hyp2f1``
    and is fully JIT-compilable and autodiff-compatible.
    """

    # TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        n_interp_points: int = 100,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            n_interp_points: Number of coarse grid points used in the
                log-log interpolation of the spectrum.
            model_name: Override instance identifier.
            model_label: Override display label.
            parameter_labels: Sparse override map for parameter labels.
            prior_by_param: Sparse override map for parameter priors.
        """
        self.n_interp_points: int = int(n_interp_points)

        default_labels = {
            "log_Gmu": r"$\log_{10}(G\mu)$",
            "log_alpha": r"$\log_{10}\alpha$",
            "q": r"$q$",
        }
        default_priors = {
            "log_Gmu": {"min": -12.0, "max": -6.0},
            "log_alpha": {"min": -3.0, "max": 0.0},
            "q": {"min": 1.01, "max": 2.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Cosmic String Model I"
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
        frequency: jax.Array,
        log_Gmu: jax.Array,
        log_alpha: jax.Array,
        q: jax.Array,
    ) -> jax.Array:
        r"""Evaluate :math:`\Omega_{\mathrm{GW}} h^2(f)` for Model I."""
        return log_log_interpolate(
            frequency,
            _compute_spectrum_jit,
            log_Gmu,
            log_alpha,
            q,
            n_points=self.n_interp_points,
        )
