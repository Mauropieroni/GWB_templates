r"""
Doubly-peaked GW spectrum from evaporating primordial black holes (PBHs)
in a general post-inflationary background with equation-of-state parameter
w != 1/3.

Models the sum of two poltergeist-enhanced scalar-induced GW peaks
following PBH reheating: an isocurvature-induced peak sourced by Poissonian
PBH number-density fluctuations, and an adiabatic-induced peak sourced by
the primordial curvature power spectrum. The physical fit parameters are
the PBH mass and abundance at formation, the background equation of state,
the PBH formation efficiency, and the amplitude/tilt of the primordial
curvature spectrum.

The isocurvature peak is modeled as a smooth BrokenPowerLaw (IR tilt 1, UV
tilt 11/3, break at f_br) multiplied by a smooth UV cutoff window
Theta_uv(f). The adiabatic peak is modeled as a smooth DoubleBrokenPowerLaw
(IR tilt 1, mid tilt 5, UV tilt n_eff(w), breaks at f_br,1 < f_br,2)
multiplied by the analogous window Theta_uv_ad(f; n_eff).

Reference: G. Domenech and J. Traenkle, Phys. Rev. D 111, 063528 (2025),
arXiv:2409.12125 [Domenech:2024wao].

NOTES: 
  - xi_1(beta, w) has no closed-form fit and is fixed to a constant (default 1.0).
  - g_*(T_rh) is fixed to the standard-model value 106.75 rather than
    computed self-consistently from M_PBH.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import math

from gwb_templates.generic_templates.broken_power_law import BrokenPowerLaw
from gwb_templates.generic_templates.double_broken_power_law import (
    DoubleBrokenPowerLaw,
)
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_G_STAR_REF: float = 106.75
_LOG10_OMEGA_R0_H2: float = math.log10(4.18e-5)
_LOG10_F_UV_KHZ_PREFACTOR: float = math.log10(1.7e3)  # 1.7 kHz -> Hz

# c_s^2 = 1/3 during radiation domination, fixed since the UV windows are
# evaluated after PBH evaporation, independent of the pre-evaporation w.
_C_S2_RD: float = 1.0 / 3.0

# Validity bound of the transfer-function fits: b <~ 0.65 (w >~ 7e-2).
_B_MAX: float = 0.65
_W_MIN: float = (1.0 - _B_MAX) / (3.0 * (1.0 + _B_MAX))  # ~= 0.0707


# ---------------------------------------------------------------------------
# w-dependent auxiliary functions
# ---------------------------------------------------------------------------


def _b_of_w(w: jax.Array) -> jax.Array:
    return (1.0 - 3.0 * w) / (1.0 + 3.0 * w)


def _log10_C_of_w(w: jax.Array, b: jax.Array) -> jax.Array:
    # C(w) = (9/20) * (0.135*(3+b))^(-1/(3w))
    return math.log10(9.0 / 20.0) - (1.0 / (3.0 * w)) * jnp.log10(0.135 * (3.0 + b))


def _n_of_w(w: jax.Array, b: jax.Array) -> jax.Array:
    # Piecewise fit; NOTE: not continuous at w=1/5 and w=2/3 in the source.
    n_soft = -(2.0 - b / 2.0)
    n_mid = -(1.83 + 0.285 * b - 0.790 * b**2)
    n_stiff = -(2.0 + b)
    return jnp.where(w < 0.2, n_soft, jnp.where(w < 2.0 / 3.0, n_mid, n_stiff))


def _n_eff(w: jax.Array, b: jax.Array, n_s: ArrayLike) -> jax.Array:
    return 11.0 / 3.0 + 2.0 * n_s + 4.0 * _n_of_w(w, b)


def _a_fit_of_b(b: jax.Array) -> jax.Array:
    return 7.76 + 18.3 * b + 12.5 * b**2


def _a_phi_of_w(w: jax.Array, b: jax.Array) -> jax.Array:
    # b_safe avoids a 1/0 in the (unselected) soft/stiff branches at w=1/3,
    # where b=0 exactly; jnp.where evaluates both branches, so a genuine
    # division by zero there would poison gradients even though the value
    # is discarded.
    b_safe = jnp.where(b == 0.0, 1e-8, b)
    a_fit = _a_fit_of_b(b)
    a_soft = (2.0 / (3.0 * b_safe)) * a_fit
    a_mid = 15.6 + 39.2 * b + 21.3 * b**2
    a_stiff = -(2.0 / (3.0 * b_safe)) * a_fit
    return jnp.where(w < 0.2, a_soft, jnp.where(w < 2.0 / 3.0, a_mid, a_stiff))


def _xi2_of_w(w: jax.Array, b: jax.Array) -> jax.Array:
    a_fit = _a_fit_of_b(b)
    base = (5.0 * w + 5.0) / (3.0 * w + 5.0) * a_fit
    exp_soft = 2.0 * (1.0 + 3.0 * w) / (3.0 * (1.0 + 5.0 * w))
    exp_stiff = (1.0 + 3.0 * w) / (3.0 + 3.0 * w)
    return jnp.where(w < 1.0 / 3.0, base**exp_soft, base**exp_stiff)

def _log10_beta_max(
    log_m_pbh: ArrayLike,
    w: jax.Array,
    gamma: ArrayLike,
) -> jax.Array:
    """
    Physical upper bound on log10(beta).
    """
    b = _b_of_w(w)
    log10_c = _log10_C_of_w(w, b)

    return (
        (3.0 * w / (4.0 * (1.0 + w))) * math.log10(2.3e-32)
        - (3.0 * w / (1.0 + w)) * log10_c
        - (2.0 * w / (1.0 + w)) * jnp.log10(gamma / 0.2)
        - (17.0 * w / (6.0 * (1.0 + w))) * (log_m_pbh - 4.0)
    )


def _log10_beta_min(
    log_m_pbh: ArrayLike,
    w: jax.Array,
    gamma: ArrayLike,
) -> jax.Array:
    """
    Physical lower bound on log10(beta).
    """
    log10_m_pl = math.log10(4.34e-6)

    return (
        (2.0 * w / (1.0 + w))
        * (
            math.log10(3.8 * math.pi / 480.0)
            + math.log10(108.0)
            - math.log10(1.0 + w)
            - math.log10(2.0 * math.pi)
            - jnp.log10(gamma)
            + 2.0 * (log10_m_pl - log_m_pbh)
        )
    )

# ---------------------------------------------------------------------------
# log10-amplitude and log10-frequency building blocks
# ---------------------------------------------------------------------------


def _log10_f_uv(log_m_pbh: ArrayLike, g_star: ArrayLike) -> jax.Array:
    log_m_ratio = log_m_pbh - 4.0  # log10(M_PBH / 1e4 g)
    return (
        _LOG10_F_UV_KHZ_PREFACTOR
        - (1.0 / 12.0) * jnp.log10(g_star / _G_STAR_REF)
        - (5.0 / 6.0) * log_m_ratio
    )


def _log10_omega_iso_peak(
    log_m_pbh: ArrayLike,
    log_beta: ArrayLike,
    w: jax.Array,
    b: jax.Array,
    gamma: ArrayLike,
    g_star: ArrayLike,
) -> jax.Array:
    log_m_ratio = log_m_pbh - 4.0
    log_gamma_ratio = jnp.log10(gamma / 0.2)
    return (
        math.log10(8.66e30)
        + 4.0 * _log10_C_of_w(w, b)
        + (4.0 * (1.0 + w) / (3.0 * w)) * log_beta
        - (1.0 / 3.0) * jnp.log10(g_star / _G_STAR_REF)
        + (8.0 / 3.0) * log_gamma_ratio
        + (34.0 / 9.0) * log_m_ratio
        + _LOG10_OMEGA_R0_H2
    )


def _log10_omega_iso_ir(
    log_m_pbh: ArrayLike,
    log_beta: ArrayLike,
    w: jax.Array,
    b: jax.Array,
    gamma: ArrayLike,
    g_star: ArrayLike,
) -> jax.Array:
    log_m_ratio = log_m_pbh - 4.0
    log_gamma_ratio = jnp.log10(gamma / 0.2)
    return (
        math.log10(3.49e24)
        + 4.0 * _log10_C_of_w(w, b)
        + (4.0 * (1.0 + w) / (3.0 * w)) * log_beta
        - (1.0 / 3.0) * jnp.log10(g_star / _G_STAR_REF)
        + (8.0 / 3.0) * log_gamma_ratio
        + (28.0 / 9.0) * log_m_ratio
        + _LOG10_OMEGA_R0_H2
    )


def _log10_omega_ad_uv(
    log_m_pbh: ArrayLike,
    log_beta: ArrayLike,
    w: jax.Array,
    b: jax.Array,
    gamma: ArrayLike,
    log_a_s: ArrayLike,
    n_s: ArrayLike,
    g_star: ArrayLike,
) -> jax.Array:
    n_w = _n_of_w(w, b)
    a_phi = _a_phi_of_w(w, b)
    log_m_ratio = log_m_pbh - 4.0
    return (
        math.log10(3.14e28)
        + (2.0 * n_w + n_s) * math.log10(3.0)
        - (3.0 * n_w + n_s) * math.log10(4.0)
        + 2.0 * log_a_s
        + 4.0 * jnp.log10(jnp.abs(a_phi))  # even power -> sign irrelevant
        - (4.0 * n_w / 3.0) * jnp.log10(gamma)
        - (2.0 * n_w * (1.0 + w) / (3.0 * w)) * log_beta
        - (1.0 / 3.0) * jnp.log10(g_star / _G_STAR_REF)
        + (34.0 / 9.0) * log_m_ratio
    )


def _log10_omega_ad_mid(
    log_m_pbh: ArrayLike,
    log_beta: ArrayLike,
    w: jax.Array,
    b: jax.Array,
    gamma: ArrayLike,
    log_a_s: ArrayLike,
    n_s: ArrayLike,
    xi_1: ArrayLike,
    g_star: ArrayLike,
) -> jax.Array:
    a_phi = _a_phi_of_w(w, b)
    n_eff = _n_eff(w, b, n_s)
    log_m_ratio = log_m_pbh - 4.0
    # NOTE: (6 - n_eff) is a genuine divisor in the source formula and can
    # vanish for pathological (w, n_s) combinations; guarded defensively.
    denom = 6.0 - n_eff
    denom_safe = jnp.where(denom == 0.0, 1e-8, denom)
    return (
        math.log10(6.88e20)
        + n_s * math.log10(2.0)
        + 2.0 * log_a_s
        + 4.0 * jnp.log10(jnp.abs(a_phi))
        - jnp.log10(jnp.abs(denom_safe))
        + (2.0 * n_s / 3.0 - 7.0 / 9.0) * jnp.log10(gamma)
        + (n_eff - 6.0) * jnp.log10(jnp.abs(xi_1))
        + ((6.0 * n_s - 7.0) * (1.0 + w) / (18.0 * w)) * log_beta
        - (1.0 / 3.0) * jnp.log10(g_star / _G_STAR_REF)
        + (28.0 / 9.0) * log_m_ratio
    )


def _log10_omega_ad_ir(
    log_m_pbh: ArrayLike,
    log_beta: ArrayLike,
    w: jax.Array,
    xi_2: jax.Array,
    gamma: ArrayLike,
    log_a_s: ArrayLike,
    n_s: ArrayLike,
    g_star: ArrayLike,
) -> jax.Array:
    log_m_ratio = log_m_pbh - 4.0
    inner_log10 = (
        (1.0 / 3.0) * jnp.log10(gamma)
        + jnp.log10(jnp.abs(xi_2))
        + ((1.0 + w) / (6.0 * w)) * log_beta
    )
    exponent = 5.0 / 3.0 + 2.0 * n_s
    return (
        math.log10(1.06e20)
        + 4.0 * jnp.log10(jnp.abs((5.0 + 3.0 * w) / (1.0 + w)))
        + n_s * math.log10(2.0)
        + 2.0 * log_a_s
        - jnp.log10(jnp.abs(5.0 + 6.0 * n_s))
        + exponent * inner_log10
        - (1.0 / 3.0) * jnp.log10(g_star / _G_STAR_REF)
        + (28.0 / 9.0) * log_m_ratio
    )


def _log10_f_br(
    log_omega_iso_ir: jax.Array, log_omega_iso_peak: jax.Array, log_f_uv: jax.Array
) -> jax.Array:
    return (3.0 / 8.0) * (log_omega_iso_ir - log_omega_iso_peak) + log_f_uv


def _log10_f_br1(
    log_omega_ad_ir: jax.Array, log_omega_ad_mid: jax.Array, log_f_uv: jax.Array
) -> jax.Array:
    return 0.25 * (log_omega_ad_ir - log_omega_ad_mid) + log_f_uv


def _log10_f_br2(
    log_omega_ad_uv: jax.Array,
    log_omega_ad_mid: jax.Array,
    n_eff: jax.Array,
    log_f_uv: jax.Array,
) -> jax.Array:
    # NOTE: exponent 1/(5 - n_eff) is singular at n_eff = 5 (degenerate
    # with the mid-branch tilt); guarded defensively, not physically fixed.
    denom = 5.0 - n_eff
    denom_safe = jnp.where(denom == 0.0, 1e-6, denom)
    return (log_omega_ad_uv - log_omega_ad_mid) / denom_safe + log_f_uv


# ---------------------------------------------------------------------------
# Smooth UV cutoff windows
# ---------------------------------------------------------------------------


def _hyp2f1_series(
    a: ArrayLike, b: ArrayLike, c: ArrayLike, z: jax.Array, n_terms: int = 50
) -> jax.Array:
    r"""
    Truncated power-series evaluation of the Gauss hypergeometric function
    :math:`{}_2F_1(a, b; c; z) = \sum_k \frac{(a)_k (b)_k}{(c)_k\,k!} z^k`.

    JAX has no native hyp2f1. Used only for the UV cutoff windows below,
    where :math:`z = c_s^2 s_0^2 \le 1/3` and :math:`|{-n_{\rm eff}}|` is
    O(1-3) in the parameter regime of interest, so the series converges
    quickly. NOT a general-purpose hyp2f1 implementation.
    """
    dtype = jnp.result_type(a, b, c, z, jnp.float32)
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)
    c = jnp.asarray(c, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)

    k = jnp.arange(n_terms - 1, dtype=dtype)
    ratio = (a + k) * (b + k) / ((c + k) * (k + 1.0))
    coeffs = jnp.concatenate([jnp.ones((1,), dtype=dtype), jnp.cumprod(ratio)])
    powers = z[..., None] ** jnp.arange(n_terms, dtype=dtype)
    return jnp.sum(coeffs * powers, axis=-1)


def _theta_uv_iso(
    s0: jax.Array, c_s2: float = _C_S2_RD, n_terms: int = 50
) -> jax.Array:
    z = c_s2 * s0**2
    c_s4 = c_s2**2
    hyp = _hyp2f1_series(5.0 / 6.0, 1.0, 1.5, z, n_terms=n_terms)
    numerator = 3.0 * s0 * (5.0 * c_s4 - 2.0 * c_s2 * (2.0 * s0**2 + 5.0) + 9.0) - (
        5.0 * c_s2 * (c_s2 + 6.0) - 27.0
    ) * s0 * (c_s2 * s0**2 - 1.0) * hyp
    denominator = 10.0 * c_s4 * (1.0 - z) ** (2.0 / 3.0)
    return numerator / denominator


def _theta_uv_ad(
    s0: jax.Array, n_eff: jax.Array, c_s2: float = _C_S2_RD, n_terms: int = 50
) -> jax.Array:
    z = c_s2 * s0**2
    hyp1 = _hyp2f1_series(2.5, -n_eff, 3.5, z, n_terms=n_terms)
    hyp2 = _hyp2f1_series(1.5, -n_eff, 2.5, z, n_terms=n_terms)
    hyp3 = _hyp2f1_series(0.5, -n_eff, 1.5, z, n_terms=n_terms)
    return (
        (2.0 / 5.0) * s0**5 * hyp1
        - (4.0 / 3.0) * s0**3 * hyp2
        + 2.0 * s0 * hyp3
    )


def _s0_of_f(
    frequency: jax.Array, log_f_uv: jax.Array, c_s2: float = _C_S2_RD
) -> jax.Array:
    c_s_inv = 1.0 / math.sqrt(c_s2)
    f_uv = 10.0**log_f_uv
    r = f_uv / frequency
    return jnp.clip(2.0 * r - c_s_inv, 0.0, 1.0)


def _window_iso(
    frequency: jax.Array, log_f_uv: jax.Array, n_terms: int = 50
) -> jax.Array:
    s0 = _s0_of_f(frequency, log_f_uv)
    theta = _theta_uv_iso(s0, n_terms=n_terms)
    return theta


def _window_ad(
    frequency: jax.Array, log_f_uv: jax.Array, n_eff: jax.Array, n_terms: int = 50
) -> jax.Array:
    s0 = _s0_of_f(frequency, log_f_uv)
    theta = _theta_uv_ad(s0, n_eff, n_terms=n_terms)
    return theta


# ---------------------------------------------------------------------------
# Validity check (b <= 0.65, i.e. w >= ~7e-2)
# ---------------------------------------------------------------------------


def _check_w_validity(w: jax.Array, label: str) -> None:
    """
    JIT/grad-compatible runtime warning (not an exception, does not alter
    the output) if any element of ``w`` violates b <= 0.65 (w >= ~7e-2).
    """
    b = _b_of_w(jnp.asarray(w))
    invalid = jnp.any(b > _B_MAX)

    def _warn() -> None:
        jax.debug.print(
            "[{label}] w = {w} (b = {b}) outside regime of validity of "
            "fit (requires b <= 0.65, i.e. w >= ~7e-2). Results are not "
            "reliable.",
            label=label,
            w=w,
            b=b,
        )
        return None

    jax.lax.cond(invalid, _warn, lambda: None)


# ---------------------------------------------------------------------------
# Template class
# ---------------------------------------------------------------------------


class EvaporatingPBHDoublyPeaked(AnalyticTemplate):
    r"""
    Doubly-peaked SIGW spectrum from PBH reheating in a general background
    equation of state (6 physical parameters).

    Free parameters
    ---------------
    log_m_pbh
        :math:`\log_{10}(M_{\rm PBH}/{\rm g})`, PBH mass at formation.
    log_beta
        :math:`\log_{10}\beta`, initial PBH energy-density fraction.
    w
        Equation-of-state parameter of the background fluid at PBH
        formation. Restricted to :math:`w \gtrsim 7\times10^{-2}`
        (:math:`b \lesssim 0.65`), the stated validity range of the
        underlying curvature-transfer-function fits; :math:`w=1/3`
        recovers the standard radiation-domination scenario.
    gamma
        :math:`\mathcal{O}(1)` PBH-formation efficiency factor
        (fraction of the horizon mass collapsing into the PBH).
    log_a_s
        :math:`\log_{10}A_s`, amplitude of the primordial curvature
        power spectrum at the PBH-formation scale.
    n_s
        Spectral tilt of the primordial curvature power spectrum.

    Notes
    -----
    Composed as the sum of a :class:`BrokenPowerLaw` (isocurvature-induced
    peak) and a :class:`DoubleBrokenPowerLaw` (adiabatic-induced peak),
    each multiplied by an analytic smooth UV cutoff window.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Domenech:2024wao,
    author = {Dom{\`e}nech, Guillem and Tr{\"a}nkle, Jan},
    title = "{From formation to evaporation: Induced gravitational wave probes of the primordial black hole reheating scenario}",
    eprint = "2409.12125",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.111.063528",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "6",
    pages = "063528",
    year = "2025"
}

""",
    )

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
        g_star: float = _G_STAR_REF,
        xi_1: float = 1.0,
        bpl_log_transition: float = -1.0,
        dbpl_log_transitions: tuple[float, float] = (-3.0, -10.0),
        hyp2f1_n_terms: int = 50,
    ) -> None:
        # g_star: fixed reference value for g_*(T_rh); not computed
        # self-consistently from T_rh(M_PBH) relation .
        # xi_1: O(1) prefactor in the adiabatic mid-branch amplitude that
        # must be computed numerically; fixed to a constant here as an approximation.
        # {bpl,dbpl}_log_transition(s): smoothness parameters of the
        # underlying templates, set to a small default (delta = 10^-1) to
        # approximate the genuine kinks of the physical piecewise spectrum
        # without risking numerical issues at delta -> 0.
        # hyp2f1_n_terms: number of terms in the truncated hyp2f1 series
        # used by the UV cutoff windows; see module docstring.
        self._bpl = BrokenPowerLaw()
        self._dbpl = DoubleBrokenPowerLaw()
        self._g_star = g_star
        self._xi_1 = xi_1
        self._bpl_log_transition = bpl_log_transition
        self._dbpl_log_transitions = dbpl_log_transitions
        self._hyp2f1_n_terms = hyp2f1_n_terms

        default_labels = {
            "log_m_pbh": r"$\log_{10}(M_{\rm PBH}/\mathrm{g})$",
            "log_beta": r"$\log_{10}\beta$",
            "w": r"$w$",
            "gamma": r"$\gamma$",
            "log_a_s": r"$\log_{10}A_s$",
            "n_s": r"$n_s$",
        }
        # NOTE: placeholder priors, not derived from the (beta, M, w)
        # BBN/Delta N_eff bounds worked out in the underlying papers.
        # In particular beta_min/beta_max depend jointly on w and M_PBH;
        # a flat independent prior in log_beta is a simplification.
        # The lower bound on w enforces b <= 0.65 (see _W_MIN above).
        default_priors = {
            "log_m_pbh": {
                "prior_type": "uniform",
                "minimum": 0.0,
                "maximum": 8.0,
            },
            "log_beta": {
                "prior_type": "uniform",
                "minimum": -20.0,
                "maximum": -1.0,
            },
            "w": {
                "prior_type": "uniform",
                "minimum": _W_MIN,
                "maximum": 1.0,
            },
            "gamma": {
                "prior_type": "uniform",
                "minimum": 0.05,
                "maximum": 0.5,
            },
            "log_a_s": {
                "prior_type": "uniform",
                "minimum": -12.0,
                "maximum": -2.0,
            },
            "n_s": {
                "prior_type": "uniform",
                "minimum": 0.8,
                "maximum": 1.2,
            },
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Evaporating PBH Doubly-Peaked Spectrum"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def log_prior(self, theta: jax.Array) -> jax.Array:
        """
        Log-prior for the physical parameter vector.

        Uniform in log_beta conditional on (log_m_pbh, w, gamma).
        """

        log_m_pbh, log_beta, w, gamma, log_a_s, n_s = theta

        valid = (
            (log_m_pbh >= 0.0)
            & (log_m_pbh <= 8.0)
            & (w >= _W_MIN)
            & (w <= 1.0)
            & (gamma >= 0.05)
            & (gamma <= 0.5)
            & (log_a_s >= -12.0)
            & (log_a_s <= -2.0)
            & (n_s >= 0.8)
            & (n_s <= 1.2)
        )

        log_beta_min = _log10_beta_min(
            log_m_pbh,
            w,
            gamma,
        )

        log_beta_max = _log10_beta_max(
            log_m_pbh,
            w,
            gamma,
        )

        beta_width = log_beta_max - log_beta_min

        valid &= log_beta >= log_beta_min
        valid &= log_beta <= log_beta_max
        valid &= beta_width > 0.0

        logp_beta = -jnp.log(beta_width)

        return jnp.where(valid, logp_beta, -jnp.inf)
    
    def _isocurvature_params(
        self,
        log_m_pbh: ArrayLike,
        log_beta: ArrayLike,
        w: jax.Array,
        gamma: ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return (log_amplitude, log_f_br, log_f_uv) for the BPL piece."""
        b = _b_of_w(w)
        log_omega_iso_peak = _log10_omega_iso_peak(
            log_m_pbh, log_beta, w, b, gamma, self._g_star
        )
        log_omega_iso_ir = _log10_omega_iso_ir(
            log_m_pbh, log_beta, w, b, gamma, self._g_star
        )
        log_f_uv = _log10_f_uv(log_m_pbh, self._g_star)
        log_f_br = _log10_f_br(log_omega_iso_ir, log_omega_iso_peak, log_f_uv)
        # BrokenPowerLaw.log_amplitude is defined at the pivot (= break)
        # frequency. Below the break, the physical spectrum scales as
        # (f/f_uv)^1, so we propagate Omega_IR to f_br.
        log_amp_iso = log_omega_iso_ir + (log_f_br - log_f_uv)
        return log_amp_iso, log_f_br, log_f_uv

    def _adiabatic_params(
        self,
        log_m_pbh: ArrayLike,
        log_beta: ArrayLike,
        w: jax.Array,
        gamma: ArrayLike,
        log_a_s: ArrayLike,
        n_s: ArrayLike,
        log_f_uv: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return (log_amplitude, log_f_br1, log_f_br2, n_eff) for the DBPL piece."""
        b = _b_of_w(w)
        n_eff = _n_eff(w, b, n_s)
        xi_2 = _xi2_of_w(w, b)

        log_omega_ad_uv = _log10_omega_ad_uv(
            log_m_pbh, log_beta, w, b, gamma, log_a_s, n_s, self._g_star
        )
        log_omega_ad_mid = _log10_omega_ad_mid(
            log_m_pbh, log_beta, w, b, gamma, log_a_s, n_s, self._xi_1, self._g_star
        )
        log_omega_ad_ir = _log10_omega_ad_ir(
            log_m_pbh, log_beta, w, xi_2, gamma, log_a_s, n_s, self._g_star
        )

        log_f_br1 = _log10_f_br1(log_omega_ad_ir, log_omega_ad_mid, log_f_uv)
        log_f_br2 = _log10_f_br2(log_omega_ad_uv, log_omega_ad_mid, n_eff, log_f_uv)

        # DoubleBrokenPowerLaw's amplitude is anchored at the SECOND
        # (higher) break frequency f_2 = f_br2, per instruction. The
        # physical mid-branch value at f_br2 is Omega_ad_mid*(f_br2/f_uv)^5;
        # this equals the UV-branch value there by construction of f_br2.
        log_amp_ad = log_omega_ad_ir + (log_f_br1 - log_f_uv)
        #log_amp_ad = (log_omega_ad_ir - 4.0 * (log_f_br1 - log_f_uv)
        #            + 5.0 * (log_f_br2 - log_f_uv))
        #log_amp_ad = log_omega_ad_ir + log_f_uv - log_f_br2

        return log_amp_ad, log_f_br1, log_f_br2, n_eff

    def omega_gw_h2_isocurvature(
        self,
        frequency: ArrayLike,
        log_m_pbh: ArrayLike,
        log_beta: ArrayLike,
        w: ArrayLike,
        gamma: ArrayLike,
    ) -> jax.Array:
        """
        Isocurvature-induced peak only (not part of the abstract template
        interface; provided for diagnostic plotting, cf. the individual
        contributions shown in the reference figure).
        """
        w_arr = jnp.asarray(w)
        _check_w_validity(w_arr, "omega_gw_h2_isocurvature")
        log_amp_iso, log_f_br, log_f_uv = self._isocurvature_params(
            log_m_pbh, log_beta, w_arr, gamma
        )
        iso = self._bpl.omega_gw_h2(
            frequency,
            log_amp_iso,
            log_f_br,
            1.0,
            11.0 / 3.0,
            self._bpl_log_transition,
        )
        window = _window_iso(
            jnp.asarray(frequency), log_f_uv, n_terms=self._hyp2f1_n_terms
        )
        return jnp.asarray(iso * window)

    def omega_gw_h2_adiabatic(
        self,
        frequency: ArrayLike,
        log_m_pbh: ArrayLike,
        log_beta: ArrayLike,
        w: ArrayLike,
        gamma: ArrayLike,
        log_a_s: ArrayLike,
        n_s: ArrayLike,
    ) -> jax.Array:
        """Adiabatic-induced peak only (see :meth:`omega_gw_h2_isocurvature`)."""
        w_arr = jnp.asarray(w)
        _check_w_validity(w_arr, "omega_gw_h2_adiabatic")
        log_f_uv = _log10_f_uv(log_m_pbh, self._g_star)
        log_amp_ad, log_f_br1, log_f_br2, n_eff = self._adiabatic_params(
            log_m_pbh, log_beta, w_arr, gamma, log_a_s, n_s, log_f_uv
        )
        a1, a2 = self._dbpl_log_transitions
        ad = self._dbpl.omega_gw_h2(
            frequency,
            log_amp_ad,
            log_f_br2,
            log_f_br1,
            n_eff,
            5.0,
            1.0,
            a1,
            a2,
        )
        window = _window_ad(
            jnp.asarray(frequency), log_f_uv, n_eff, n_terms=self._hyp2f1_n_terms
        )
        return jnp.asarray(ad * window)

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_m_pbh: ArrayLike,
        log_beta: ArrayLike,
        w: ArrayLike,
        gamma: ArrayLike,
        log_a_s: ArrayLike,
        n_s: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the total (isocurvature + adiabatic) spectrum at
        ``frequency``, including the smooth UV cutoff windows.
        """
        w_arr = jnp.asarray(w)
        _check_w_validity(w_arr, "omega_gw_h2")
        frequency_arr = jnp.asarray(frequency)

        log_amp_iso, log_f_br, log_f_uv = self._isocurvature_params(
            log_m_pbh, log_beta, w_arr, gamma
        )
        iso = self._bpl.omega_gw_h2(
            frequency,
            log_amp_iso,
            log_f_br,
            1.0,
            11.0 / 3.0,
            self._bpl_log_transition,
        )
        window_iso = _window_iso(
            frequency_arr, log_f_uv, n_terms=self._hyp2f1_n_terms
        )

        log_amp_ad, log_f_br1, log_f_br2, n_eff = self._adiabatic_params(
            log_m_pbh, log_beta, w_arr, gamma, log_a_s, n_s, log_f_uv
        )
        a1, a2 = self._dbpl_log_transitions
        ad = self._dbpl.omega_gw_h2(
            frequency,
            log_amp_ad,
            log_f_br2,
            log_f_br1,
            n_eff,
            5.0,
            1.0,
            a1,
            a2,
        )
        window_ad = _window_ad(
            frequency_arr, log_f_uv, n_eff, n_terms=self._hyp2f1_n_terms
        )

        return jnp.asarray(iso * window_iso + ad * window_ad)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Jacobian via JAX forward-mode autodiff of :meth:`omega_gw_h2`.

        We do not hand-derive a closed-form Jacobian here: with 6 scalar
        parameters, many chained power-law/piecewise terms, and the UV
        cutoff windows (which themselves involve hyp2f1 series), a manual
        derivation would be long and error-prone, and mathematically
        offers no advantage over automatic differentiation of the
        (already analytic) forward formula. ``jacfwd`` is used rather than
        the reverse-mode default because the number of parameters (6) is
        much smaller than the typical size of ``frequency``.
        """
        log_m_pbh, log_beta, w, gamma, log_a_s, n_s = theta

        def f(
            log_m_pbh: jax.Array,
            log_beta: jax.Array,
            w: jax.Array,
            gamma: jax.Array,
            log_a_s: jax.Array,
            n_s: jax.Array,
        ) -> jax.Array:
            return self.omega_gw_h2(
                frequency, log_m_pbh, log_beta, w, gamma, log_a_s, n_s
            )

        jac = jax.jacfwd(f, argnums=(0, 1, 2, 3, 4, 5))(
            log_m_pbh, log_beta, w, gamma, log_a_s, n_s
        )
        return jnp.stack(jac, axis=-1)
    
    