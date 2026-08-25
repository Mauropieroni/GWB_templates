"""
Cosmic String Model I with Extra Degrees of Freedom (EDF) template (5 parameters).

Extends :class:`CosmicStringModelI` by allowing an additional BSM particle species to
become relativistic at temperature ``T_Extra``, which modifies the effective degrees
of freedom and hence the GW spectrum.

Reference: arXiv:2405.03740, Section 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.special as jsc

from gwb_templates import constants as ct
from gwb_templates.template import AnalyticTemplate
from gwb_templates.utils import log_log_interpolate
from gwb_templates.cosmic_string_templates.cosmic_string_model_i import (
    _Delta_gr,
    _a0,
    _Gamma,
    _M_delta,
    _Omega_m,
    _Omega_rm,
    _Omega_small_alpha,
    _get_C_r_no_dof,
    _get_D,
    _get_an_star,
    _get_epsilon_r,
    _get_f_min_m,
    _get_f_min_r,
    _nu_r,
    _safe_q,
    _xi_r,
)

Delta1_gr, Delta2_gr, Delta3_gr, Delta4_gr = (
    _Delta_gr[0],
    _Delta_gr[1],
    _Delta_gr[2],
    _Delta_gr[3],
)

_T_EXTRA_MIN = 0.005  # GeV
# CMB temperature today in eV (ct.T0_CMB_GeV is in GeV; T_Extra_eff below is in
# eV too, so T_star must match or the comparison picks the wrong DOF branch).
_T0_CMB_eV = ct.T0_CMB_GeV * 1e9
# Effective DOF for entropy/energy density at low temperature today, used to
# build the g0in normalization below (not ct.g_star_0, the *current* count).
_GSTAR_LOW_T = 3.36
_G_ENTROPY_LOW_T = 3.91
# Normalization for the extra-DOF step rescaling (Eq. after A.19 of 2405.03740):
# g0in = (g_entropy_low_T^(4/3) / gstar_low_T)^3.
_G0IN = (_G_ENTROPY_LOW_T ** (4.0 / 3.0) / _GSTAR_LOW_T) ** 3.0


# ── Extra-DOF helpers ─────────────────────────────────────────────────────────


def _delta_gr_from_delta(
    g0: float | jax.Array, Dg_Extra: jax.Array, delta: jax.Array
) -> jax.Array:
    """Rescale a DOF ratio to account for extra BSM degrees of freedom."""
    return (g0 / (Dg_Extra + g0 / delta**3)) ** (1.0 / 3.0)


def _get_Delta_gr_extra(
    Gmu: jax.Array, alpha: jax.Array, T_Extra: jax.Array, Dg_Extra: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute modified DOF ratios and scale factors with an extra BSM species.

    T_Extra is expected to already satisfy T_Extra >= _T_EXTRA_MIN (the prior's
    log_T_Extra range keeps it well above); not re-checked here, since a jax-traced
    value can't drive a Python ``if ...: raise`` under jit.

    Finds where T_Extra falls among the 4 existing SM dof-transition steps and inserts
    a new step there, rescaling every step before it via _delta_gr_from_delta.
    jnp.searchsorted replaces the Python linear search with an early ``break`` (not
    expressible under jit -- the loop bound would be data-dependent), and the mask-based
    jnp.where replaces a ``for j in range(flag)`` loop for the same reason.

    One corner deliberately not reproduced: the original also special-cased T_Extra_eff
    landing on *exactly* one of the 4 step temperatures, skipping the insertion -- which
    changes the output length depending on that equality, incompatible with jit's static
    shapes, and needs an exact float64 tie between independently-derived quantities
    (probability 0 for any continuous sampler).
    """
    T_Extra_eff = T_Extra * 1e9 / (1.0 + _get_epsilon_r(Gmu, alpha)) ** 0.5

    g0 = jnp.where(
        Dg_Extra == 0,
        1.0,
        _G0IN * Dg_Extra / (ct.g_star_high_T + Dg_Extra - _G0IN / Delta1_gr**3),
    )

    nsm = len(_Delta_gr)
    T_star = jnp.array(
        [
            _T0_CMB_eV * _a0 * _Delta_gr[i] / _get_an_star(Gmu, alpha, i)
            for i in range(nsm)
        ]
    )

    # T_star decreases with step index; searching the ascending reversal for
    # where T_Extra_eff sits reproduces the original's "walk backwards, stop
    # at the first T_star it's smaller than" search as one vectorized op.
    i = jnp.searchsorted(T_star[::-1], T_Extra_eff, side="right")
    flag = nsm - i

    T_star = jnp.insert(T_star, flag, T_Extra_eff)
    Delta_gr_extra = jnp.insert(_Delta_gr, flag, _Delta_gr[jnp.maximum(flag - 1, 0)])

    corrected = _delta_gr_from_delta(g0, Dg_Extra, Delta_gr_extra)
    Delta_gr_extra = jnp.where(jnp.arange(nsm + 1) < flag, corrected, Delta_gr_extra)

    a_star_dof = _a0 * jnp.append(Delta_gr_extra * _T0_CMB_eV / T_star, ct.a_eq)
    return Delta_gr_extra, a_star_dof


def _get_A_n_extra(
    f_eV: jax.Array, Gmu: jax.Array, a_star_dof: jax.Array, epoch: int
) -> jax.Array:
    """Integration bound Script-A_n with EDF scale factors (Eq. A.19)."""
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    a_star_n = a_star_dof[epoch] / _a0
    return (D_r / a_star_n) / f_eV


def _Omega_r_dof_edf(
    f_eV: jax.Array,
    Gmu: jax.Array,
    alpha: jax.Array,
    q: jax.Array,
    T_Extra: jax.Array,
    Dg_Extra: jax.Array,
    N: jax.Array,
) -> jax.Array:
    """Radiation-era loops with extra BSM DOF (Eq. A.18 with modified DOF history).

    The DOF steps are batched into a single (step, freq)-shaped _M_delta call rather
    than an unrolled Python loop: same arithmetic, far fewer XLA ops.
    """
    Cr = _get_C_r_no_dof(Gmu, alpha) / jsc.zeta(q, 1.0)
    Delta_gr_extra, a_star_dof = _get_Delta_gr_extra(Gmu, alpha, T_Extra, Dg_Extra)
    A_all = jnp.stack(
        [
            _get_A_n_extra(f_eV, Gmu, a_star_dof, i)
            for i in range(len(Delta_gr_extra) + 1)
        ]
    )
    sqrt_dg = jnp.sqrt(Delta_gr_extra)[:, jnp.newaxis]  # (n_steps, 1)
    A_now = sqrt_dg * A_all[:-1]
    A_next = sqrt_dg * A_all[1:]
    contrib = Delta_gr_extra[:, jnp.newaxis] * _M_delta(
        -q, A_next, A_now, N[jnp.newaxis, ...]
    )
    return Cr * jnp.sum(contrib, axis=0)


def _compute_spectrum_edf(
    freq: jax.Array,
    log_Gmu: jax.Array,
    log_alpha: jax.Array,
    q: jax.Array,
    log_T_Extra: jax.Array,
    Dg_Extra: jax.Array,
) -> jax.Array:
    """Evaluate h^2 * Omega_GW(freq) for the EDF variant."""
    q = _safe_q(q)
    f_eV = freq * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha
    T_Extra = 10.0**log_T_Extra

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    N_r = jnp.maximum(1.0, jnp.floor(f_eV / f_min_r))
    N_m = jnp.maximum(1.0, jnp.floor(f_eV / f_min_m))

    def large_alpha_branch(_: None) -> jax.Array:
        omega_r = jnp.where(
            f_eV >= f_min_r,
            _Omega_r_dof_edf(f_eV, Gmu, alpha, q, T_Extra, Dg_Extra, N_r),
            0.0,
        )
        omega_rm = jnp.where(f_eV >= f_min_m, _Omega_rm(f_eV, Gmu, alpha, q, N_m), 0.0)
        omega_m = jnp.where(f_eV >= f_min_m, _Omega_m(f_eV, Gmu, alpha, q, N_m), 0.0)
        # Matter-era loops only contribute in the small-alpha regime.
        omega_m_contrib = jnp.where(alpha < 1e-3, omega_m, jnp.zeros_like(f_eV))
        return omega_r + omega_rm + omega_m_contrib

    def small_alpha_branch(_: None) -> jax.Array:
        return jnp.where(
            f_eV > f_min_m,
            _Omega_small_alpha(f_eV, Gmu, alpha, q, N_m),
            0.0,
        )

    Omega = jax.lax.cond(
        alpha > _Gamma * Gmu * _xi_r, large_alpha_branch, small_alpha_branch, None
    )

    return Omega * ct.h**2


_compute_spectrum_edf_jit = jax.jit(_compute_spectrum_edf)


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelIEdf(AnalyticTemplate):
    r"""
    Cosmic String Model I with Extra Degrees of Freedom (arXiv:2405.03740, Sec. 4).

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`.
    log_alpha
        :math:`\log_{10}` of the loop-size parameter :math:`\alpha`.
    q
        Harmonic power-law index.
    log_T_Extra
        :math:`\log_{10}` of the BSM transition temperature in GeV.
    Dg_Extra
        Number of extra effective degrees of freedom contributed by the
        BSM species.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
    @article{Blanco-Pillado:2024aca,
        author = "Blanco-Pillado, Jose J. and Cui, Yanou and Kuroyanagi, Sachiko and
            Lewicki, Marek and Nardini, Germano and Pieroni, Mauro and Rybak, Ivan Yu. 
            and Sousa, Lara and Wachter, Jeremy M.",
        collaboration = "LISA Cosmology Working Group",
        title = "{Gravitational waves from cosmic strings in LISA: reconstruction 
            pipeline and physics interpretation}",
        eprint = "2405.03740",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        reportNumber = "LISA-COSWG-24-02, CERN-TH-2024-085",
        doi = "10.1088/1475-7516/2025/05/006",
        journal = "JCAP",
        volume = "05",
        pages = "006",
        year = "2025"
    }
    """,
    )

    def __init__(
        self,
        n_interp_points: int = 100,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.n_interp_points: int = int(n_interp_points)

        default_labels = {
            "log_Gmu": r"$\log_{10}(G\mu)$",
            "log_alpha": r"$\log_{10}\alpha$",
            "q": r"$q$",
            "log_T_Extra": r"$\log_{10}(T_\Delta/\mathrm{GeV})$",
            "Dg_Extra": r"$\Delta g$",
        }
        default_priors = {
            "log_Gmu": {"min": -12.0, "max": -6.0},
            "log_alpha": {"min": -3.0, "max": 0.0},
            "q": {"min": 1.01, "max": 2.0},
            "log_T_Extra": {"min": -2.0, "max": 5.0},
            "Dg_Extra": {"min": 0.0, "max": 200.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Cosmic String Model I (EDF)"
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
        log_T_Extra: jax.Array,
        Dg_Extra: jax.Array,
    ) -> jax.Array:
        return log_log_interpolate(
            frequency,
            _compute_spectrum_edf_jit,
            log_Gmu,
            log_alpha,
            q,
            log_T_Extra,
            Dg_Extra,
            n_points=self.n_interp_points,
        )
