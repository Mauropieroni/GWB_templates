"""
Cosmic String Model I with Extra Degrees of Freedom (EDF) template (5 parameters).

Extends :class:`CosmicStringModelI` by allowing an additional BSM particle
species to become relativistic at temperature ``T_Extra``, which modifies
the effective degrees of freedom and hence the GW spectrum.

Reference: arXiv:2405.03740, Section 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.typing as jtp
import numpy as np
import scipy.special as sc

from gwb_templates import constants as ct
from gwb_templates.template import NumericalTemplate
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
    _xi_r,
)

ArrayLike: TypeAlias = jtp.ArrayLike

Delta1_gr, Delta2_gr, Delta3_gr, Delta4_gr = (
    _Delta_gr[0],
    _Delta_gr[1],
    _Delta_gr[2],
    _Delta_gr[3],
)

_T_EXTRA_MIN = 0.005  # GeV


# ── Extra-DOF helpers ─────────────────────────────────────────────────────────


def _delta_gr_from_delta(g0: float, Dg_Extra: float, delta: float) -> float:
    """Rescale a DOF ratio to account for extra BSM degrees of freedom."""
    return (g0 / (Dg_Extra + g0 / delta**3)) ** (1.0 / 3.0)


def _get_Delta_gr_extra(
    Gmu: float, alpha: float, T_Extra: float, Dg_Extra: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute modified DOF ratios and scale factors with an extra BSM species."""
    if T_Extra < _T_EXTRA_MIN:
        raise ValueError(f"T_Extra = {T_Extra:.3g} GeV must be >= {_T_EXTRA_MIN} GeV")

    T_Extra_eff = T_Extra * 1e9 / (1.0 + _get_epsilon_r(Gmu, alpha)) ** 0.5

    if Dg_Extra == 0:
        g0 = 1.0
    else:
        g0 = (
            ct.g_star_0
            * Dg_Extra
            / (ct.g_star_high_T + Dg_Extra - ct.g_star_0 / Delta1_gr**3)
        )

    nsm = len(_Delta_gr)
    T_star = np.array(
        [
            ct.T0_CMB_GeV * _a0 * _Delta_gr[i] / _get_an_star(Gmu, alpha, i)
            for i in range(nsm)
        ]
    )

    Delta_gr_extra = _Delta_gr.copy()

    for i in range(nsm):
        if T_Extra_eff < T_star[nsm - i - 1]:
            flag = nsm - i
            T_star = np.insert(T_star, flag, T_Extra_eff)
            Delta_gr_extra = np.insert(_Delta_gr, flag, _Delta_gr[flag - 1])
            for j in range(flag):
                Delta_gr_extra[j] = _delta_gr_from_delta(
                    g0, Dg_Extra, Delta_gr_extra[j]
                )
            break
        elif T_Extra_eff == T_star[nsm - i - 1]:
            flag = nsm - i
            Delta_gr_extra = _Delta_gr.copy()
            for j in range(flag - 1):
                Delta_gr_extra[j] = _delta_gr_from_delta(
                    g0, Dg_Extra, Delta_gr_extra[j]
                )
            break

    a_star_dof = _a0 * np.append(Delta_gr_extra * ct.T0_CMB_GeV / T_star, ct.a_eq)
    return Delta_gr_extra, a_star_dof


def _get_A_n_extra(
    f_eV: np.ndarray, Gmu: float, a_star_dof: np.ndarray, epoch: int
) -> np.ndarray:
    """Integration bound Script-A_n with EDF scale factors (Eq. A.19)."""
    D_r = _get_D(_nu_r, ct.Omega_R, Gmu)
    a_star_n = a_star_dof[epoch] / _a0
    return (D_r / a_star_n) / f_eV


def _Omega_r_dof_edf(
    f_eV: np.ndarray,
    Gmu: float,
    alpha: float,
    q: float,
    T_Extra: float,
    Dg_Extra: float,
    N: np.ndarray,
) -> np.ndarray:
    """Radiation-era loops with extra BSM DOF (Eq. A.18 with modified DOF history)."""
    Cr = _get_C_r_no_dof(Gmu, alpha) / sc.zeta(q)
    Delta_gr_extra, a_star_dof = _get_Delta_gr_extra(Gmu, alpha, T_Extra, Dg_Extra)
    A_all = [
        _get_A_n_extra(f_eV, Gmu, a_star_dof, i) for i in range(len(Delta_gr_extra) + 1)
    ]
    total = 0.0
    for i, dg in enumerate(Delta_gr_extra):
        A_now = np.sqrt(dg) * A_all[i]
        A_next = np.sqrt(dg) * A_all[i + 1]
        total += dg * _M_delta(-q, A_next, A_now, N)
    return Cr * total


def _compute_spectrum_edf(
    freq: np.ndarray,
    log_Gmu: float,
    log_alpha: float,
    q: float,
    log_T_Extra: float,
    Dg_Extra: float,
) -> np.ndarray:
    """Evaluate h^2 * Omega_GW(freq) for the EDF variant (no log-interpolation)."""
    freq = np.asarray(freq)
    f_eV = freq * ct.h_bar_eV_s
    Gmu = 10.0**log_Gmu
    alpha = 10.0**log_alpha
    T_Extra = 10.0**log_T_Extra

    f_min_r = _get_f_min_r(Gmu, alpha)
    f_min_m = _get_f_min_m(Gmu, alpha)

    Omega = np.zeros(len(freq))

    if alpha > _Gamma * Gmu * _xi_r:
        mask_r = f_eV >= f_min_r
        if mask_r.any():
            fn_r = f_eV[mask_r]
            N_r = np.floor(fn_r / f_min_r)
            Omega[mask_r] += _Omega_r_dof_edf(
                fn_r, Gmu, alpha, q, T_Extra, Dg_Extra, N_r
            )
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


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelIEdf(NumericalTemplate):
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

    jittable: ClassVar[bool] = False
    bibtex_entries: ClassVar[tuple[str, ...]] = ()  # TODO: cite

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
                model_label if model_label is not None else "Cosmic String Model I (EDF)"
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
        log_Gmu: ArrayLike,
        log_alpha: ArrayLike,
        q: ArrayLike,
        log_T_Extra: ArrayLike,
        Dg_Extra: ArrayLike,
    ) -> jax.Array:
        freq = np.asarray(frequency, dtype=float)
        return log_log_interpolate(
            freq,
            _compute_spectrum_edf,
            float(log_Gmu),
            float(log_alpha),
            float(q),
            float(log_T_Extra),
            float(Dg_Extra),
            n_points=self.n_interp_points,
        )
