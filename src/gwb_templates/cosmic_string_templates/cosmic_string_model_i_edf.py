"""
Cosmic String Model I with Extra Degrees of Freedom (EDF) template (5 parameters).

Extends cosmic_string_model_i_template by allowing an additional BSM particle
species to become relativistic at temperature T_Extra, which modifies the
effective degrees of freedom and hence the GW spectrum.

Reference: arXiv:2405.03740, Section 4.
"""

from collections.abc import Sequence

import jax
import numpy as np
import scipy.special as sc

from gwb_templates import constants as ct
from gwb_templates import utils as ut
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

jax.config.update("jax_enable_x64", True)

Delta1_gr, Delta2_gr, Delta3_gr, Delta4_gr = (
    _Delta_gr[0],
    _Delta_gr[1],
    _Delta_gr[2],
    _Delta_gr[3],
)

_T_EXTRA_MIN = 0.005  # GeV


# ── Extra-DOF helpers ─────────────────────────────────────────────────────────


def _delta_gr_from_delta(g0: float, Dg_Extra: float, delta: float) -> float:
    """
    Rescale a DOF ratio to account for extra BSM degrees of freedom.

    Args:
        g0: Fitting constant for the first DOF ratio step.
        Dg_Extra: Number of extra effective DOF contributed by the BSM species.
        delta: Original DOF ratio step (from _Delta_gr).
    Returns:
        Rescaled DOF ratio step including the extra species.
    """
    # Rescale Delta_gr to account for extra DOF (used inside _get_Delta_gr_extra)
    return (g0 / (Dg_Extra + g0 / delta**3)) ** (1.0 / 3.0)


def _get_Delta_gr_extra(
    Gmu: float, alpha: float, T_Extra: float, Dg_Extra: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute modified DOF ratios and scale factors with an extra BSM species.

    Inserts an additional step in the degree-of-freedom history when extra
    BSM particles become relativistic at temperature T_Extra.

    Args:
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        T_Extra: Temperature [GeV] at which the BSM species becomes relativistic.
        Dg_Extra: Number of extra effective DOF contributed by the BSM species.
    Returns:
        Tuple (Delta_gr_extra, a_star_dof):
            Delta_gr_extra: Modified array of DOF ratio steps.
            a_star_dof: Corresponding scale factors for each DOF step.
    """
    # Compute modified Delta_gr and scale factors when an extra step is
    # introduced in the spectrum due to extra degrees of freedom.
    if T_Extra < _T_EXTRA_MIN:
        raise ValueError(f"T_Extra = {T_Extra:.3g} GeV must be >= {_T_EXTRA_MIN} GeV")

    # 1st factor converts GeV to eV; 2nd accounts for the delay in the spectral
    # impact caused by dynamical effects (see Sec. 4 of arXiv:2405.03740)
    T_Extra_eff = T_Extra * 1e9 / (1.0 + _get_epsilon_r(Gmu, alpha)) ** 0.5

    if Dg_Extra == 0:
        g0 = 1.0  # avoid division by zero; result is independent of g0 here
    else:
        # Fitting constant that ensures the first Delta_rn matches the
        # realistic universe once Dg_Extra extra DOF are added
        g0 = (
            ct.g_star_0
            * Dg_Extra
            / (ct.g_star_high_T + Dg_Extra - ct.g_star_0 / Delta1_gr**3)
        )

    nsm = len(_Delta_gr)
    # Temperatures at which each original DOF step occurred
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
    """
    Integration bound Script-A_n with EDF scale factors (Eq. A.19).

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        a_star_dof: Array of scale factors for each DOF step (from
            ``_get_Delta_gr_extra``).
        epoch: DOF step index.
    Returns:
        np.ndarray of Script-A_n values at each frequency.
    """
    # Script-A_n from Eq. A.19 of arXiv:2405.03740 (EDF scale factors)
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
    """
    GW energy density from radiation-era loops with extra BSM DOF.

    Same as Eq. A.18 of arXiv:2405.03740 but with the modified DOF history
    from ``_get_Delta_gr_extra``.

    Args:
        f_eV: Natural-unit frequency array.
        Gmu: String tension G*mu.
        alpha: Loop-size parameter.
        q: Harmonic power-law index.
        T_Extra: Temperature [GeV] at which the extra species contributes.
        Dg_Extra: Number of extra effective DOF.
        N: Euler-Maclaurin truncation order.
    Returns:
        h^2 * Omega_GW contribution from radiation-era loops (EDF variant).
    """
    # Same as Eq. A.18 of arXiv:2405.03740 but with updated Delta_gr and
    # a_star_dof to account for the extra degrees of freedom
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


# ── Full spectrum + log-interpolated wrapper ──────────────────────────────────


def _compute_spectrum_edf(
    freq: np.ndarray,
    log_Gmu: float,
    log_alpha: float,
    q: float,
    log_T_Extra: float,
    Dg_Extra: float,
) -> np.ndarray:
    """
    Evaluate h^2 * Omega_GW(freq) for the EDF variant without log-interpolation.

    Args:
        freq: Frequency array [Hz].
        log_Gmu: log10 of the string tension.
        log_alpha: log10 of the loop-size parameter.
        q: Harmonic power-law index.
        log_T_Extra: log10 of the BSM transition temperature [GeV].
        Dg_Extra: Number of extra effective DOF.
    Returns:
        numpy.ndarray of h^2 * Omega_GW values at each frequency.
    """
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


# ── Public template function ──────────────────────────────────────────────────


def cosmic_string_model_i_edf(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Cosmic String Model I with Extra Degrees of Freedom (arXiv:2405.03740, Sec. 4).

    Computes h^2 * Omega_GW(f) for a Nambu-Goto cosmic string network with
    an additional BSM particle species becoming relativistic at T_Extra.

    No JAX automatic differentiation is available (uses NumPy/SciPy internally).

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu, log10 alpha, q, log10 T_Extra (GeV), Dg_Extra].
    Returns:
        jax.Array of h^2 * Omega_GW at each frequency (via log-log interpolation).
    """

    log_Gmu = pars[0]
    log_alpha = pars[1]
    q = pars[2]
    log_T_Extra = pars[3]
    Dg_Extra = pars[4]
    return ut.log_log_interpolate(
        freq,
        _compute_spectrum_edf,
        log_Gmu,
        log_alpha,
        q,
        log_T_Extra,
        Dg_Extra,
    )


cosmic_string_model_i_edf_model = ut.Signal_model(
    "cosmic_string_model_i_edf",
    cosmic_string_model_i_edf,
    model_label="Cosmic String Model I (EDF)",
    parameter_names=["log_Gmu", "log_alpha", "q", "log_T_Extra", "Dg_Extra"],
    parameter_labels=[
        r"$\log_{10}(G\mu)$",
        r"$\log_{10}\alpha$",
        r"$q$",
        r"$\log_{10}(T_\Delta/\mathrm{GeV})$",
        r"$\Delta g$",
    ],
    prior={
        "log_Gmu": {"min": -12.0, "max": -6.0},
        "log_alpha": {"min": -3.0, "max": 0.0},
        "q": {"min": 1.01, "max": 2.0},
        "log_T_Extra": {"min": -2.0, "max": 5.0},
        "Dg_Extra": {"min": 0.0, "max": 200.0},
    },
)
