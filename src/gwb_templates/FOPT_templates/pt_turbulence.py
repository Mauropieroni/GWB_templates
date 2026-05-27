from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.double_broken_power_law import (
    double_broken_power_law,
    d1double_broken_power_law,
)
from gwb_templates.FOPT_templates.pt_base import a_hubble, redshift_omega

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)

# Spectral parameters from Roper Pol et al. 2022 [arXiv:2201.05630]
_A_AMP = 0.085
_N_EDDY = 2.0
_EXPONENTS = (3.0, 1.0, -8.0, 4.0, 2.15)  # n_1, n_2, n_3, a_1, a_2


def pt_turbulence(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    MHD turbulence contribution to the GW spectrum from a cosmological
    first-order phase transition (3 parameters).

    Based on:
    A. Roper Pol, C. Caprini, A. Neronov, and D. Semikoz,
    "Gravitational wave signal from primordial magnetic fields in the
    Pulsar Timing Array frequency band",
    Phys.Rev.D 105 (2022) 12; [arXiv:2201.05630 [astro-ph.CO]].

    Args:
        freq: Frequency grid.
        pars: [log10 Omega_s, log10 R_H_star, log10 T_star/GeV].
              Omega_s  -- fractional energy density in the turbulent source
              R_H_star -- bubble size times the Hubble rate
              T_star   -- transition temperature in GeV
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_Omega_s, log_R_H_star, log_T_star = (
        pars[0],
        pars[1],
        pars[2],
    )

    Omega_s = 10.0**log_Omega_s
    R_H_star = 10.0**log_R_H_star
    T_star = 10.0**log_T_star

    # Alfven speed
    vA = jnp.sqrt(0.75 * Omega_s)

    # Hubble rate at production redshifted to today
    aH_star = a_hubble(T_star)
    # Redshift factor for the amplitude
    h2FGW0 = redshift_omega(T_star)

    # Break frequencies
    f_1 = vA / _N_EDDY * aH_star / R_H_star
    f_2 = 2.2 * aH_star / R_H_star
    r_f = f_2 / f_1  # = 2.2 * N_eddy / vA

    # Normalization factor (matches DBPL value at f_2)
    n_1, n_2, n_3, a_1, a_2 = _EXPONENTS
    norm = r_f**3 * (1.0 + r_f**a_1) ** (-2.0 / a_1) * 2.0 ** (-11.0 / 3.0 / a_2)

    prefactor = 3.0 * _A_AMP * norm / (4.0 * jnp.pi**2 * _N_EDDY)
    h2Omega2 = h2FGW0 * prefactor * vA * Omega_s**2 * R_H_star**2

    return double_broken_power_law(
        freq,
        jnp.array(
            [
                jnp.log10(h2Omega2),
                jnp.log10(f_1),
                jnp.log10(f_2),
                n_1,
                n_2,
                n_3,
                a_1,
                a_2,
            ]
        ),
    )


def d1pt_turbulence(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``pt_turbulence`` w.r.t. pars.

    Uses the chain rule through ``double_broken_power_law``.  The g_star
    functions are piecewise-constant, so their T-derivatives are zero almost
    everywhere (consistent with JAX autodiff through ``jnp.select``).

    Args:
        freq: Frequency grid.
        pars: [log10 Omega_s, log10 R_H_star, log10 T_star/GeV].
    Returns:
        jax.Array of shape (N_freq, 3).
    """
    log_Omega_s, log_R_H_star, log_T_star = pars[0], pars[1], pars[2]

    Omega_s = 10.0**log_Omega_s
    R_H_star = 10.0**log_R_H_star
    T_star = 10.0**log_T_star

    vA = jnp.sqrt(0.75 * Omega_s)
    aH_star = a_hubble(T_star)
    h2FGW0 = redshift_omega(T_star)

    f_1 = vA / _N_EDDY * aH_star / R_H_star
    f_2 = 2.2 * aH_star / R_H_star
    r_f = f_2 / f_1  # = 2.2 * N_eddy / vA

    n_1, n_2, n_3, a_1, a_2 = _EXPONENTS
    norm = r_f**3 * (1.0 + r_f**a_1) ** (-2.0 / a_1) * 2.0 ** (-11.0 / 3.0 / a_2)
    prefactor = 3.0 * _A_AMP * norm / (4.0 * jnp.pi**2 * _N_EDDY)
    h2Omega2 = h2FGW0 * prefactor * vA * Omega_s**2 * R_H_star**2

    inner_pars = jnp.array(
        [jnp.log10(h2Omega2), jnp.log10(f_1), jnp.log10(f_2), n_1, n_2, n_3, a_1, a_2]
    )
    J_inner = d1double_broken_power_law(freq, inner_pars)  # (N_freq, 8)

    # d([log_h2Omega2, log_f_1, log_f_2]) / d([log_Omega_s, log_R_H_star, log_T_star])
    #
    # h2Omega2 ∝ norm(r_f) * vA * Omega_s^2 * R_H_star^2,  r_f = C/vA ∝ Omega_s^{-0.5}
    # d(log_h2Omega2)/d(log_Omega_s) = 1 + r_f^{a_1} / (1 + r_f^{a_1})
    # d(log_h2Omega2)/d(log_R_H_star) = 2   (∝ R_H_star^2)
    # d(log_h2Omega2)/d(log_T_star)   = 0   (h2FGW0 piecewise-constant in T)
    #
    # f_1 ∝ vA * aH_star / R_H_star,  vA ∝ Omega_s^0.5,  aH_star ∝ T
    # d(log_f_1)/d(log_Omega_s)   = 0.5
    # d(log_f_1)/d(log_R_H_star)  = -1
    # d(log_f_1)/d(log_T_star)    = 1
    #
    # f_2 ∝ aH_star / R_H_star  (no Omega_s)
    # d(log_f_2)/d(log_Omega_s)   = 0
    # d(log_f_2)/d(log_R_H_star)  = -1
    # d(log_f_2)/d(log_T_star)    = 1
    rf_a1 = r_f**a_1
    d_logOmega2_d_logOs = 1.0 + rf_a1 / (1.0 + rf_a1)

    dq_dp = jnp.array(
        [
            [d_logOmega2_d_logOs, 2.0, 0.0],  # log_h2Omega2
            [0.5, -1.0, 1.0],  # log_f_1
            [0.0, -1.0, 1.0],  # log_f_2
        ]
    )

    return J_inner[:, :3] @ dq_dp


pt_turbulence_model = ut.Signal_model(
    "pt_turbulence",
    pt_turbulence,
    dtemplate=d1pt_turbulence,
    model_label="PT MHD Turbulence",
    parameter_names=[
        "log_Omega_s",
        "log_R_H_star",
        "log_T_star",
    ],
    parameter_labels=[
        r"$\log_{10}\Omega_s$",
        r"$\log_{10}(R_* H_*)$",
        r"$\log_{10}(T_*/\mathrm{GeV})$",
    ],
    prior={
        "log_Omega_s": {"prior_type": "uniform", "minimum": -6.0, "maximum": 0.0},
        "log_R_H_star": {"prior_type": "uniform", "minimum": -3.0, "maximum": 0.0},
        "log_T_star": {"prior_type": "uniform", "minimum": -2.0, "maximum": 4.0},
    },
)
