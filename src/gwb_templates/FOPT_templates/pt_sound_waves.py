from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.double_broken_power_law import (
    double_broken_power_law,
    d1double_broken_power_law,
)
from gwb_templates.FOPT_templates.pt_base import a_hubble, h_star_tau, redshift_omega

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)

# Spectral shape parameters from Jinno et al. 2023 [arXiv:2209.04369]
_A = 0.11
_C_S = 1.0 / jnp.sqrt(3.0)
_EXPONENTS = (3.0, 1.0, -3.0, 2.0, 4.0)  # n_1, n_2, n_3, a_1, a_2


def pt_sound_waves(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Sound wave contribution to the GW spectrum from a cosmological
    first-order phase transition (4 parameters).

    Based on:
    R. Jinno, T. Konstandin, H. Rubira and I. Stomberg,
    "Higgsless simulations of cosmological phase transitions and
    gravitational waves",
    JCAP 02 (2023), 011; [arXiv:2209.04369 [astro-ph.CO]].

    Args:
        freq: Frequency grid.
        pars: [log10 K, log10 R_H_star, xi_w, log10 T_star/GeV].
              K        -- fluid kinetic energy fraction
              R_H_star -- bubble size times the Hubble rate
              xi_w     -- bubble wall velocity
              T_star   -- transition temperature in GeV
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_K, log_R_H_star, xi_w, log_T_star = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
    )

    K = 10.0**log_K
    R_H_star = 10.0**log_R_H_star
    T_star = 10.0**log_T_star

    c_s = _C_S
    xi_shell = jnp.abs(xi_w - c_s)
    xi_bubble = jnp.maximum(xi_w, c_s)

    # Hubble rate at production redshifted to today
    aH_star = a_hubble(T_star)
    # Redshift factor for the amplitude
    h2FGW0 = redshift_omega(T_star)
    # Duration of the sound wave source
    H_tau = h_star_tau(K, R_H_star)

    # Break frequencies
    f_1 = 0.2 * aH_star / R_H_star
    f_2 = 0.5 * aH_star / R_H_star * xi_bubble / xi_shell

    # Frequency ratio and normalization
    r_f = 2.5 * xi_bubble / xi_shell
    norm = (jnp.sqrt(2.0) + 2.0 * r_f / (1.0 + r_f**2)) / jnp.pi

    # Amplitude at f_2
    h2Omega2 = norm * h2FGW0 * _A * K**2 * H_tau * R_H_star

    n_1, n_2, n_3, a_1, a_2 = _EXPONENTS
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


def d1pt_sound_waves(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``pt_sound_waves`` w.r.t. pars.

    Uses the chain rule through ``double_broken_power_law``.  The g_star
    functions are piecewise-constant, so their T-derivatives are zero almost
    everywhere (consistent with JAX autodiff through ``jnp.select``).

    Args:
        freq: Frequency grid.
        pars: [log10 K, log10 R_H_star, xi_w, log10 T_star/GeV].
    Returns:
        jax.Array of shape (N_freq, 4).
    """
    log_K, log_R_H_star, xi_w, log_T_star = pars[0], pars[1], pars[2], pars[3]

    K = 10.0**log_K
    R_H_star = 10.0**log_R_H_star
    T_star = 10.0**log_T_star

    c_s = _C_S
    xi_shell = jnp.abs(xi_w - c_s)
    xi_bubble = jnp.maximum(xi_w, c_s)

    aH_star = a_hubble(T_star)
    h2FGW0 = redshift_omega(T_star)
    H_tau = h_star_tau(K, R_H_star)

    f_1 = 0.2 * aH_star / R_H_star
    f_2 = 0.5 * aH_star / R_H_star * xi_bubble / xi_shell
    r_f = 2.5 * xi_bubble / xi_shell
    norm = (jnp.sqrt(2.0) + 2.0 * r_f / (1.0 + r_f**2)) / jnp.pi
    h2Omega2 = norm * h2FGW0 * _A * K**2 * H_tau * R_H_star

    n_1, n_2, n_3, a_1, a_2 = _EXPONENTS
    inner_pars = jnp.array(
        [jnp.log10(h2Omega2), jnp.log10(f_1), jnp.log10(f_2), n_1, n_2, n_3, a_1, a_2]
    )
    J_inner = d1double_broken_power_law(freq, inner_pars)  # (N_freq, 8)

    ln10 = jnp.log(10.0)

    # --- d(log_h2Omega2) / d(p) ---
    # h2Omega2 = norm(xi_w) * h2FGW0 * A * K^2 * H_tau(K,R) * R
    # H_tau = min(1, R/sqrt(0.75*K))
    # When H_tau < 1:  K^2*H_tau*R ∝ K^{1.5}*R^2  ⇒ d/d(log_K)=1.5, d/d(log_R)=2
    # When H_tau = 1:  K^2*H_tau*R ∝ K^2*R      ⇒ d/d(log_K)=2.0, d/d(log_R)=1
    d_logOmega2_d_logK = jnp.where(H_tau < 1.0, 1.5, 2.0)
    d_logOmega2_d_logR = jnp.where(H_tau < 1.0, 2.0, 1.0)
    # norm depends on r_f = 2.5*xi_bubble/xi_shell
    # d(log10 norm)/d(xi_w) = d(norm)/d(r_f) * d(r_f)/d(xi_w) / (norm * ln10)
    # d(norm)/d(r_f) = 2*(1 - r_f^2) / (pi*(1 + r_f^2)^2)
    # d(r_f)/d(xi_w) = -2.5*c_s*sign(xi_w - c_s) / xi_shell^2
    d_norm_d_rf = 2.0 * (1.0 - r_f**2) / (jnp.pi * (1.0 + r_f**2) ** 2)
    d_rf_d_xiw = -2.5 * c_s * jnp.sign(xi_w - c_s) / xi_shell**2
    d_logOmega2_d_xiw = d_norm_d_rf * d_rf_d_xiw / (norm * ln10)

    # --- d(log_f_1) / d(p) ---
    # f_1 = 0.2 * aH_star / R  (aH_star ∝ T_star)
    # d/d(log_K)=0,  d/d(log_R)=-1,  d/d(xi_w)=0,  d/d(log_T)=1

    # --- d(log_f_2) / d(p) ---
    # f_2 = 0.5*aH_star/R * xi_bubble/xi_shell
    # d(log10(f_2))/d(xi_w) = -c_s*sign(xi_w-c_s) / (xi_shell * xi_bubble * ln10)
    d_logf2_d_xiw = -c_s * jnp.sign(xi_w - c_s) / (xi_shell * xi_bubble * ln10)

    dq_dp = jnp.array(
        [
            [
                d_logOmega2_d_logK,
                d_logOmega2_d_logR,
                d_logOmega2_d_xiw,
                0.0,
            ],  # log_h2Omega2
            [0.0, -1.0, 0.0, 1.0],  # log_f_1
            [0.0, -1.0, d_logf2_d_xiw, 1.0],  # log_f_2
        ]
    )

    return J_inner[:, :3] @ dq_dp


pt_sound_waves_model = ut.Signal_model(
    "pt_sound_waves",
    pt_sound_waves,
    dtemplate=d1pt_sound_waves,
    model_label="PT Sound Waves",
    parameter_names=[
        "log_K",
        "log_R_H_star",
        "xi_w",
        "log_T_star",
    ],
    parameter_labels=[
        r"$\log_{10}K$",
        r"$\log_{10}(R_* H_*)$",
        r"$\xi_w$",
        r"$\log_{10}(T_*/\mathrm{GeV})$",
    ],
    prior={
        "log_K": {"prior_type": "uniform", "minimum": -4.0, "maximum": 0.0},
        "log_R_H_star": {"prior_type": "uniform", "minimum": -3.0, "maximum": 0.0},
        "xi_w": {"prior_type": "uniform", "minimum": 0.01, "maximum": 0.99},
        "log_T_star": {"prior_type": "uniform", "minimum": -2.0, "maximum": 4.0},
    },
)
