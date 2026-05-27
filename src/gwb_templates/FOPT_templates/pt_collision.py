from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.broken_power_law_a1 import (
    broken_power_law_a1,
    d1broken_power_law_a1,
)
from gwb_templates.FOPT_templates.pt_base import a_hubble, redshift_omega

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)

# Fixed spectral shape for the U(1)-symmetric scalar field scenario
# from Lewicki & Vaskonen 2023 [arXiv:2208.11697].
# Default: A=0.05, omega_p/beta=0.7, a=2.4, b=2.4, c=4.0
_A_RAW = 0.05
_OMEGA_P_OVER_BETA = 0.7
_A_SLOPE = 2.4
_B_SLOPE = 2.4
_C_SHARP = 4.0

_A_B = (
    _A_RAW
    * (
        0.5 * (_A_SLOPE / _B_SLOPE) ** (_B_SLOPE / (_A_SLOPE + _B_SLOPE))
        + 0.5 * (_B_SLOPE / _A_SLOPE) ** (_A_SLOPE / (_A_SLOPE + _B_SLOPE))
    )
    ** _C_SHARP
)
_OMEGA_B_OVER_BETA = _OMEGA_P_OVER_BETA * (_B_SLOPE / _A_SLOPE) ** (
    _C_SHARP / (_A_SLOPE + _B_SLOPE)
)
_N_1 = _A_SLOPE  # 2.4
_N_2 = -_B_SLOPE  # -2.4
_A_1 = (_A_SLOPE + _B_SLOPE) / _C_SHARP  # 1.2


def pt_collision(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Bubble collision contribution to the GW spectrum from a cosmological
    first-order phase transition (3 parameters).

    Based on:
    M. Lewicki and V. Vaskonen,
    "Gravitational waves from bubble collisions and fluid motion in
    strongly supercooled phase transitions",
    Eur.Phys.J. C83 (2023) 2, 109; [arXiv:2208.11697 [astro-ph.CO]].

    The spectral shape is fixed to the default U(1)-scalar scenario
    (A=0.05, omega_p/beta=0.7, a=b=2.4, c=4.0).

    Args:
        freq: Frequency grid.
        pars: [log10 K_tilde, log10 (beta/H_*), log10 T_star/GeV].
              K_tilde    -- fraction of vacuum energy converted to kinetic energy
              beta/H_*   -- inverse duration of the transition in Hubble units
              T_star     -- transition temperature in GeV
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_K_tilde, log_beta_over_H, log_T_star = (
        pars[0],
        pars[1],
        pars[2],
    )

    K_tilde = 10.0**log_K_tilde
    beta_over_H = 10.0**log_beta_over_H
    T_star = 10.0**log_T_star

    # Amplitude at the break frequency
    h2FGW0 = redshift_omega(T_star)
    h2Omega_b = h2FGW0 * _A_B * K_tilde**2 / beta_over_H**2

    # Break frequency today
    aH_star = a_hubble(T_star)
    f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * _OMEGA_B_OVER_BETA

    return broken_power_law_a1(
        freq,
        jnp.array([jnp.log10(h2Omega_b), jnp.log10(f_b), _N_1, _N_2, _A_1]),
    )


def d1pt_collision(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``pt_collision`` w.r.t. pars.

    Uses the chain rule through ``broken_power_law_a1``.  The g_star functions
    are piecewise-constant, so their T-derivatives are zero almost everywhere
    (consistent with JAX autodiff through ``jnp.select``).

    Args:
        freq: Frequency grid.
        pars: [log10 K_tilde, log10 (beta/H_*), log10 T_star/GeV].
    Returns:
        jax.Array of shape (N_freq, 3).
    """
    log_K_tilde, log_beta_over_H, log_T_star = pars[0], pars[1], pars[2]

    K_tilde = 10.0**log_K_tilde
    beta_over_H = 10.0**log_beta_over_H
    T_star = 10.0**log_T_star

    h2FGW0 = redshift_omega(T_star)
    h2Omega_b = h2FGW0 * _A_B * K_tilde**2 / beta_over_H**2
    aH_star = a_hubble(T_star)
    f_b = aH_star / (2.0 * jnp.pi) * beta_over_H * _OMEGA_B_OVER_BETA

    inner_pars = jnp.array([jnp.log10(h2Omega_b), jnp.log10(f_b), _N_1, _N_2, _A_1])
    J_inner = d1broken_power_law_a1(freq, inner_pars)  # (N_freq, 5)

    # d([log_h2Omega_b, log_f_b]) / d([log_K_tilde, log_beta_over_H, log_T_star])
    # h2Omega_b ∝ K_tilde^2 / beta_over_H^2  →  [2, -2, 0]
    # f_b ∝ a_hubble(T) * beta_over_H,  a_hubble ∝ T  →  [0, 1, 1]
    dq_dp = jnp.array([[2.0, -2.0, 0.0], [0.0, 1.0, 1.0]])

    return J_inner[:, :2] @ dq_dp


pt_collision_model = ut.Signal_model(
    "pt_collision",
    pt_collision,
    dtemplate=d1pt_collision,
    model_label="PT Bubble Collisions",
    parameter_names=[
        "log_K_tilde",
        "log_beta_over_H",
        "log_T_star",
    ],
    parameter_labels=[
        r"$\log_{10}\tilde{K}$",
        r"$\log_{10}(\beta/H_*)$",
        r"$\log_{10}(T_*/\mathrm{GeV})$",
    ],
    prior={
        "log_K_tilde": {"prior_type": "uniform", "minimum": -4.0, "maximum": 0.0},
        "log_beta_over_H": {"prior_type": "uniform", "minimum": 0.0, "maximum": 4.0},
        "log_T_star": {"prior_type": "uniform", "minimum": -2.0, "maximum": 4.0},
    },
)
