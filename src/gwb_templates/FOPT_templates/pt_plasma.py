from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.FOPT_templates.pt_sound_waves import pt_sound_waves, d1pt_sound_waves
from gwb_templates.FOPT_templates.pt_turbulence import pt_turbulence, d1pt_turbulence

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def pt_plasma(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Combined sound wave + MHD turbulence GW spectrum from a cosmological
    first-order phase transition (5 parameters).

    The turbulence source energy is set to a fraction ``epsilon`` of the
    bulk kinetic energy:  log_Omega_s = log_K + log10(epsilon).

    Args:
        freq: Frequency grid.
        pars: [log10 K, log10 R_H_star, xi_w, log10 T_star/GeV, epsilon].
              K        -- fluid kinetic energy fraction
              R_H_star -- bubble size times the Hubble rate
              xi_w     -- bubble wall velocity
              T_star   -- transition temperature in GeV
              epsilon  -- fraction of bulk kinetic energy in turbulence
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_K, log_R_H_star, xi_w, log_T_star, epsilon = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )

    sw = pt_sound_waves(freq, jnp.array([log_K, log_R_H_star, xi_w, log_T_star]))
    log_Omega_s = log_K + jnp.log10(epsilon)
    mhd = pt_turbulence(freq, jnp.array([log_Omega_s, log_R_H_star, log_T_star]))

    return sw + mhd


def d1pt_plasma(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``pt_plasma`` w.r.t. pars.

    Composes the Jacobians of ``pt_sound_waves`` and ``pt_turbulence`` via the
    chain rule.

    Args:
        freq: Frequency grid.
        pars: [log10 K, log10 R_H_star, xi_w, log10 T_star/GeV, epsilon].
    Returns:
        jax.Array of shape (N_freq, 5).
    """
    log_K, log_R_H_star, xi_w, log_T_star, epsilon = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )

    # Sound waves Jacobian: (N_freq, 4) for [log_K, log_R, xi_w, log_T]
    J_sw = d1pt_sound_waves(
        freq, jnp.array([log_K, log_R_H_star, xi_w, log_T_star])
    )  # (N_freq, 4)
    # Pad with zero column for epsilon
    N_freq = J_sw.shape[0]
    J_sw_full = jnp.concatenate([J_sw, jnp.zeros((N_freq, 1))], axis=1)  # (N_freq, 5)

    # Turbulence Jacobian: (N_freq, 3) for [log_Omega_s, log_R, log_T]
    # log_Omega_s = log_K + log10(epsilon)
    log_Omega_s = log_K + jnp.log10(epsilon)
    J_mhd = d1pt_turbulence(
        freq, jnp.array([log_Omega_s, log_R_H_star, log_T_star])
    )  # (N_freq, 3)

    # Chain rule: d([log_Omega_s, log_R, log_T]) / d(
    # [log_K, log_R, xi_w, log_T, epsilon])
    ln10 = jnp.log(10.0)
    d_inner_d_outer = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0 / (ln10 * epsilon)],  # log_Omega_s  # noqa: E501
            [0.0, 1.0, 0.0, 0.0, 0.0],  # log_R_H_star
            [0.0, 0.0, 0.0, 1.0, 0.0],  # log_T_star
        ]
    )  # (3, 5)
    J_mhd_full = J_mhd @ d_inner_d_outer  # (N_freq, 5)

    return J_sw_full + J_mhd_full


pt_plasma_model = ut.Signal_model(
    "pt_plasma",
    pt_plasma,
    dtemplate=d1pt_plasma,
    model_label="PT Plasma (SW + Turbulence)",
    parameter_names=[
        "log_K",
        "log_R_H_star",
        "xi_w",
        "log_T_star",
        "epsilon",
    ],
    parameter_labels=[
        r"$\log_{10}K$",
        r"$\log_{10}(R_* H_*)$",
        r"$\xi_w$",
        r"$\log_{10}(T_*/\mathrm{GeV})$",
        r"$\epsilon$",
    ],
    prior={
        "log_K": {"prior_type": "uniform", "minimum": -4.0, "maximum": 0.0},
        "log_R_H_star": {"prior_type": "uniform", "minimum": -3.0, "maximum": 0.0},
        "xi_w": {"prior_type": "uniform", "minimum": 0.01, "maximum": 0.99},
        "log_T_star": {"prior_type": "uniform", "minimum": -2.0, "maximum": 4.0},
        "epsilon": {"prior_type": "uniform", "minimum": 0.0, "maximum": 1.0},
    },
)
