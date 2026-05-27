"""
Galactic compact binary foreground template (old / 6-parameter form).

Implements Eq. 14 of Mangiagli et al. 2020 (arXiv:1803.01944).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def galactic_binaries_old(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Galactic binary foreground (Eq. 14 of arXiv:1803.01944).

    Omega(f) = 10^log_A * f^(2/3)
               * exp(-f^alpha - beta * f * sin(kappa * f))
               * (1 + tanh(gamma * (fk - f)))

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, alpha, beta, kappa, gamma, fk].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_galactic, alpha, beta, kappa, gamma, fk = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
    )
    fvec = jnp.asarray(freq)
    return (
        10.0**log_galactic
        * fvec ** (2.0 / 3.0)
        * jnp.exp(-(fvec**alpha) - beta * fvec * jnp.sin(kappa * fvec))
        * (1.0 + jnp.tanh(gamma * (fk - fvec)))
    )


def d1galactic_binaries_old(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``galactic_binaries_old`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, alpha, beta, kappa, gamma, fk].
    Returns:
        jax.Array of shape (N_freq, 6).
    """
    _, alpha, beta, kappa, gamma, fk = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
    )
    fvec = jnp.asarray(freq)
    t = jnp.tanh(gamma * (fk - fvec))
    model = galactic_binaries_old(fvec, pars)
    ln10 = jnp.log(10.0)

    d_logA = model * ln10
    d_alpha = model * (-(fvec**alpha) * jnp.log(fvec))
    d_beta = model * (-fvec * jnp.sin(kappa * fvec))
    d_kappa = model * (-beta * fvec**2 * jnp.cos(kappa * fvec))
    # T = 1 + tanh(gamma*(fk-f)),  d(T)/d(gamma) = (1-t^2)*(fk-f)
    # d(model)/d(gamma) = model * (1-t^2)/(1+t) * (fk-f) = model * (1-t) * (fk-f)
    d_gamma = model * (1.0 - t) * (fk - fvec)
    # d(T)/d(fk) = (1-t^2)*gamma
    # d(model)/d(fk) = model * (1-t) * gamma
    d_fk = model * (1.0 - t) * gamma

    return jnp.stack([d_logA, d_alpha, d_beta, d_kappa, d_gamma, d_fk], axis=1)


galactic_binaries_old_model = ut.Signal_model(
    "galactic_binaries_old",
    galactic_binaries_old,
    dtemplate=d1galactic_binaries_old,
    model_label="Galactic Binaries (old)",
    parameter_names=["log_galactic", "alpha", "beta", "kappa", "gamma", "fk"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{\rm Gal})$",
        r"$\alpha$",
        r"$\beta$",
        r"$\kappa$",
        r"$\gamma$",
        r"$f_k\,[\mathrm{Hz}]$",
    ],
    prior={
        "log_galactic": {"min": -14.0, "max": -5.0},
        "alpha": {"min": 0.01, "max": 5.0},
        "beta": {"min": 0.0, "max": 1000.0},
        "kappa": {"min": 0.0, "max": 1000.0},
        "gamma": {"min": 0.0, "max": 5000.0},
        "fk": {"min": 1e-5, "max": 1e-2},
    },
)


# ── Amplitude-only variant (fiducial shape from arXiv:1803.01944) ─────────────

# Fiducial shape parameters from Eq. 14 of arXiv:1803.01944 (Table 1)
_ALPHA_FID = 0.138
_BETA_FID = 221.0
_KAPPA_FID = 521.0
_GAMMA_FID = 1680.0
_FK_FID = 0.00113

_FIDUCIAL_SHAPE_PARS_OLD = jnp.array(
    [_ALPHA_FID, _BETA_FID, _KAPPA_FID, _GAMMA_FID, _FK_FID]
)


def galactic_binaries_old_A(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Old galactic binary foreground with shape fixed to the fiducial values
    (Eq. 14 of arXiv:1803.01944, Table 1).

    Free parameter: log10 amplitude only.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq,).
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_SHAPE_PARS_OLD])
    return galactic_binaries_old(freq, full_pars)


def d1galactic_binaries_old_A(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``galactic_binaries_old_A`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
    Returns:
        jax.Array of shape (N_freq, 1): d/d(log_galactic) only.
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_SHAPE_PARS_OLD])
    return d1galactic_binaries_old(freq, full_pars)[:, :1]


galactic_binaries_old_A_model = ut.Signal_model(
    "galactic_binaries_old_A",
    galactic_binaries_old_A,
    dtemplate=d1galactic_binaries_old_A,
    model_label="Galactic Binaries old (amplitude only)",
    parameter_names=["log_galactic"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_{\rm Gal})$"],
    prior={"log_galactic": {"min": -14.0, "max": -5.0}},
)
