"""
Extragalactic compact binary merger foreground template.

Power-law spectrum with the 2/3 spectral index expected from GW emission
during the inspiral phase of compact binary mergers integrated over redshift.
Three variants are provided:
  - ``extragalactic``: 1-parameter model with tilt fixed to 2/3.
  - ``extragalactic_sobbh_bns``: 2-parameter model with free tilt (log_A + tilt).
  - ``extragalactic_sobbh_bns_A``: 1-parameter model (log_A, tilt fixed to 2/3).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike


# ── Extragalactic SOBBH + BNS (free spectral index) ─────────────────────────


def extragalactic_sobbh_bns(
    freq: ArrayLike,
    pars: ParamLike,
    ref_freq: float = 1e-3,
) -> jax.Array:
    """
    Extragalactic SOBBH+BNS foreground: Omega(f) = 10^A * (f / f_ref)^tilt.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, tilt].
        ref_freq: Reference frequency [Hz] (default 1 mHz).
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_A, tilt = pars[0], pars[1]
    return 10.0**log_A * (jnp.asarray(freq) / ref_freq) ** tilt


def d1extragalactic_sobbh_bns(
    freq: ArrayLike,
    pars: ParamLike,
    ref_freq: float = 1e-3,
) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_sobbh_bns`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, tilt].
        ref_freq: Reference frequency [Hz] (default 1 mHz).
    Returns:
        jax.Array of shape (N_freq, 2):
            col 0: d/d(log_A) = model * ln(10)
            col 1: d/d(tilt)  = model * ln(f / f_ref)
    """
    fvec = jnp.asarray(freq)
    model = extragalactic_sobbh_bns(fvec, pars, ref_freq=ref_freq)
    d_logA = model * jnp.log(10.0)
    d_tilt = model * jnp.log(fvec / ref_freq)
    return jnp.stack([d_logA, d_tilt], axis=1)


extragalactic_sobbh_bns_model = ut.Signal_model(
    "extragalactic_sobbh_bns",
    extragalactic_sobbh_bns,
    dtemplate=d1extragalactic_sobbh_bns,
    model_label="Extragalactic SOBBH+BNS",
    parameter_names=["log_extragalactic_sobbh_bns", "tilt_sobbh_bns"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
        r"$\alpha_{\rm EG}$",
    ],
    prior={
        "log_extragalactic_sobbh_bns": {"min": -20.0, "max": -5.0},
        "tilt_sobbh_bns": {"min": -3.0, "max": 5.0},
    },
)


# ── Amplitude-only variant (tilt fixed to 2/3) ────────────────────────────────

_TILT_SOBBH_BNS_FID = 2.0 / 3.0
_FIDUCIAL_TILT_PARS = jnp.array([_TILT_SOBBH_BNS_FID])


def extragalactic_sobbh_bns_A(
    freq: ArrayLike,
    pars: ParamLike,
    ref_freq: float = 1e-3,
) -> jax.Array:
    """
    Extragalactic SOBBH+BNS foreground with tilt fixed to 2/3.

    Free parameter: log10 amplitude only.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        ref_freq: Reference frequency [Hz] (default 1 mHz).
    Returns:
        jax.Array of shape (N_freq,).
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_TILT_PARS])
    return extragalactic_sobbh_bns(freq, full_pars, ref_freq=ref_freq)


def d1extragalactic_sobbh_bns_A(
    freq: ArrayLike,
    pars: ParamLike,
    ref_freq: float = 1e-3,
) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_sobbh_bns_A`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        ref_freq: Reference frequency [Hz] (default 1 mHz).
    Returns:
        jax.Array of shape (N_freq, 1): d/d(log_A) only.
    """
    full_pars = jnp.concatenate([jnp.atleast_1d(pars[0]), _FIDUCIAL_TILT_PARS])
    return d1extragalactic_sobbh_bns(freq, full_pars, ref_freq=ref_freq)[:, :1]


extragalactic_sobbh_bns_A_model = ut.Signal_model(
    "extragalactic_sobbh_bns_A",
    extragalactic_sobbh_bns_A,
    dtemplate=d1extragalactic_sobbh_bns_A,
    model_label="Extragalactic SOBBH+BNS (amplitude only)",
    parameter_names=["log_extragalactic_sobbh_bns"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_{\rm EG})$"],
    prior={"log_extragalactic_sobbh_bns": {"min": -20.0, "max": -5.0}},
)
