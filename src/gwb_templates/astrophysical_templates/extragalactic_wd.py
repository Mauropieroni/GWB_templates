"""
Extragalactic white-dwarf binary foreground template.

Implements Eq. B1 of Mangiagli et al. 2025 (arXiv:2506.18390).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def extragalactic_wd(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Extragalactic WD binary foreground (Eq. B1 of arXiv:2506.18390).

    Omega(f) = 10^log_A * x^(-alpha1) * H^{(alpha1 - alpha2) * delta}

    where  x = f / f_knee,  H = 0.5 * (1 + x^{1/delta}).

    At low frequencies the slope approaches ``-alpha1``; at high frequencies
    it approaches ``-alpha2``.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, f_knee [Hz], delta, alpha1, alpha2].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_A, f_knee, delta, alpha1, alpha2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fvec = jnp.asarray(freq)
    x = fvec / f_knee
    low_pl = x ** (-alpha1)
    H = 0.5 * (1.0 + x ** (1.0 / delta))
    power = (alpha1 - alpha2) * delta
    return 10.0**log_A * low_pl * H**power


def d1extragalactic_wd(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_wd`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, f_knee [Hz], delta, alpha1, alpha2].
    Returns:
        jax.Array of shape (N_freq, 5).
    """
    _, f_knee, delta, alpha1, alpha2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fvec = jnp.asarray(freq)
    x = fvec / f_knee
    x_inv_delta = x ** (1.0 / delta)
    factor = 1.0 + x_inv_delta  # = 2 * H
    H = 0.5 * factor
    power = (alpha1 - alpha2) * delta
    model = extragalactic_wd(fvec, pars)

    d_logA = model * jnp.log(10.0)

    # d(ln model)/d(f_knee) = (1/f_knee) * (alpha1 - power*(0.5/delta)*x_inv_delta/H)
    d_fknee = model / f_knee * (alpha1 - (power / H) * (0.5 / delta) * x_inv_delta)

    # d(ln model)/d(delta) — see derivation in notes
    # = (alpha1-alpha2) * [ln(H) - x_inv_delta*ln(x) / (delta*factor)]
    log_derivative_delta = (alpha1 - alpha2) * (
        jnp.log(H) - x_inv_delta * jnp.log(x) / (delta * factor)
    )
    d_delta = model * log_derivative_delta

    # d(ln model)/d(alpha1) = -ln(x) + delta*ln(H)
    d_alpha1 = model * (-jnp.log(x) + delta * jnp.log(H))

    # d(ln model)/d(alpha2) = delta * (ln(2) - ln(factor))
    d_alpha2 = model * delta * (jnp.log(2.0) - jnp.log(factor))

    return jnp.stack([d_logA, d_fknee, d_delta, d_alpha1, d_alpha2], axis=1)


extragalactic_wd_model = ut.Signal_model(
    "extragalactic_wd",
    extragalactic_wd,
    dtemplate=d1extragalactic_wd,
    model_label="Extragalactic WD Binaries",
    parameter_names=[
        "log_extragalactic_wd",
        "f_knee_wd",
        "delta_wd",
        "alpha1_wd",
        "alpha2_wd",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{\rm EWD})$",
        r"$f_{\rm knee}$",
        r"$\delta$",
        r"$\alpha_1$",
        r"$\alpha_2$",
    ],
    prior={
        "log_extragalactic_wd": {"min": -20.0, "max": -5.0},
        "f_knee_wd": {"min": 1e-5, "max": 0.1},
        "delta_wd": {"min": 0.1, "max": 10.0},
        "alpha1_wd": {"min": -5.0, "max": 5.0},
        "alpha2_wd": {"min": -5.0, "max": 5.0},
    },
)


# ── Amplitude-only variant (shape fixed at fiducials) ────────────────────────

_WD_FID_LOG_A: float = -11.301029995663981
_WD_FID_F_KNEE: float = 7e-3
_WD_FID_DELTA: float = 0.24
_WD_FID_ALPHA1: float = -0.72
_WD_FID_ALPHA2: float = 2.43


def extragalactic_wd_A(
    freq: ArrayLike,
    pars: ParamLike,
    *,
    f_knee: float = _WD_FID_F_KNEE,
    delta: float = _WD_FID_DELTA,
    alpha1: float = _WD_FID_ALPHA1,
    alpha2: float = _WD_FID_ALPHA2,
) -> jax.Array:
    """
    Extragalactic WD binary foreground — amplitude-only variant.

    Shape parameters are fixed at the fiducial values from arXiv:2506.18390
    but can be overridden via keyword arguments.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        f_knee: Knee frequency [Hz] (default 7e-3).
        delta: Transition width (default 0.24).
        alpha1: Low-frequency slope (default -0.72).
        alpha2: High-frequency slope (default 2.43).
    Returns:
        jax.Array of shape (N_freq,).
    """
    full_pars = jnp.concatenate(
        [jnp.atleast_1d(pars[0]), jnp.array([f_knee, delta, alpha1, alpha2])]
    )
    return extragalactic_wd(freq, full_pars)


def d1extragalactic_wd_A(
    freq: ArrayLike,
    pars: ParamLike,
    *,
    f_knee: float = _WD_FID_F_KNEE,
    delta: float = _WD_FID_DELTA,
    alpha1: float = _WD_FID_ALPHA1,
    alpha2: float = _WD_FID_ALPHA2,
) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_wd_A`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        f_knee: Knee frequency [Hz] (default 7e-3).
        delta: Transition width (default 0.24).
        alpha1: Low-frequency slope (default -0.72).
        alpha2: High-frequency slope (default 2.43).
    Returns:
        jax.Array of shape (N_freq, 1): d/d(log_A) only.
    """
    full_pars = jnp.concatenate(
        [jnp.atleast_1d(pars[0]), jnp.array([f_knee, delta, alpha1, alpha2])]
    )
    return d1extragalactic_wd(freq, full_pars)[:, :1]


extragalactic_wd_A_model = ut.Signal_model(
    "extragalactic_wd_A",
    extragalactic_wd_A,
    dtemplate=d1extragalactic_wd_A,
    model_label="Extragalactic WD Binaries (amplitude only)",
    parameter_names=["log_extragalactic_wd"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_{\rm EWD})$"],
    prior={"log_extragalactic_wd": {"min": -20.0, "max": -5.0}},
)
