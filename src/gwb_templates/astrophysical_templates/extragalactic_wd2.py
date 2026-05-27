"""
Alternative extragalactic white-dwarf binary foreground template.

Double-transition broken power law with a fixed 2/3 mid-slope.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)

_F0 = 1e-3  # fixed reference frequency [Hz]
_MID_SLOPE = 2.0 / 3.0
_DELTA = 2.0


def extragalactic_wd2(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Alternative extragalactic WD binary foreground.

    Double-transition broken power law:

    Omega(f) = 10^log_A * (f/f0)^(2/3)
               * (1 + (f_low/f)^2)^{(alpha_low - 2/3) / 2}
               * (1 + (f/f_high)^2)^{(alpha_high - 2/3) / 2}

    with fixed f0 = 1e-3 Hz, mid_slope = 2/3, delta = 2.

    The spectral slope approaches ``alpha_low`` at f << f_low,
    ``2/3`` between the two knees, and ``alpha_high`` at f >> f_high.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, f_low [Hz], f_high [Hz], alpha_low, alpha_high].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_A, f_low, f_high, alpha_low, alpha_high = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fvec = jnp.asarray(freq)
    L = 1.0 + (f_low / fvec) ** _DELTA
    H = 1.0 + (fvec / f_high) ** _DELTA
    low_term = L ** ((alpha_low - _MID_SLOPE) / _DELTA)
    high_term = H ** ((alpha_high - _MID_SLOPE) / _DELTA)
    return 10.0**log_A * (fvec / _F0) ** _MID_SLOPE * low_term * high_term


def d1extragalactic_wd2(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_wd2`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude, f_low [Hz], f_high [Hz], alpha_low, alpha_high].
    Returns:
        jax.Array of shape (N_freq, 5).
    """
    _, f_low, f_high, alpha_low, alpha_high = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
    )
    fvec = jnp.asarray(freq)
    L = 1.0 + (f_low / fvec) ** _DELTA
    H = 1.0 + (fvec / f_high) ** _DELTA
    model = extragalactic_wd2(fvec, pars)

    d_logA = model * jnp.log(10.0)

    # d(ln model)/d(f_low) = (alpha_low - mid_slope)/f_low * (f_low/f)^delta / L
    d_flow = model * (alpha_low - _MID_SLOPE) / f_low * (f_low / fvec) ** _DELTA / L

    # d(ln model)/d(f_high) = -(alpha_high - mid_slope)/f_high * (f/f_high)^delta / H
    d_fhigh = (
        model * (-(alpha_high - _MID_SLOPE)) / f_high * (fvec / f_high) ** _DELTA / H
    )

    # d(ln model)/d(alpha_low) = (1/delta) * ln(L)
    d_alpha_low = model * (1.0 / _DELTA) * jnp.log(L)

    # d(ln model)/d(alpha_high) = (1/delta) * ln(H)
    d_alpha_high = model * (1.0 / _DELTA) * jnp.log(H)

    return jnp.stack([d_logA, d_flow, d_fhigh, d_alpha_low, d_alpha_high], axis=1)


extragalactic_wd2_model = ut.Signal_model(
    "extragalactic_wd2",
    extragalactic_wd2,
    dtemplate=d1extragalactic_wd2,
    model_label="Extragalactic WD Binaries (alt.)",
    parameter_names=[
        "log_extragalactic_wd2",
        "f_low_wd2",
        "f_high_wd2",
        "alpha_low_wd2",
        "alpha_high_wd2",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$",
        r"$f_{\rm low}$",
        r"$f_{\rm high}$",
        r"$\alpha_{\rm low}$",
        r"$\alpha_{\rm high}$",
    ],
    prior={
        "log_extragalactic_wd2": {"min": -20.0, "max": -5.0},
        "f_low_wd2": {"min": 1e-5, "max": 0.1},
        "f_high_wd2": {"min": 1e-4, "max": 1.0},
        "alpha_low_wd2": {"min": -5.0, "max": 5.0},
        "alpha_high_wd2": {"min": -5.0, "max": 5.0},
    },
)


# ── Amplitude-only variant (shape fixed at fiducials) ────────────────────────

_WD2_FID_LOG_A: float = -10.39794
_WD2_FID_F_LOW: float = 3e-4
_WD2_FID_F_HIGH: float = 1.5e-2
_WD2_FID_ALPHA_LOW: float = 0.0
_WD2_FID_ALPHA_HIGH: float = -5.5


def extragalactic_wd2_A(
    freq: ArrayLike,
    pars: ParamLike,
    *,
    f_low: float = _WD2_FID_F_LOW,
    f_high: float = _WD2_FID_F_HIGH,
    alpha_low: float = _WD2_FID_ALPHA_LOW,
    alpha_high: float = _WD2_FID_ALPHA_HIGH,
) -> jax.Array:
    """
    Alternative extragalactic WD binary foreground — amplitude-only variant.

    Shape parameters are fixed at the fiducial values but can be overridden
    via keyword arguments.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        f_low: Low-frequency knee [Hz] (default 3e-4).
        f_high: High-frequency knee [Hz] (default 1.5e-2).
        alpha_low: Low-frequency slope (default 0.0).
        alpha_high: High-frequency slope (default -5.5).
    Returns:
        jax.Array of shape (N_freq,).
    """
    full_pars = jnp.concatenate(
        [jnp.atleast_1d(pars[0]), jnp.array([f_low, f_high, alpha_low, alpha_high])]
    )
    return extragalactic_wd2(freq, full_pars)


def d1extragalactic_wd2_A(
    freq: ArrayLike,
    pars: ParamLike,
    *,
    f_low: float = _WD2_FID_F_LOW,
    f_high: float = _WD2_FID_F_HIGH,
    alpha_low: float = _WD2_FID_ALPHA_LOW,
    alpha_high: float = _WD2_FID_ALPHA_HIGH,
) -> jax.Array:
    """
    Analytical Jacobian of ``extragalactic_wd2_A`` w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 amplitude].
        f_low: Low-frequency knee [Hz] (default 3e-4).
        f_high: High-frequency knee [Hz] (default 1.5e-2).
        alpha_low: Low-frequency slope (default 0.0).
        alpha_high: High-frequency slope (default -5.5).
    Returns:
        jax.Array of shape (N_freq, 1): d/d(log_A) only.
    """
    full_pars = jnp.concatenate(
        [jnp.atleast_1d(pars[0]), jnp.array([f_low, f_high, alpha_low, alpha_high])]
    )
    return d1extragalactic_wd2(freq, full_pars)[:, :1]


extragalactic_wd2_A_model = ut.Signal_model(
    "extragalactic_wd2_A",
    extragalactic_wd2_A,
    dtemplate=d1extragalactic_wd2_A,
    model_label="Extragalactic WD Binaries alt. (amplitude only)",
    parameter_names=["log_extragalactic_wd2"],
    parameter_labels=[r"$\log_{10}(h^2\,\Omega_{\rm EWD2})$"],
    prior={"log_extragalactic_wd2": {"min": -20.0, "max": -5.0}},
)
