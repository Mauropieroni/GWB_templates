"""
Cosmic String Model II (1 parameter) and Abelian Higgs Model II (2 parameters).

Model II uses a precomputed 2D data grid over (log_Gmu, log10_frequency) and
evaluates h^2 * Omega_GW via JAX bilinear interpolation so that JAX automatic
differentiation works and analytical gradients can be verified against it.

Analytical gradient derivation (chain rule for bilinear interpolation):

  Let S(log_Gmu, log10_f) = bilinear_interp(log10_omega_grid, ix, iy)
  where  ix = (log_Gmu  - gmu_min)  / (gmu_max  - gmu_min)  * (N_gmu  - 1)
         iy = (log10_f  - freq_min) / (freq_max - freq_min) * (N_freq - 1)

  h2_Omega = 10^S

  d(h2_Omega)/d(log_Gmu)
    = ln(10) * h2_Omega * dS/d(log_Gmu)
    = ln(10) * h2_Omega * dS/d(ix) * d(ix)/d(log_Gmu)

  dS/d(ix) = (1 - ty) * (g[kx+1, ky] - g[kx, ky])
           +       ty  * (g[kx+1, ky+1] - g[kx, ky+1])

  where kx = floor(ix), tx = ix - kx, ky = floor(iy), ty = iy - ky
  and g[i, j] are the log10_omega grid values.

  d(ix)/d(log_Gmu) = (N_gmu - 1) / (gmu_max - gmu_min)

For abelian_higgs_model_ii = 10^logf * h2_Omega_II:
  d/d(log_Gmu) = 10^logf * d(h2_Omega_II)/d(log_Gmu)
  d/d(logf)    = ln(10) * h2_Omega_AH

Reference: arXiv:1909.00819 (BOS P_n loop distribution);
          arXiv:2405.03740 (GW from cosmic strings in LISA: reconstruction
          pipeline and physics interpretation).
"""

import os
from collections.abc import Sequence

import jax
import numpy as np
import jax.numpy as jnp

from gwb_templates import utils as ut

# ── Load precomputed data grid ────────────────────────────────────────────────

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "Model-II_BOS-loggrid.dat",
)

_data_np = np.loadtxt(_DATA_PATH)
# Row 0, cols 1..  → log10_frequency axis
# Col 0, rows 1..  → log_Gmu axis
# Submatrix [1:,1:] → log10(h^2 Omega_GW)
_gmu_axis = jnp.array(_data_np[1:, 0])  # shape (N_GMU,)
_freq_axis = jnp.array(_data_np[0, 1:])  # shape (N_FREQ_GRID,)
_log10_omega = jnp.array(_data_np[1:, 1:])  # shape (N_GMU, N_FREQ_GRID)

_N_GMU = int(_gmu_axis.shape[0])
_N_FREQ_GRID = int(_freq_axis.shape[0])

_LOG_GMU_MIN = float(_gmu_axis[0])
_LOG_GMU_MAX = float(_gmu_axis[-1])


# ── JAX bilinear interpolation primitives ────────────────────────────────────


def _to_frac_ix(val: float | jax.Array, axis: jax.Array) -> jax.Array:
    """
    Convert a physical value to a fractional grid index along ``axis``.

    Args:
        val: Physical value(s) to convert (scalar or 1-D array).
        axis: 1-D JAX array of evenly-spaced grid coordinates.
    Returns:
        Fractional index (0-based) within the grid.
    """
    n = axis.shape[0]
    return (val - axis[0]) / (axis[-1] - axis[0]) * (n - 1)


def _bilinear_corners(ix: jax.Array, iy: jax.Array) -> tuple[jax.Array, ...]:
    """
    Compute bilinear corner indices, weights, and grid values.

    Args:
        ix: Fractional gmu grid index (JAX scalar).
        iy: Fractional freq grid indices, one per query frequency (1-D JAX array).
    Returns:
        Tuple (tx, ty, g00, g01, g10, g11) — all JAX arrays of shape (n_query,).
    """
    # ── x (gmu) axis ──────────────────────────────────────────────────────────
    kx = jnp.clip(
        jnp.floor(jnp.clip(ix, 0.0, _N_GMU - 1.0)).astype(jnp.int32),
        0,
        _N_GMU - 2,
    )
    tx = jnp.clip(ix - kx, 0.0, 1.0)

    # ── y (freq) axis ─────────────────────────────────────────────────────────
    ky = jnp.clip(
        jnp.floor(jnp.clip(iy, 0.0, _N_FREQ_GRID - 1.0)).astype(jnp.int32),
        0,
        _N_FREQ_GRID - 2,
    )
    ty = jnp.clip(iy - ky, 0.0, 1.0)

    # ── Corner grid values ────────────────────────────────────────────────────
    # Dynamic row indexing: _log10_omega[kx] and _log10_omega[kx + 1]
    row0 = _log10_omega[kx]  # shape (N_FREQ_GRID,)
    row1 = _log10_omega[kx + 1]  # shape (N_FREQ_GRID,)
    g00 = row0[ky]
    g01 = row0[ky + 1]
    g10 = row1[ky]
    g11 = row1[ky + 1]

    return tx, ty, g00, g01, g10, g11


def _bilinear_eval(ix: jax.Array, iy: jax.Array) -> jax.Array:
    """
    Bilinear interpolation of log10(h^2 Omega) at fractional indices.

    Args:
        ix: Fractional gmu grid index (JAX scalar).
        iy: Fractional freq grid indices (1-D JAX array).
    Returns:
        Interpolated log10(h^2 * Omega_GW) at each query frequency.
    """
    tx, ty, g00, g01, g10, g11 = _bilinear_corners(ix, iy)
    return (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )


def _bilinear_dS_dix(ix: jax.Array, iy: jax.Array) -> jax.Array:
    """
    Analytical dS/d(ix) for the bilinear interpolation (see module docstring).

    Matches JAX's JVP of ``_bilinear_eval`` w.r.t. ix to machine precision.

    Args:
        ix: Fractional gmu grid index (JAX scalar).
        iy: Fractional freq grid indices (1-D JAX array).
    Returns:
        Derivative of interpolated S w.r.t. ix at each query frequency.
    """
    _, ty, g00, g01, g10, g11 = _bilinear_corners(ix, iy)
    return (1.0 - ty) * (g10 - g00) + ty * (g11 - g01)


# ── Public template functions ─────────────────────────────────────────────────


def cosmic_string_model_ii(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Cosmic String Model II (arXiv:1909.00819, BOS P_n).

    1-parameter model evaluated from a precomputed data grid via bilinear
    interpolation.  The template is JAX-differentiable.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu].  Grid range: -18 ≤ log_Gmu ≤ -9.5.
    Returns:
        jax.Array of h^2 * Omega_GW at each frequency.
    """
    log_Gmu = pars[0]
    log10_f = jnp.log10(freq)
    ix = _to_frac_ix(log_Gmu, _gmu_axis)
    iy = _to_frac_ix(log10_f, _freq_axis)
    return 10.0 ** _bilinear_eval(ix, iy)


def d1_cosmic_string_model_ii(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Analytical first derivative of cosmic_string_model_ii w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu].
    Returns:
        jax.Array of shape (len(freq), 1): gradient w.r.t. [log_Gmu].
    """
    log_Gmu = pars[0]
    log10_f = jnp.log10(freq)
    ix = _to_frac_ix(log_Gmu, _gmu_axis)
    iy = _to_frac_ix(log10_f, _freq_axis)

    S = _bilinear_eval(ix, iy)
    h2_omega = 10.0**S

    # d(ix)/d(log_Gmu) = (N_GMU - 1) / (gmu_max - gmu_min)
    d_ix_d_log_Gmu = (_N_GMU - 1) / (_gmu_axis[-1] - _gmu_axis[0])
    dS_dix = _bilinear_dS_dix(ix, iy)

    grad = jnp.log(10.0) * h2_omega * dS_dix * d_ix_d_log_Gmu
    return grad[:, None]  # shape (n_freq, 1)


cosmic_string_model_ii_model = ut.Signal_model(
    "cosmic_string_model_ii",
    cosmic_string_model_ii,
    dtemplate=d1_cosmic_string_model_ii,
    model_label="Cosmic String Model II",
    parameter_names=["log_Gmu"],
    parameter_labels=[r"$\log_{10}(G\mu)$"],
    prior={"log_Gmu": {"min": _LOG_GMU_MIN, "max": _LOG_GMU_MAX}},
)


def abelian_higgs_model_ii(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Abelian Higgs Model II.

    Scales the BOS Model II spectrum by an overall amplitude 10^logf, allowing
    a continuous interpolation between the Nambu-Goto and Abelian Higgs limits.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu, log10 f_NG].
    Returns:
        jax.Array of h^2 * Omega_GW at each frequency.
    """
    logf = pars[1]
    return 10.0**logf * cosmic_string_model_ii(freq, pars[:1])


def d1_abelian_higgs_model_ii(freq: np.ndarray, pars: Sequence[float]) -> jax.Array:
    """
    Analytical first derivative of abelian_higgs_model_ii w.r.t. pars.

    Args:
        freq: Frequency grid [Hz].
        pars: [log10 Gmu, log10 f_NG].
    Returns:
        jax.Array of shape (len(freq), 2): gradients w.r.t. [log_Gmu, logf].
    """
    logf = pars[1]

    h2_omega_ii = cosmic_string_model_ii(freq, pars[:1])
    h2_omega_ah = 10.0**logf * h2_omega_ii

    # d/d(log_Gmu): scale Model II gradient by 10^logf
    d_d_log_Gmu = 10.0**logf * d1_cosmic_string_model_ii(freq, pars[:1])[:, 0]
    # d/d(logf): ln(10) * 10^logf * h2_omega_ii = ln(10) * h2_omega_ah
    d_d_logf = jnp.log(10.0) * h2_omega_ah

    return jnp.stack([d_d_log_Gmu, d_d_logf], axis=-1)  # shape (n_freq, 2)


abelian_higgs_model_ii_model = ut.Signal_model(
    "abelian_higgs_model_ii",
    abelian_higgs_model_ii,
    dtemplate=d1_abelian_higgs_model_ii,
    model_label="Abelian Higgs Model II",
    parameter_names=["log_Gmu", "logf"],
    parameter_labels=[
        r"$\log_{10}(G\mu)$",
        r"$\log_{10}(f_\mathrm{NG})$",
    ],
    prior={
        "log_Gmu": {"min": _LOG_GMU_MIN, "max": _LOG_GMU_MAX},
        "logf": {"min": -3.0, "max": 3.0},
    },
)
