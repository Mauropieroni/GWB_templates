"""
Shared NumPy preprocessing and JAX interpolation for parity-even or -odd SIGWs.
Some part of the computation is jaxed.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Path to precomputed signal induced by primordial BPL power spectrum
_ARCHIVE_PATH = Path(__file__).with_name("VGW_output.npz")
jax.config.update("jax_enable_x64", True)


def _endpoint_slope(log_x: np.ndarray, log_y: np.ndarray) -> np.ndarray:
    """
    Estimate a log-log power-law slope along the last axis.

    The slope is obtained by a least-squares fit of ``log_y`` against
    ``log_x`` after centering both arrays.

    Parameters
    ----------
    log_x : numpy.ndarray
        One-dimensional log-transformed coordinates used as the independent
        variable.
    log_y : numpy.ndarray
        Log-transformed values. Its last axis must have the same length as
        ``log_x``.

    Returns
    -------
    numpy.ndarray
        The fitted slope for each leading index of ``log_y``.
    """
    x_centered = log_x - np.mean(log_x)
    y_centered = log_y - np.mean(log_y, axis=-1, keepdims=True)
    return np.sum(x_centered * y_centered, axis=-1) / np.sum(x_centered**2)


def _load_archive(path: Path) -> tuple[np.ndarray, ...]:
    """
    Load and preprocess the tabulated spectrum using NumPy.

    The archive contains four signed spectrum components tabulated on an
    ``n2`` by frequency grid. The component magnitudes are converted to
    base-10 logarithms for interpolation, while their signs and endpoint
    power-law slopes are retained separately for extrapolation.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.npz`` archive containing ``f_grid``, ``n2_grid`` and
        the four component arrays.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The log-frequency grid, ``n2`` grid, log-magnitude component values,
        component signs, infrared slopes, ultraviolet slopes, ultraviolet
        log-frequency start, and ultraviolet starting values, in that order.

    Raises
    ------
    ValueError
        If the grids are not one-dimensional and positive, if component
        shapes do not match the grids, if any value is zero, or if a
        component changes sign across the archive.
    """
    archive = np.load(path)
    f_grid = np.asarray(archive["f_grid"], dtype=float)
    n2_grid = np.asarray(archive["n2_grid"], dtype=float)
    values = np.stack(
        [
            np.asarray(archive["omega_gauss"], dtype=float),
            np.asarray(archive["omega_fnl"], dtype=float),
            np.asarray(archive["omega_taunl"], dtype=float),
            np.asarray(archive["omega_tildetaunl"], dtype=float),
        ]
    )
    if f_grid.ndim != 1 or n2_grid.ndim != 1:
        raise ValueError("f_grid and n2_grid must be one-dimensional arrays")
    if values.shape[1:] != (len(n2_grid), len(f_grid)):
        raise ValueError("The spectrum matrices do not have the expected shape")
    if np.any(f_grid <= 0) or np.any(n2_grid <= 0):
        raise ValueError("The grids must contain positive values")
    if np.any(values == 0):
        raise ValueError("Logarithmic interpolation requires nonzero values")
    signs = np.sign(values[:, 0, 0])
    if not np.all(np.sign(values) == signs[:, None, None]):
        raise ValueError("A component changes sign: log|Omega| cannot be used")
    log_f_grid = np.log10(f_grid)
    log_values = np.log10(np.abs(values))
    uv_start = max(0, len(log_f_grid) - 12)
    ir_slopes = _endpoint_slope(log_f_grid[:3], log_values[:, :, :3])
    uv_slopes = _endpoint_slope(log_f_grid[uv_start:], log_values[:, :, uv_start:])
    return (
        log_f_grid,
        n2_grid,
        log_values,
        signs,
        ir_slopes,
        uv_slopes,
        np.asarray(log_f_grid[uv_start]),
        np.asarray(log_values[:, :, uv_start]),
    )


(
    _LOG_F_GRID,
    _N2_GRID,
    _LOG_VALUES,
    _SIGNS,
    _IR_SLOPES,
    _UV_SLOPES,
    _UV_LOG_F_START,
    _UV_LOG_VALUES_START,
) = _load_archive(_ARCHIVE_PATH)
FREQUENCY_RANGE = (_LOG_F_GRID[0], _LOG_F_GRID[-1])
N2_RANGE = (_N2_GRID[0], _N2_GRID[-1])


def validate_target_n2(target_n2: float) -> None:
    """Reject ``n2`` values outside the precomputed archive range."""
    if isinstance(target_n2, jax.core.Tracer):
        def check_traced_value(value: jax.Array) -> None:
            numeric_value = float(value)
            if not N2_RANGE[0] <= numeric_value <= N2_RANGE[1]:
                raise ValueError(
                    "target_n2 is outside the precomputed spectrum range "
                    f"[{N2_RANGE[0]}, {N2_RANGE[1]}]: got {numeric_value}"
                )

        jax.debug.callback(check_traced_value, target_n2)
        return
    value = float(np.asarray(target_n2))
    if not N2_RANGE[0] <= value <= N2_RANGE[1]:
        raise ValueError(
            "target_n2 is outside the precomputed spectrum range "
            f"[{N2_RANGE[0]}, {N2_RANGE[1]}]: got {value}"
        )

_JAX_LOG_F_GRID = jnp.asarray(_LOG_F_GRID, dtype=jnp.float64)
_JAX_N2_GRID = jnp.asarray(_N2_GRID, dtype=jnp.float64)
_JAX_LOG_VALUES = jnp.asarray(_LOG_VALUES, dtype=jnp.float64)
_JAX_SIGNS = jnp.asarray(_SIGNS, dtype=jnp.float64)
_JAX_IR_SLOPES = jnp.asarray(_IR_SLOPES, dtype=jnp.float64)
_JAX_UV_SLOPES = jnp.asarray(_UV_SLOPES, dtype=jnp.float64)
_JAX_UV_LOG_F_START = jnp.asarray(_UV_LOG_F_START, dtype=jnp.float64)
_JAX_UV_LOG_VALUES_START = jnp.asarray(_UV_LOG_VALUES_START, dtype=jnp.float64)


def _interpolate_component_jax(
    log_values: jax.Array,
    ir_slopes: jax.Array,
    uv_slopes: jax.Array,
    uv_log_values_start: jax.Array,
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
) -> jax.Array:
    """
    Interpolate one spectrum component in log-frequency and ``n2``.

    Frequencies inside the archive range are interpolated linearly in
    ``log10(f)`` and then in ``n2``. Frequencies below the archive range use
    the infrared endpoint slope; frequencies at or above the ultraviolet
    threshold use the ultraviolet endpoint slope. The returned values are
    still logarithms of component magnitudes; the component sign is restored
    by :func:`interpolate_components_jax`.

    Parameters
    ----------
    log_values : jax.Array
        Log10 magnitudes with shape ``(n2, n_frequency_grid)``.
    ir_slopes : jax.Array
        Infrared slopes with one value for each ``n2`` grid point.
    uv_slopes : jax.Array
        Ultraviolet slopes with one value for each ``n2`` grid point.
    uv_log_values_start : jax.Array
        Log10 magnitudes at the start of the ultraviolet extrapolation, with
        shape ``(n2,)``.
    target_frequencies : jax.Array
        Positive target frequencies in Hz, with shape ``(n_frequency,)``.
    target_f_peak : float
        Positive peak frequency in Hz used to rescale the archive frequency
        grid.
    target_n2 : float
        Target value of the spectral parameter ``n2``.

    Returns
    -------
    jax.Array
        Interpolated log10 magnitudes with shape ``(n_frequency,)``.
    """
    log_target_f = jnp.log10(target_frequencies / target_f_peak)

    def interpolate_at_n2(log_values_at_n2: jax.Array) -> jax.Array:
        return jnp.interp(log_target_f, _JAX_LOG_F_GRID, log_values_at_n2)

    log_value_by_n2 = jax.vmap(interpolate_at_n2)(log_values)
    ir_value = log_values[:, 0, None] + ir_slopes[:, None] * (
        log_target_f[None, :] - _JAX_LOG_F_GRID[0]
    )
    uv_value = uv_log_values_start[:, None] + uv_slopes[:, None] * (
        log_target_f[None, :] - _JAX_UV_LOG_F_START
    )
    log_value_by_n2 = jnp.where(
        log_target_f[None, :] < _JAX_LOG_F_GRID[0], ir_value, log_value_by_n2
    )
    log_value_by_n2 = jnp.where(
        log_target_f[None, :] >= _JAX_UV_LOG_F_START, uv_value, log_value_by_n2
    )
    return jax.vmap(
        lambda values_at_frequency: jnp.interp(
            target_n2, _JAX_N2_GRID, values_at_frequency
        )
    )(log_value_by_n2.T)


def interpolate_components_jax(
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
) -> jax.Array:
    """
    Return the four unnormalised interpolated spectrum components.

    The four components are evaluated using the archive tables and the
    endpoint extrapolation rules implemented by
    :func:`_interpolate_component_jax`. Their original signs are restored
    after interpolation. No amplitude or non-Gaussianity normalization is
    applied here.

    Parameters
    ----------
    target_frequencies : jax.Array
        Positive target frequencies in Hz, with any shape, including ``()``.
    target_f_peak : float
        Positive peak frequency in Hz used to rescale the archive grid.
    target_n2 : float
        Target value of the spectral parameter ``n2``.

    Returns
    -------
    jax.Array
        Signed component values with shape ``(4,) + target_frequencies.shape``.
        The first axis is ordered as ``(IGW1, IGW2, IGW3, VGW)``.
    """
    original_shape = target_frequencies.shape
    flat_frequencies = jnp.ravel(target_frequencies)
    log_components = jax.vmap(
        _interpolate_component_jax,
        in_axes=(0, 0, 0, 0, None, None, None),
    )(
        _JAX_LOG_VALUES,
        _JAX_IR_SLOPES,
        _JAX_UV_SLOPES,
        _JAX_UV_LOG_VALUES_START,
        flat_frequencies,
        target_f_peak,
        target_n2,
    )
    components = _JAX_SIGNS[:, None] * 10.0**log_components
    return components.reshape((components.shape[0],) + original_shape)


def normalization_factors_jax(
    log10_A_zeta: float,
    log10_f_NL: float,
    log10_tau_NL: float,
    log10_tilde_tau_NL: float,
) -> jax.Array:
    """
    Return the four perturbative normalization factors.

    The inputs are base-10 logarithms of the primordial scalar amplitude and
    the three non-Gaussianity parameters. The returned factors multiply the
    corresponding unnormalised components in the order ``(IGW1, IGW2,
    IGW3, VGW)``.

    Parameters
    ----------
    log10_A_zeta : float
        Base-10 logarithm of the scalar amplitude ``A_zeta``.
    log10_f_NL : float
        Base-10 logarithm of ``f_NL``.
    log10_tau_NL : float
        Base-10 logarithm of ``tau_NL``.
    log10_tilde_tau_NL : float
        Base-10 logarithm of ``tilde_tau_NL``.

    Returns
    -------
    jax.Array
        One-dimensional array containing ``A_zeta**2``,
        ``A_zeta**3 * f_NL**2``, ``A_zeta**3 * tau_NL``, and
        ``A_zeta**3 * tilde_tau_NL``.
    """
    A_zeta = 10.0**log10_A_zeta
    f_NL = 10.0**log10_f_NL
    tau_NL = 10.0**log10_tau_NL
    tilde_tau_NL = 10.0**log10_tilde_tau_NL
    return jnp.stack(
        [
            A_zeta**2,
            A_zeta**3 * f_NL**2,
            A_zeta**3 * tau_NL,
            A_zeta**3 * tilde_tau_NL,
        ]
    )


def normalization_factor_gradients_jax(
    log10_A_zeta: float,
    log10_f_NL: float,
    log10_tau_NL: float,
    log10_tilde_tau_NL: float,
) -> jax.Array:
    """
    Return derivatives of the four normalization factors.

    Derivatives are taken with respect to the four base-10 logarithmic
    parameters in the order ``(log10_A_zeta, log10_f_NL, log10_tau_NL,
    log10_tilde_tau_NL)``.

    Parameters
    ----------
    log10_A_zeta : float
        Base-10 logarithm of the scalar amplitude ``A_zeta``.
    log10_f_NL : float
        Base-10 logarithm of ``f_NL``.
    log10_tau_NL : float
        Base-10 logarithm of ``tau_NL``.
    log10_tilde_tau_NL : float
        Base-10 logarithm of ``tilde_tau_NL``.

    Returns
    -------
    jax.Array
        Jacobian with shape ``(4, 4)``. Rows correspond to the four logarithmic
        inputs and columns to the four normalization factors.
    """
    log10_values = jnp.log(10.0)
    factors = normalization_factors_jax(
        log10_A_zeta,
        log10_f_NL,
        log10_tau_NL,
        log10_tilde_tau_NL,
    )
    return log10_values * jnp.stack(
        [
            jnp.stack([2.0 * factors[0], 0.0, 0.0, 0.0]),
            jnp.stack([3.0 * factors[1], 2.0 * factors[1], 0.0, 0.0]),
            jnp.stack([3.0 * factors[2], 0.0, factors[2], 0.0]),
            jnp.stack([3.0 * factors[3], 0.0, 0.0, factors[3]]),
        ],
        axis=1,
    )


def evaluate_components_jax(
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
    log10_A_zeta: float,
    log10_f_NL: float,
    log10_tau_NL: float,
    log10_tilde_tau_NL: float,
) -> jax.Array:
    """
    Return normalized ``(IGW1, IGW2, IGW3, VGW)`` components.

    This combines :func:`interpolate_components_jax` with
    :func:`normalization_factors_jax`. The interpolation is performed in the
    archive coordinates and the resulting signed spectra are multiplied by
    their perturbative normalization factors.

    Parameters
    ----------
    target_frequencies : jax.Array
        Positive target frequencies in Hz, with any shape, including ``()``.
    target_f_peak : float
        Positive peak frequency in Hz used to rescale the archive grid.
    target_n2 : float
        Target value of the spectral parameter ``n2``.
    log10_A_zeta : float
        Base-10 logarithm of the scalar amplitude ``A_zeta``.
    log10_f_NL : float
        Base-10 logarithm of ``f_NL``.
    log10_tau_NL : float
        Base-10 logarithm of ``tau_NL``.
    log10_tilde_tau_NL : float
        Base-10 logarithm of ``tilde_tau_NL``.

    Returns
    -------
    jax.Array
        Normalized component values with shape ``(4,) + target_frequencies.shape``.
        The first axis is ordered as ``(IGW1, IGW2, IGW3, VGW)``.
    """
    frequency_axes = (1,) * target_frequencies.ndim
    normalization_factors = normalization_factors_jax(
        log10_A_zeta, log10_f_NL, log10_tau_NL, log10_tilde_tau_NL
    ).reshape((4,) + frequency_axes)
    return interpolate_components_jax(target_frequencies, target_f_peak, target_n2) * (
        normalization_factors
    )
