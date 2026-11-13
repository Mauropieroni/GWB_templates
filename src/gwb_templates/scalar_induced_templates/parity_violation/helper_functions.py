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


def _load_archive(path: Path) -> tuple[np.ndarray, ...]:
    """
    Load and preprocess the tabulated spectrum using NumPy.

    The archive contains four signed spectrum components tabulated on an
    ``n2`` by frequency grid. The component magnitudes are converted to
    base-10 logarithms for interpolation, while their signs are retained
    separately.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.npz`` archive containing ``f_grid``, ``n2_grid`` and
        the four component arrays.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The log-frequency grid, ``n2`` grid, log-magnitude component values,
        and component signs, in that order.

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
    return (
        log_f_grid,
        n2_grid,
        log_values,
        signs,
    )


(
    _LOG_F_GRID,
    _N2_GRID,
    _LOG_VALUES,
    _SIGNS,
) = _load_archive(_ARCHIVE_PATH)
FREQUENCY_RANGE = (_LOG_F_GRID[0], _LOG_F_GRID[-1])
N2_RANGE = (_N2_GRID[0], _N2_GRID[-1])


_JAX_LOG_F_GRID = jnp.asarray(_LOG_F_GRID, dtype=jnp.float64)
_JAX_N2_GRID = jnp.asarray(_N2_GRID, dtype=jnp.float64)
_JAX_LOG_VALUES = jnp.asarray(_LOG_VALUES, dtype=jnp.float64)
_JAX_SIGNS = jnp.asarray(_SIGNS, dtype=jnp.float64)


def _interpolate_component_jax(
    log_values: jax.Array,
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
) -> jax.Array:
    """
    Interpolate one spectrum component in log-frequency and ``n2``.

    Frequencies are interpolated linearly in ``log10(f)`` and then in ``n2``.
    The returned values are still logarithms of component magnitudes; the
    component sign is restored by :func:`interpolate_components_jax`.

    Parameters
    ----------
    log_values : jax.Array
        Log10 magnitudes with shape ``(n2, n_frequency_grid)``.
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
    return jax.vmap(
        lambda values_at_frequency: jnp.interp(
            target_n2,
            _JAX_N2_GRID,
            values_at_frequency,
            left=0.0,
            right=0.0,
        )
    )(log_value_by_n2.T)


def interpolate_components_jax(
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
) -> jax.Array:
    """
    Return the four unnormalised interpolated spectrum components.

    The four components are evaluated using the archive tables by
    :func:`_interpolate_component_jax`. Their original signs are restored
    after interpolation. Values outside the supported ``n2`` range are zero.
    No amplitude or non-Gaussianity normalization is applied here.

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
        in_axes=(0, None, None, None),
    )(
        _JAX_LOG_VALUES,
        flat_frequencies,
        target_f_peak,
        target_n2,
    )
    components = _JAX_SIGNS[:, None] * 10.0**log_components
    in_n2_range = (target_n2 >= N2_RANGE[0]) & (target_n2 <= N2_RANGE[1])
    components = jnp.where(in_n2_range, components, 0.0)
    return components.reshape((components.shape[0],) + original_shape)


def normalization_factors_jax(
    log10_A_zeta: float,
    f_NL: float,
    tau_NL: float,
    tilde_tau_NL: float,
) -> jax.Array:
    """
    Return the four perturbative normalization factors.

    The scalar amplitude is supplied as a base-10 logarithm, while the
    non-Gaussianity parameters are supplied in linear coordinates. The
    returned factors multiply the corresponding unnormalised components in
    the order ``(IGW1, IGW2, IGW3, VGW)``.

    Parameters
    ----------
    log10_A_zeta : float
        Base-10 logarithm of the scalar amplitude ``A_zeta``.
    f_NL : float
        Bispectrum coefficient in linear coordinates.
    tau_NL : float
        Even trispectrum coefficient in linear coordinates.
    tilde_tau_NL : float
        Odd trispectrum coefficient in linear coordinates.

    Returns
    -------
    jax.Array
        One-dimensional array containing ``A_zeta**2``,
        ``A_zeta**3 * f_NL**2``, ``A_zeta**3 * tau_NL``, and
        ``A_zeta**3 * tilde_tau_NL``.
    """
    A_zeta = 10.0**log10_A_zeta
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
    f_NL: float,
    tau_NL: float,
    tilde_tau_NL: float,
) -> jax.Array:
    """
    Return derivatives of the four normalization factors.

    Derivatives are taken with respect to the parameters in the order
    ``(log10_A_zeta, f_NL, tau_NL, tilde_tau_NL)``.

    Parameters
    ----------
    log10_A_zeta : float
        Base-10 logarithm of the scalar amplitude ``A_zeta``.
    f_NL : float
        Bispectrum coefficient in linear coordinates.
    tau_NL : float
        Even trispectrum coefficient in linear coordinates.
    tilde_tau_NL : float
        Odd trispectrum coefficient in linear coordinates.

    Returns
    -------
    jax.Array
        Jacobian with shape ``(4, 4)``. Rows correspond to the four inputs and
        columns to the four normalization factors.
    """
    factors = normalization_factors_jax(
        log10_A_zeta, f_NL, tau_NL, tilde_tau_NL
    )
    log10_derivatives = jnp.log(10.0) * jnp.stack(
        [
            jnp.stack(
                [
                    2.0 * factors[0],
                    3.0 * factors[1],
                    3.0 * factors[2],
                    3.0 * factors[3],
                ]
            ),
        ]
    )
    A_zeta = 10.0**log10_A_zeta
    return jnp.concatenate(
        (
            log10_derivatives,
            jnp.stack(
                [
                    jnp.stack([0.0, 2.0 * A_zeta**3 * f_NL, 0.0, 0.0]),
                    jnp.stack([0.0, 0.0, A_zeta**3, 0.0]),
                    jnp.stack([0.0, 0.0, 0.0, A_zeta**3]),
                ]
            ),
        )
    )


def evaluate_components_jax(
    target_frequencies: jax.Array,
    target_f_peak: float,
    target_n2: float,
    log10_A_zeta: float,
    f_NL: float,
    tau_NL: float,
    tilde_tau_NL: float,
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
    f_NL : float
        Bispectrum coefficient in linear coordinates.
    tau_NL : float
        Even trispectrum coefficient in linear coordinates.
    tilde_tau_NL : float
        Odd trispectrum coefficient in linear coordinates.

    Returns
    -------
    jax.Array
        Normalized component values with shape ``(4,) + target_frequencies.shape``.
        The first axis is ordered as ``(IGW1, IGW2, IGW3, VGW)``.
    """
    frequency_axes = (1,) * target_frequencies.ndim
    normalization_factors = normalization_factors_jax(
        log10_A_zeta, f_NL, tau_NL, tilde_tau_NL
    ).reshape((4,) + frequency_axes)
    return interpolate_components_jax(target_frequencies, target_f_peak, target_n2) * (
        normalization_factors
    )
