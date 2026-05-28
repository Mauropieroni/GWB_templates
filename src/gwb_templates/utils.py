# Global imports
from collections.abc import Callable
from typing import Any, TypeAlias

import os
import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np
from interpax import Interpolator1D

from collections.abc import Iterable

Array: TypeAlias = jax.Array
AnyArray: TypeAlias = jax.Array | np.ndarray
ArrayLike: TypeAlias = jtp.ArrayLike
TemplateFn: TypeAlias = Callable[..., Array]
IndexedDerivativeFn: TypeAlias = Callable[..., Array]


# Change jax config to use double precision
jax.config.update("jax_enable_x64", True)


def check_paths(paths: Iterable[str]) -> None:
    """
    Ensures that all provided directories exist.

    Args:
        paths: Iterable of directory paths to create if missing.

    Returns:
        None.
    """
    # Create each directory only if it does not already exist
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def gradient_autodiff(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> Array:
    """
    Build a vectorized first-derivative tensor.

    Args:
        npars: Number of model parameters.
        function: Callable that takes parameter index and returns dS/df.
        frequency: Frequency grid.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with parameter axis in the last position: (..., npars).

    """

    return jax.jacfwd(function, argnums=1)(frequency, parameters, *args, **kwargs)


def hessian_autodiff(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> Array:
    """
    Build a vectorized second-derivative tensor.

    Args:
        npars: Number of model parameters.
        function: Callable that takes two parameter indices and returns
            second derivatives.
        frequency: Frequency grid.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with the two parameter axes in the last two positions:
        (..., npars, npars).

    """

    # Keep parameter indices on the last two axes: (..., npars, npars)
    return jax.jacfwd(jax.jacfwd(function, argnums=1), argnums=1)(
        frequency, parameters, *args, **kwargs
    )


def make_log_log_interpolator(
    freq: AnyArray,
    compute_fn: Callable[..., AnyArray],
    *args: Any,
    n_points: int = 100,
    method: str = "linear",
) -> Callable[[AnyArray], Array]:
    """
    Pre-compute a log-log interpolator for ``compute_fn`` over [freq_min, freq_max].

    Evaluates ``compute_fn`` once on a coarse log-spaced grid, builds an
    ``interpax.Interpolator1D`` on ``log(freq)`` vs ``log(y)``, and returns a
    callable that maps any target frequency array to interpolated values via
    ``exp(spline(log(freq)))``.  Use at module level to avoid repeating the
    expensive ``compute_fn`` evaluation on every template call.

    Args:
        freq: Target frequency array.
        compute_fn: Callable ``(freq_array, *args) -> jnp.ndarray``.
        *args: Extra positional arguments forwarded to ``compute_fn``.
        n_points: Number of coarse-grid evaluation points (default 100).
        method: Interpolation method passed to ``Interpolator1D`` (default
            ``"linear"``).

    Returns:
        JAX-differentiable callable that accepts a frequency array and returns
        ``exp(spline(log(freq)))``.
    """
    xx = jnp.geomspace(freq[0], freq[-1], n_points)
    yy = compute_fn(xx, *args)
    zero_mask = yy <= 0
    safe_yy = jnp.where(zero_mask, 1.0, yy)
    log_xx = jnp.log(xx)
    _log_interp = Interpolator1D(log_xx, jnp.log(safe_yy), method=method, extrap=False)
    _zero_float = zero_mask * 1.0

    def _interpolator(freq: AnyArray) -> Array:
        out_zero = jnp.interp(jnp.log(freq), log_xx, _zero_float) > 0.5
        return jnp.where(out_zero, 0.0, jnp.exp(_log_interp(jnp.log(freq))))

    return _interpolator


def log_log_interpolate(
    freq: AnyArray,
    compute_fn: Callable[..., AnyArray],
    *args: Any,
    n_points: int = 100,
) -> Array:
    """
    Evaluate compute_fn on a coarse log-spaced grid and interpolate in log-log space.

    Builds n_points frequencies spanning [freq[0], freq[-1]], evaluates
    compute_fn(grid, *args), then interpolates log(y) vs log(freq) and returns
    exp(interp(log(freq))).

    Args:
        freq: Target frequency array.
        compute_fn: Callable(freq_array, *args) -> jnp.ndarray.
        *args: Extra positional arguments forwarded to compute_fn.
        n_points: Number of coarse-grid evaluation points (default 100).

    Returns:
        Array of the same shape as freq.
    """
    xx = jnp.geomspace(freq[0], freq[-1], n_points)
    yy = compute_fn(xx, *args)
    zero_mask = yy <= 0
    safe_yy = jnp.where(zero_mask, 1.0, yy)
    log_xx = jnp.log(xx)
    log_result = jnp.interp(jnp.log(freq), log_xx, jnp.log(safe_yy))
    out_zero = jnp.interp(jnp.log(freq), log_xx, zero_mask * 1.0) > 0.5
    return jnp.where(out_zero, 0.0, jnp.exp(log_result))
