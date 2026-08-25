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


def finite_difference_grad_theta(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    step: float = 1e-6,
    **kwargs: Any,
) -> Array:
    """
    Central finite-difference gradient with respect to model parameters.
    """
    params = np.asarray(parameters, dtype=float)
    npars = params.size

    f0 = np.asarray(function(frequency, params, *args, **kwargs), dtype=float)
    grad = np.empty(f0.shape + (npars,), dtype=float)

    for i in range(npars):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += step
        p_minus[i] -= step

        f_plus = np.asarray(function(frequency, p_plus, *args, **kwargs), dtype=float)
        f_minus = np.asarray(function(frequency, p_minus, *args, **kwargs), dtype=float)
        grad[..., i] = (f_plus - f_minus) / (2.0 * step)

    return jnp.asarray(grad)


def finite_difference_hess_theta(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    step: float = 1e-5,
    **kwargs: Any,
) -> Array:
    """
    Central finite-difference Hessian with respect to model parameters.
    """
    params = np.asarray(parameters, dtype=float)
    npars = params.size

    f0 = np.asarray(function(frequency, params, *args, **kwargs), dtype=float)
    hess = np.empty(f0.shape + (npars, npars), dtype=float)

    for i in range(npars):
        for j in range(i, npars):
            if i == j:
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[i] += step
                p_minus[i] -= step

                f_plus = np.asarray(
                    function(frequency, p_plus, *args, **kwargs), dtype=float
                )
                f_minus = np.asarray(
                    function(frequency, p_minus, *args, **kwargs), dtype=float
                )
                value = (f_plus - 2.0 * f0 + f_minus) / (step**2)
            else:
                p_pp = params.copy()
                p_pm = params.copy()
                p_mp = params.copy()
                p_mm = params.copy()
                p_pp[i] += step
                p_pp[j] += step
                p_pm[i] += step
                p_pm[j] -= step
                p_mp[i] -= step
                p_mp[j] += step
                p_mm[i] -= step
                p_mm[j] -= step

                f_pp = np.asarray(
                    function(frequency, p_pp, *args, **kwargs), dtype=float
                )
                f_pm = np.asarray(
                    function(frequency, p_pm, *args, **kwargs), dtype=float
                )
                f_mp = np.asarray(
                    function(frequency, p_mp, *args, **kwargs), dtype=float
                )
                f_mm = np.asarray(
                    function(frequency, p_mm, *args, **kwargs), dtype=float
                )
                value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step**2)

            hess[..., i, j] = value
            hess[..., j, i] = value

    return jnp.asarray(hess)


def finite_difference_df(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    step: float = 1e-6,
    **kwargs: Any,
) -> Array:
    """
    Central finite-difference first derivative with respect to frequency.
    """
    freq = np.asarray(frequency, dtype=float)

    if freq.ndim == 0:
        f_plus = np.asarray(
            function(freq + step, parameters, *args, **kwargs), dtype=float
        )
        f_minus = np.asarray(
            function(freq - step, parameters, *args, **kwargs), dtype=float
        )
        return jnp.asarray((f_plus - f_minus) / (2.0 * step))

    flat = freq.ravel()
    deriv = np.empty(flat.shape, dtype=float)
    for k, ff in enumerate(flat):
        f_plus = np.asarray(
            function(ff + step, parameters, *args, **kwargs), dtype=float
        )
        f_minus = np.asarray(
            function(ff - step, parameters, *args, **kwargs), dtype=float
        )
        deriv[k] = (f_plus - f_minus) / (2.0 * step)

    return jnp.asarray(deriv.reshape(freq.shape))


def finite_difference_d2f2(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    step: float = 1e-5,
    **kwargs: Any,
) -> Array:
    """
    Central finite-difference second derivative with respect to frequency.
    """
    freq = np.asarray(frequency, dtype=float)

    if freq.ndim == 0:
        f0 = np.asarray(function(freq, parameters, *args, **kwargs), dtype=float)
        f_plus = np.asarray(
            function(freq + step, parameters, *args, **kwargs), dtype=float
        )
        f_minus = np.asarray(
            function(freq - step, parameters, *args, **kwargs), dtype=float
        )
        return jnp.asarray((f_plus - 2.0 * f0 + f_minus) / (step**2))

    flat = freq.ravel()
    deriv2 = np.empty(flat.shape, dtype=float)
    for k, ff in enumerate(flat):
        f0 = np.asarray(function(ff, parameters, *args, **kwargs), dtype=float)
        f_plus = np.asarray(
            function(ff + step, parameters, *args, **kwargs), dtype=float
        )
        f_minus = np.asarray(
            function(ff - step, parameters, *args, **kwargs), dtype=float
        )
        deriv2[k] = (f_plus - 2.0 * f0 + f_minus) / (step**2)

    return jnp.asarray(deriv2.reshape(freq.shape))


def finite_difference_d2f_dtheta(
    function: IndexedDerivativeFn,
    frequency: Array,
    parameters: ArrayLike,
    *args: Any,
    step_f: float = 1e-5,
    step_theta: float = 1e-6,
    **kwargs: Any,
) -> Array:
    """
    Central finite-difference mixed derivative d/df(d/dtheta).
    """

    def _mixed_at_scalar_freq(ff: float) -> np.ndarray:
        g_plus = np.asarray(
            finite_difference_grad_theta(
                function,
                ff + step_f,
                parameters,
                *args,
                step=step_theta,
                **kwargs,
            ),
            dtype=float,
        )
        g_minus = np.asarray(
            finite_difference_grad_theta(
                function,
                ff - step_f,
                parameters,
                *args,
                step=step_theta,
                **kwargs,
            ),
            dtype=float,
        )
        return (g_plus - g_minus) / (2.0 * step_f)

    freq = np.asarray(frequency, dtype=float)

    if freq.ndim == 0:
        return jnp.asarray(_mixed_at_scalar_freq(float(freq)))

    flat = freq.ravel()
    mixed = np.asarray([_mixed_at_scalar_freq(float(ff)) for ff in flat], dtype=float)
    return jnp.asarray(mixed.reshape(freq.shape + (np.asarray(parameters).size,)))


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
