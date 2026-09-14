# Global imports
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

import os
import jax
import jax.numpy as jnp
from interpax import Interpolator1D

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
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    **kwargs: Any,
) -> jax.Array:
    """
    Build a vectorized first-derivative tensor via forward-mode autodiff.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Frequency grid.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with parameter axis in the last position: (..., npars).
    """
    return jax.jacfwd(function, argnums=1)(frequency, parameters, *args, **kwargs)


def hessian_autodiff(
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    **kwargs: Any,
) -> jax.Array:
    """
    Build a vectorized second-derivative tensor via forward-mode autodiff.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
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
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    step: float = 1e-6,
    **kwargs: Any,
) -> jax.Array:
    """
    Central finite-difference gradient with respect to model parameters.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Frequency grid.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        step: Central-difference step size.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with the parameter axis appended last: (..., npars).
    """
    npars = parameters.size

    f0 = function(frequency, parameters, *args, **kwargs)
    grad = jnp.empty(f0.shape + (npars,), dtype=float)

    for i in range(npars):
        p_plus = parameters.at[i].set(parameters[i] + step)
        p_minus = parameters.at[i].set(parameters[i] - step)

        f_plus = function(frequency, p_plus, *args, **kwargs)
        f_minus = function(frequency, p_minus, *args, **kwargs)
        grad = grad.at[..., i].set((f_plus - f_minus) / (2.0 * step))

    return grad


def finite_difference_hess_theta(
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    step: float = 1e-5,
    **kwargs: Any,
) -> jax.Array:
    """
    Central finite-difference Hessian with respect to model parameters.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Frequency grid.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        step: Central-difference step size.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with the two parameter axes appended last: (..., npars, npars).
    """
    npars = parameters.size

    f0 = function(frequency, parameters, *args, **kwargs)
    hess = jnp.empty(f0.shape + (npars, npars), dtype=float)

    for i in range(npars):
        for j in range(i, npars):
            if i == j:
                p_plus = parameters.at[i].set(parameters[i] + step)
                p_minus = parameters.at[i].set(parameters[i] - step)

                f_plus = function(frequency, p_plus, *args, **kwargs)
                f_minus = function(frequency, p_minus, *args, **kwargs)
                value = (f_plus - 2.0 * f0 + f_minus) / (step**2)
            else:
                p_pp = parameters.at[i].set(parameters[i] + step)
                p_pp = p_pp.at[j].set(p_pp[j] + step)
                p_pm = parameters.at[i].set(parameters[i] + step)
                p_pm = p_pm.at[j].set(p_pm[j] - step)
                p_mp = parameters.at[i].set(parameters[i] - step)
                p_mp = p_mp.at[j].set(p_mp[j] + step)
                p_mm = parameters.at[i].set(parameters[i] - step)
                p_mm = p_mm.at[j].set(p_mm[j] - step)

                f_pp = function(frequency, p_pp, *args, **kwargs)
                f_pm = function(frequency, p_pm, *args, **kwargs)
                f_mp = function(frequency, p_mp, *args, **kwargs)
                f_mm = function(frequency, p_mm, *args, **kwargs)
                value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step**2)

            hess = hess.at[..., i, j].set(value)
            hess = hess.at[..., j, i].set(value)

    return hess


def finite_difference_df(
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    step: float = 1e-6,
    **kwargs: Any,
) -> jax.Array:
    """
    Central finite-difference first derivative with respect to frequency.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Scalar or array of frequencies.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        step: Central-difference step size.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array of the same shape as frequency.
    """
    if frequency.ndim == 0:
        f_plus = function(frequency + step, parameters, *args, **kwargs)
        f_minus = function(frequency - step, parameters, *args, **kwargs)
        return (f_plus - f_minus) / (2.0 * step)

    flat = frequency.ravel()
    deriv = jnp.empty(flat.shape, dtype=float)
    for k in range(flat.size):
        ff = flat[k]
        f_plus = function(ff + step, parameters, *args, **kwargs)
        f_minus = function(ff - step, parameters, *args, **kwargs)
        deriv = deriv.at[k].set((f_plus - f_minus) / (2.0 * step))

    return deriv.reshape(frequency.shape)


def finite_difference_d2f2(
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    step: float = 1e-5,
    **kwargs: Any,
) -> jax.Array:
    """
    Central finite-difference second derivative with respect to frequency.

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Scalar or array of frequencies.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        step: Central-difference step size.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array of the same shape as frequency.
    """
    if frequency.ndim == 0:
        f0 = function(frequency, parameters, *args, **kwargs)
        f_plus = function(frequency + step, parameters, *args, **kwargs)
        f_minus = function(frequency - step, parameters, *args, **kwargs)
        return (f_plus - 2.0 * f0 + f_minus) / (step**2)

    flat = frequency.ravel()
    deriv2 = jnp.empty(flat.shape, dtype=float)
    for k in range(flat.size):
        ff = flat[k]
        f0 = function(ff, parameters, *args, **kwargs)
        f_plus = function(ff + step, parameters, *args, **kwargs)
        f_minus = function(ff - step, parameters, *args, **kwargs)
        deriv2 = deriv2.at[k].set((f_plus - 2.0 * f0 + f_minus) / (step**2))

    return deriv2.reshape(frequency.shape)


def finite_difference_d2f_dtheta(
    function: Callable[..., jax.Array],
    frequency: jax.Array,
    parameters: jax.Array,
    *args: Any,
    step_f: float = 1e-5,
    step_theta: float = 1e-6,
    **kwargs: Any,
) -> jax.Array:
    """
    Central finite-difference mixed derivative d/df(d/dtheta).

    Args:
        function: Callable ``(frequency, parameters, *args, **kwargs) -> jax.Array``.
        frequency: Scalar or array of frequencies.
        parameters: Parameter vector.
        *args: Additional positional arguments forwarded to function.
        step_f: Central-difference step size in frequency.
        step_theta: Central-difference step size in parameters, forwarded to
            :func:`finite_difference_grad_theta`.
        **kwargs: Additional keyword arguments forwarded to function.

    Returns:
        Array with the parameter axis appended last: frequency.shape + (npars,).
    """

    def _mixed_at_scalar_freq(ff: jax.Array) -> jax.Array:
        g_plus = finite_difference_grad_theta(
            function, ff + step_f, parameters, *args, step=step_theta, **kwargs
        )
        g_minus = finite_difference_grad_theta(
            function, ff - step_f, parameters, *args, step=step_theta, **kwargs
        )
        return (g_plus - g_minus) / (2.0 * step_f)

    npars = parameters.size

    if frequency.ndim == 0:
        return _mixed_at_scalar_freq(frequency)

    flat = frequency.ravel()
    mixed = jnp.empty(flat.shape + (npars,), dtype=float)
    for k in range(flat.size):
        mixed = mixed.at[k].set(_mixed_at_scalar_freq(flat[k]))

    return mixed.reshape(frequency.shape + (npars,))


def make_log_log_interpolator(
    freq: jax.Array,
    compute_fn: Callable[..., jax.Array],
    *args: Any,
    n_points: int = 100,
    method: str = "linear",
) -> Callable[[jax.Array], jax.Array]:
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

    def _interpolator(freq: jax.Array) -> jax.Array:
        out_zero = jnp.interp(jnp.log(freq), log_xx, _zero_float) > 0.5
        return jnp.where(out_zero, 0.0, jnp.exp(_log_interp(jnp.log(freq))))

    return _interpolator


def log_log_interpolate(
    freq: jax.Array,
    compute_fn: Callable[..., jax.Array],
    *args: Any,
    n_points: int = 100,
) -> jax.Array:
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
