"""U(1) axion spectra for the two-parameter linear-slope potential family.

The NPZ contains the interpolation data. Spectra are Omega_GW h^2 at the
stored frequencies. Thresholds and priors are configured through the constructor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np

from gwb_templates.template import NumericalTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_DEFAULT_DATA_FILENAME = "axion_u1_linear_slope_lisa_et.npz"
_PARAMETER_NAMES = ("inv_f_tilde", "abs_vprime")
_COMMON_ARRAY_NAMES = (
    *_PARAMETER_NAMES,
    "log10_frequency_hz",
    "ratio_grad_over_kin",
    "sampled_node",
)
_INTERPOLATION_ARRAY_NAMES = {
    "matern52": (
        "centers",
        "parameter_offset",
        "parameter_scale",
        "length_scale",
        "weights",
        "trend_coefficients",
    ),
    "bilinear": ("log10_omega_gw_h2",),
}


def _validate_arrays(
    arrays: Mapping[str, Any], interpolation: str = "matern52"
) -> dict[str, np.ndarray]:
    """Validate the selected interpolator's numeric arrays, values and shapes."""
    if interpolation not in _INTERPOLATION_ARRAY_NAMES:
        raise ValueError("interpolation must be 'matern52' or 'bilinear'.")
    names = _COMMON_ARRAY_NAMES + _INTERPOLATION_ARRAY_NAMES[interpolation]
    missing = set(names) - set(arrays)
    if missing:
        raise ValueError(f"Missing {interpolation} arrays: {sorted(missing)}.")
    result = {}
    for name in names:
        value = np.asarray(arrays[name])
        if name == "sampled_node":
            if value.dtype.kind not in "biuf" or not np.all(
                (value == 0) | (value == 1)
            ):
                raise ValueError("sampled_node must be a boolean or numeric 0/1 mask.")
            result[name] = value.astype(bool)
        else:
            if value.dtype.kind not in "iuf":
                raise ValueError(f"{name} must contain finite real numeric values.")
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype(np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite real numeric values.")
            result[name] = value

    for name in (*_PARAMETER_NAMES, "log10_frequency_hz"):
        axis = result[name]
        minimum = 1 if name == "log10_frequency_hz" else 2
        if axis.ndim != 1 or axis.size < minimum or not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"{name} must be a strictly increasing 1D axis "
                f"with >= {minimum} values."
            )
    n_inv, n_abs = (result[name].size for name in _PARAMETER_NAMES)
    n_frequency = result["log10_frequency_hz"].size
    shapes = {"sampled_node": (n_abs, n_inv)}
    if interpolation == "matern52":
        centers = result["centers"]
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
            raise ValueError(
                "centers must have shape (n_centers, 2), with n_centers > 0."
            )
        shapes.update(
            parameter_offset=(2,),
            parameter_scale=(2,),
            length_scale=(2,),
            weights=(len(centers), n_frequency),
            trend_coefficients=(3, n_frequency),
        )
        for name in ("parameter_scale", "length_scale"):
            if not np.all(result[name] > 0):
                raise ValueError(f"{name} must be positive.")
    else:
        shapes["log10_omega_gw_h2"] = (n_abs, n_inv, n_frequency)
    for name, shape in shapes.items():
        if result[name].shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}; got {result[name].shape}."
            )
    ratio = result["ratio_grad_over_kin"]
    if ratio.ndim != 3 or ratio.shape[:2] != (n_abs, n_inv) or ratio.shape[2] == 0:
        raise ValueError(
            "ratio_grad_over_kin must have shape (n_abs, n_inv, n_history > 0)."
        )
    return result


def _load_arrays(source, interpolation: str) -> dict[str, np.ndarray]:
    with source.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        # Load required fields lazily, leaving auxiliary fields untouched.
        return _validate_arrays(payload, interpolation)


def _cell_coordinate(value: ArrayLike, axis: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Use the right-hand cell at interior nodes, clipping dense evaluations."""
    value_array = jnp.asarray(value)
    safe_value = jnp.nan_to_num(
        value_array, nan=axis[0], neginf=axis[0], posinf=axis[-1]
    )
    lower = jnp.searchsorted(axis, safe_value, side="right") - 1
    lower = jnp.clip(lower, 0, axis.shape[0] - 2).astype(jnp.int32)
    left = axis[lower]
    right = axis[lower + 1]
    fraction = jnp.clip((safe_value - left) / (right - left), 0.0, 1.0)
    return lower, fraction


def _bilinear_parameter_interpolation(
    table: jax.Array,
    abs_index: jax.Array,
    inv_index: jax.Array,
    abs_fraction: jax.Array,
    inv_fraction: jax.Array,
) -> jax.Array:
    """Interpolate a table ordered as [abs_vprime, inv_f_tilde, ...]."""
    trailing_dimensions = table.ndim - 2
    weight_shape = abs_fraction.shape + (1,) * trailing_dimensions
    wa = jnp.reshape(abs_fraction, weight_shape)
    wi = jnp.reshape(inv_fraction, weight_shape)
    g00 = table[abs_index, inv_index]
    g10 = table[abs_index + 1, inv_index]
    g01 = table[abs_index, inv_index + 1]
    g11 = table[abs_index + 1, inv_index + 1]
    return (
        (1.0 - wa) * (1.0 - wi) * g00
        + wa * (1.0 - wi) * g10
        + (1.0 - wa) * wi * g01
        + wa * wi * g11
    )


@jax.custom_jvp
def _matern52(radius_squared: jax.Array) -> jax.Array:
    radius = jnp.sqrt(jnp.maximum(radius_squared, jnp.finfo(radius_squared.dtype).tiny))
    z = jnp.sqrt(5.0) * radius
    return (1.0 + z + (5.0 / 3.0) * radius_squared) * jnp.exp(-z)


@_matern52.defjvp
def _matern52_jvp(primals, tangents):
    """Use the finite analytic slope in r², including at a kernel center."""
    (radius_squared,), (tangent,) = primals, tangents
    radius = jnp.sqrt(jnp.maximum(radius_squared, jnp.finfo(radius_squared.dtype).tiny))
    z = jnp.sqrt(5.0) * radius
    slope = -(5.0 / 6.0) * (1.0 + z) * jnp.exp(-z)
    return _matern52(radius_squared), slope * tangent


def _boundary_crossings(a0, a1, b0, b1, threshold, u0, u1, v0, v1):
    """Find intersections of history boundaries inside a local grid cell."""
    i, j = np.triu_indices(len(a0), k=1)
    q, p = threshold - a0, -a1
    c0 = q[i] * b0[j] - q[j] * b0[i]
    c1 = p[i] * b0[j] + q[i] * b1[j] - p[j] * b0[i] - q[j] * b1[i]
    c2 = p[i] * b1[j] - p[j] * b1[i]
    discriminant = c1 * c1 - 4.0 * c2 * c0
    roots = []
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = -0.5 * (
            c1 + np.copysign(np.sqrt(np.maximum(discriminant, 0.0)), c1)
        )
        candidates = (
            np.where((c2 == 0.0) & (c1 != 0.0), -c0 / c1, np.nan),
            np.where((c2 != 0.0) & (discriminant >= 0.0), numerator / c2, np.nan),
            np.where((c2 != 0.0) & (discriminant >= 0.0), c0 / numerator, np.nan),
        )
        for u in candidates:
            slope = b0[i] + b1[i] * u
            v = (q[i] + p[i] * u) / slope
            inside = (u > u0) & (u < u1) & (v > v0) & (v < v1) & (slope != 0.0)
            roots.extend(u[inside])
    return roots


def _valid_prior_mass(x, y, ratios, support_cell, bounds, threshold, epsabs):
    """Integrate strict ratio validity for an independent uniform rectangle.

    At fixed local x coordinate, each bilinear history value is ``A + B*y``.
    Intersect its strict threshold constraint with those of every other history
    sample, then integrate the remaining y interval. All boundary intersections
    split the integration so narrow interior valid regions cannot be skipped.

    ``ratios`` is ordered as [y, x, history]; ``bounds`` rows are [x, y].
    Unsupported cells and prior area outside the grid contribute zero. The error
    estimates numerical quadrature error, with a floating-point floor.
    """
    from scipy.integrate import quad

    x, y, bounds = (np.asarray(value, dtype=float) for value in (x, y, bounds))
    prior_area = np.prod(bounds[:, 1] - bounds[:, 0])
    mass = error = 0.0
    for j, i in np.argwhere(support_cell):
        dx, dy = x[i + 1] - x[i], y[j + 1] - y[j]
        u0, u1 = np.clip((bounds[0] - x[i]) / dx, 0.0, 1.0)
        v0, v1 = np.clip((bounds[1] - y[j]) / dy, 0.0, 1.0)
        if u0 >= u1 or v0 >= v1:
            continue
        corners = np.asarray(ratios[j : j + 2, i : i + 2], dtype=float)
        area_fraction = dx * dy / prior_area
        if np.all(corners < threshold):
            mass += (u1 - u0) * (v1 - v0) * area_fraction
            continue
        if np.any(np.all(corners >= threshold, axis=(0, 1))):
            continue

        a0 = corners[0, 0]
        a1 = corners[0, 1] - a0
        b0 = corners[1, 0] - a0
        b1 = corners[1, 1] - corners[0, 1] - b0

        def width(u):
            a, b = a0 + a1 * u, b0 + b1 * u
            if np.any((b == 0.0) & (a >= threshold)):
                return 0.0
            positive, negative = b > 0.0, b < 0.0
            upper = np.min((threshold - a[positive]) / b[positive], initial=v1)
            lower = np.max((threshold - a[negative]) / b[negative], initial=v0)
            return max(0.0, upper - lower)

        points = []
        for intercept, slope in (
            (a0 + b0 * v0 - threshold, a1 + b1 * v0),
            (a0 + b0 * v1 - threshold, a1 + b1 * v1),
            (b0, b1),
        ):
            nonzero = slope != 0.0
            roots = -intercept[nonzero] / slope[nonzero]
            points.extend(roots[(roots > u0) & (roots < u1)])
        points.extend(_boundary_crossings(a0, a1, b0, b1, threshold, u0, u1, v0, v1))
        points = np.unique(points)
        integral, estimate = quad(
            width,
            u0,
            u1,
            points=points,
            epsabs=epsabs,
            epsrel=epsabs,
            limit=max(100, 2 * len(points)),
        )
        mass += integral * area_fraction
        error += estimate * area_fraction
    return float(np.clip(mass, 0.0, 1.0)), max(error, 64 * np.finfo(float).eps)


class AxionU1LinearSlope(NumericalTemplate):
    r"""U(1)-sourced SGWB for the piecewise potential of arXiv:2303.13425.

    Free parameters
    ---------------
    inv_f_tilde
        Dimensionless inverse coupling :math:`1/\tilde{f} = M_p/f`.
    abs_vprime
        Intermediate linear-branch slope magnitude
        :math:`|(M_p/V_0)\,dV/d\phi|`. The study uses a negative signed slope.

    Other potential settings follow the study's baseline and dependent rescaling.
    Matérn-5/2 parameter interpolation is smooth inside the grid rectangle;
    bilinear derivatives can jump at grid lines. Both use piecewise-linear
    interpolation in log frequency. Use ``jax.vmap`` for parameter batches.

    Spectrum evaluation clips parameters to the grid rectangle. Apply
    :meth:`is_valid` as the analysis cut for sampled support and energy ratio.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{GarciaBellido:2023AxionBeacon,
    author = "Garcia-Bellido, Juan and Papageorgiou, Alexandros and
        Peloso, Marco and Sorbo, Lorenzo",
    title = "{A flashing beacon in axion inflation: recurring bursts of
        gravitational waves in the strong backreaction regime}",
    eprint = "2303.13425",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    year = "2023"
}
""",
    )

    def __init__(
        self,
        data_file: str | Path | None = None,
        *,
        interpolation: str = "matern52",
        ratio_threshold: float = 0.1,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the spectrum interpolator.

        Args:
            data_file: Path to a prepared NPZ for this model and parameter
                definition. None selects the packaged LISA–ET study data.
            interpolation: 'matern52' uses fitted weights and a linear trend;
                'bilinear' uses a spectrum grid. Each loads its required arrays.
            ratio_threshold: Strict upper bound on the interpolated
                gradient/kinetic energy ratio used by is_valid.
            prior_by_param: Parameter priors. Defaults to independent uniforms
                over the stored parameter axes.
        """
        if interpolation not in _INTERPOLATION_ARRAY_NAMES:
            raise ValueError("interpolation must be 'matern52' or 'bilinear'.")
        self._interpolation = interpolation
        self.data_file = Path(data_file) if data_file is not None else None
        self._validity_threshold = float(ratio_threshold)
        if not math.isfinite(self._validity_threshold):
            raise ValueError("ratio_threshold must be finite.")
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Axion U(1), linear slope"
            ),
            parameter_labels=(
                parameter_labels
                if parameter_labels is not None
                else {"inv_f_tilde": r"$1/\tilde{f}$", "abs_vprime": r"$|v'|$"}
            ),
            prior_by_param=prior_by_param,
        )
        if prior_by_param is None:
            self.prior_by_param = MappingProxyType(
                {
                    name: {"min": float(axis[0]), "max": float(axis[-1])}
                    for name, axis in zip(
                        _PARAMETER_NAMES, (self.inv_f_tilde_axis, self.abs_vprime_axis)
                    )
                }
            )

    @property
    def interpolation(self) -> str:
        """Parameter interpolation selected when constructing this template."""
        return self._interpolation

    @property
    def validity_threshold(self) -> float:
        """Strict upper bound on the interpolated gradient/kinetic ratio."""
        return self._validity_threshold

    def setup(self) -> None:
        """Load the selected spectrum data and aligned ratio histories once."""
        source = (
            self.data_file
            if self.data_file is not None
            else resources.files("gwb_templates").joinpath(
                "inflation_templates", "data", _DEFAULT_DATA_FILENAME
            )
        )
        arrays = _load_arrays(source, self.interpolation)
        self.inv_f_tilde_axis = jnp.asarray(arrays.pop("inv_f_tilde"))
        self.abs_vprime_axis = jnp.asarray(arrays.pop("abs_vprime"))
        self.ratio_grad_over_kin_table = jnp.asarray(arrays.pop("ratio_grad_over_kin"))
        if self.interpolation == "bilinear":
            self.log10_omega_gw_h2_table = jnp.asarray(arrays.pop("log10_omega_gw_h2"))
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
        if self.interpolation == "matern52":
            self.centers_normalized = (
                self.centers - self.parameter_offset
            ) / self.parameter_scale
        self.support_cell = (
            self.sampled_node[:-1, :-1]
            & self.sampled_node[1:, :-1]
            & self.sampled_node[:-1, 1:]
            & self.sampled_node[1:, 1:]
        )
        self.frequency_bounds_hz = tuple(
            float(value) for value in 10.0 ** arrays["log10_frequency_hz"][[0, -1]]
        )
        self.reference_prior_bounds = None
        self.valid_prior_mass = None
        self.valid_prior_mass_error = None
        self.log_evidence_correction = None

    def _log10_spectrum(
        self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike
    ) -> jax.Array:
        if self.interpolation == "bilinear":
            return _bilinear_parameter_interpolation(
                self.log10_omega_gw_h2_table,
                *self._parameter_cell(inv_f_tilde, abs_vprime),
            )
        parameters = jnp.stack(jnp.broadcast_arrays(inv_f_tilde, abs_vprime), axis=-1)
        lower = jnp.array([self.inv_f_tilde_axis[0], self.abs_vprime_axis[0]])
        upper = jnp.array([self.inv_f_tilde_axis[-1], self.abs_vprime_axis[-1]])
        safe = jnp.clip(
            jnp.nan_to_num(parameters, nan=lower, neginf=lower, posinf=upper),
            lower,
            upper,
        )
        normalized = (safe - self.parameter_offset) / self.parameter_scale
        delta = (normalized[..., None, :] - self.centers_normalized) / self.length_scale
        kernel = _matern52(jnp.sum(delta * delta, axis=-1))
        return (
            kernel @ self.weights
            + self.trend_coefficients[0]
            + normalized @ self.trend_coefficients[1:]
        )

    def omega_gw_h2(
        self, frequency: ArrayLike, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike
    ) -> jax.Array:
        r"""Evaluate :math:`\Omega_{\mathrm{GW}}h^2`, zero outside the stored band."""
        log_spectrum = self._log10_spectrum(inv_f_tilde, abs_vprime)
        frequency = jnp.asarray(frequency)
        lower, upper = self.frequency_bounds_hz
        in_band = jnp.isfinite(frequency) & (frequency >= lower) & (frequency <= upper)
        safe_frequency = jnp.where(in_band, frequency, lower)
        log_omega = jnp.interp(
            jnp.log10(safe_frequency), self.log10_frequency_hz, log_spectrum
        )
        return jnp.where(in_band, 10.0**log_omega, 0.0)

    def _parameter_cell(self, inv_f_tilde, abs_vprime):
        inv_value, abs_value = jnp.broadcast_arrays(inv_f_tilde, abs_vprime)
        inv_index, inv_fraction = _cell_coordinate(inv_value, self.inv_f_tilde_axis)
        abs_index, abs_fraction = _cell_coordinate(abs_value, self.abs_vprime_axis)
        return abs_index, inv_index, abs_fraction, inv_fraction

    def max_grad_over_kin(
        self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike
    ) -> jax.Array:
        """Interpolate aligned ratio histories, then maximize over time."""
        history = _bilinear_parameter_interpolation(
            self.ratio_grad_over_kin_table,
            *self._parameter_cell(inv_f_tilde, abs_vprime),
        )
        return jnp.max(history, axis=-1)

    def is_valid(self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike) -> jax.Array:
        """Check original inputs against bounds, four-corner support and ratio."""
        inv_value, abs_value = jnp.broadcast_arrays(inv_f_tilde, abs_vprime)
        abs_index, inv_index, _, _ = self._parameter_cell(inv_value, abs_value)
        return (
            jnp.isfinite(inv_value)
            & jnp.isfinite(abs_value)
            & (inv_value >= self.inv_f_tilde_axis[0])
            & (inv_value <= self.inv_f_tilde_axis[-1])
            & (abs_value >= self.abs_vprime_axis[0])
            & (abs_value <= self.abs_vprime_axis[-1])
            & self.support_cell[abs_index, inv_index]
            & (self.max_grad_over_kin(inv_value, abs_value) < self.validity_threshold)
        )

    def compute_evidence_correction(self, *, epsabs: float = 1.0e-8) -> float:
        """Compute and store alpha=P(valid) and -log(alpha) for uniform priors.

        Evaluate during analysis setup and recompute after changing the priors.
        Each prior uses a finite ``{"min": lo, "max": hi}`` interval. Prior
        area outside the calibration rectangle contributes zero valid mass.
        ``valid_prior_mass_error`` estimates numerical quadrature error.

        Add the returned correction to log evidence computed with the rectangular
        prior and a hard validity cut to obtain log evidence for the prior
        conditioned on validity. Evidence computed directly with that conditional
        prior is already normalized.
        """
        self.reference_prior_bounds = None
        self.valid_prior_mass = None
        self.valid_prior_mass_error = None
        self.log_evidence_correction = None
        if not math.isfinite(epsabs) or epsabs <= 0:
            raise ValueError("epsabs must be finite and positive.")
        bounds = {}
        for name in _PARAMETER_NAMES:
            prior = self.prior_by_param.get(name)
            if not isinstance(prior, Mapping) or set(prior) != {"min", "max"}:
                raise ValueError(
                    "Evidence correction requires independent min/max uniforms."
                )
            lower, upper = float(prior["min"]), float(prior["max"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid prior bounds for {name}.")
            bounds[name] = (lower, upper)
        mass, error = _valid_prior_mass(
            np.asarray(self.inv_f_tilde_axis),
            np.asarray(self.abs_vprime_axis),
            np.asarray(self.ratio_grad_over_kin_table),
            np.asarray(self.support_cell),
            np.asarray(list(bounds.values())),
            self.validity_threshold,
            epsabs,
        )
        if mass <= 0:
            raise ValueError("The configured prior has zero valid mass.")
        self.reference_prior_bounds = MappingProxyType(bounds)
        self.valid_prior_mass = mass
        self.valid_prior_mass_error = error
        self.log_evidence_correction = -math.log(mass)
        return self.log_evidence_correction
