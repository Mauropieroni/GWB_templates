"""U(1) axion-inflation spectra from a fitted Matérn-5/2 model in one NPZ.

The file supplies numerical data, not inference settings. Frequencies are used
as stored and spectra are Omega_GW h^2. The threshold and priors belong to the
template instance; evidence normalization is an explicit, optional setup step.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from gwb_templates.inflation_templates.axion_inflation import (
    ArrayLike,
    _bilinear_parameter_interpolation,
    _cell_coordinate,
    _data_resource,
)
from gwb_templates.template import NumericalTemplate

_DEFAULT_DATA_FILENAME = "axion_inflation_lisa_et_matern52.npz"
_PARAMETER_NAMES = ("inv_f_tilde", "abs_vprime")
_ARRAY_NAMES = (
    *_PARAMETER_NAMES,
    "log10_frequency_hz",
    "centers",
    "parameter_offset",
    "parameter_scale",
    "length_scale",
    "weights",
    "trend_coefficients",
    "ratio_grad_over_kin",
    "sampled_node",
)


def _validate_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Select usable real arrays and check dimensions, not dataset identity."""
    missing = set(_ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError(f"Missing Matérn arrays: {sorted(missing)}.")
    result = {}
    for name in _ARRAY_NAMES:
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
    centers = result["centers"]
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("centers must have shape (n_centers, 2), with n_centers > 0.")
    n_inv, n_abs = (result[name].size for name in _PARAMETER_NAMES)
    n_frequency = result["log10_frequency_hz"].size
    shapes = {
        "parameter_offset": (2,),
        "parameter_scale": (2,),
        "length_scale": (2,),
        "weights": (len(centers), n_frequency),
        "trend_coefficients": (3, n_frequency),
        "sampled_node": (n_abs, n_inv),
    }
    for name, shape in shapes.items():
        if result[name].shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}; got {result[name].shape}."
            )
    for name in ("parameter_scale", "length_scale"):
        if not np.all(result[name] > 0):
            raise ValueError(f"{name} must be positive.")
    ratio = result["ratio_grad_over_kin"]
    if ratio.ndim != 3 or ratio.shape[:2] != (n_abs, n_inv) or ratio.shape[2] == 0:
        raise ValueError(
            "ratio_grad_over_kin must have shape (n_abs, n_inv, n_history > 0)."
        )
    return result


def _load_arrays(source) -> dict[str, np.ndarray]:
    with source.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        # Do not access unused fields, which may include object-valued metadata.
        return _validate_arrays(payload)


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


class AxionInflationU1Matern52(NumericalTemplate):
    r"""Anisotropic Matérn-5/2 plus linear trend in two physical parameters.

    ``data_file=None`` selects the packaged LISA–ET study data. Otherwise pass
    a filesystem path to a fitted model NPZ; no companion JSON is needed.
    The parameter meanings remain ``inv_f_tilde`` and ``abs_vprime``, but grid
    sizes, ranges, centers and frequency bins may differ.

    Parameter priors default to independent uniforms over the stored axes.
    ``ratio_threshold`` controls :meth:`is_valid`, not the spectrum fit.
    Invalid parameters are clipped only for safe spectrum evaluation; callers
    must apply the validity predicate themselves. Use ``jax.vmap`` to batch
    spectra over parameter points. Parameter derivatives are smooth in the
    rectangle's interior; log-frequency interpolation is piecewise linear.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    def __init__(
        self,
        data_file: str | Path | None = None,
        *,
        ratio_threshold: float = 0.1,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.data_file = Path(data_file) if data_file is not None else None
        self._validity_threshold = float(ratio_threshold)
        if not math.isfinite(self._validity_threshold):
            raise ValueError("ratio_threshold must be finite.")
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Axion Inflation (U(1), Matérn-5/2)"
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
    def validity_threshold(self) -> float:
        """Strict upper bound on the interpolated gradient/kinetic ratio."""
        return self._validity_threshold

    def setup(self) -> None:
        """Load the fitted model and aligned ratio histories once."""
        source = (
            self.data_file
            if self.data_file is not None
            else _data_resource(_DEFAULT_DATA_FILENAME)
        )
        arrays = _load_arrays(source)
        self.inv_f_tilde_axis = jnp.asarray(arrays.pop("inv_f_tilde"))
        self.abs_vprime_axis = jnp.asarray(arrays.pop("abs_vprime"))
        self.ratio_grad_over_kin_table = jnp.asarray(arrays.pop("ratio_grad_over_kin"))
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
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

        Call during analysis setup, not in a JIT/inference loop. Each prior
        must be a finite ``{"min": lo, "max": hi}`` rectangle interval. Its
        area outside the calibration rectangle is invalid. The error is a
        numerical integration estimate, not uncertainty in the physical model.

        Add the returned correction only when converting evidence obtained
        with a rectangular prior and a hard validity cut to evidence under
        the prior conditioned on validity. Do not apply it again if the
        inference already uses that normalized conditional prior. Recompute
        if you change the priors; no automatic inference-side action is taken.
        """
        from ._axion_validity import _valid_prior_mass

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
