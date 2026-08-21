"""Numerical U(1) axion-inflation stochastic-background template.

The frequency axis is consumed as stored in the selected interpolation data;
this module does not apply a separate frequency conversion.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib import resources
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np

from gwb_templates.template import NumericalTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_DEFAULT_DATA_FILENAME = "axion_inflation_current_default_exp7_dev_v0.npz"
_PARAMETER_NAMES = ("inv_f_tilde", "abs_vprime")
_EXPECTED_DTYPES = {
    "inv_f_tilde": "<f8",
    "abs_vprime": "<f8",
    "log10_frequency_hz": "<f8",
    "log10_omega_gw_h2": "<f4",
    "ratio_n": "<f4",
    "ratio_grad_over_kin": "<f4",
    "sampled_node": "|b1",
}


def _data_resource(filename: str) -> resources.abc.Traversable:
    return resources.files("gwb_templates.inflation_templates").joinpath(
        "data", filename
    )


def _metadata_filename(data_filename: str) -> str:
    if not data_filename.endswith(".npz"):
        raise ValueError("Axion data_filename must end with '.npz'.")
    return f"{data_filename[:-4]}.meta.json"


def _load_metadata(filename: str) -> dict[str, Any]:
    resource = _data_resource(filename)
    return json.loads(resource.read_text(encoding="utf-8"))


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = set(_EXPECTED_DTYPES) - set(arrays)
    if missing:
        raise ValueError(
            f"Axion interpolation data is missing arrays: {sorted(missing)}."
        )
    for name, dtype in _EXPECTED_DTYPES.items():
        array = arrays[name]
        if array.dtype.str != dtype:
            raise ValueError(
                f"Axion array {name!r} has dtype {array.dtype.str}; "
                f"expected {dtype}."
            )
        if array.dtype.kind != "b" and not np.all(np.isfinite(array)):
            raise ValueError(f"Axion array {name!r} is not finite.")

    axis_names = (
        "inv_f_tilde",
        "abs_vprime",
        "log10_frequency_hz",
        "ratio_n",
    )
    for name in axis_names:
        axis = arrays[name]
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"Axion axis {name!r} must be one-dimensional.")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"Axion axis {name!r} is not strictly increasing.")

    n_abs = len(arrays["abs_vprime"])
    n_inv = len(arrays["inv_f_tilde"])
    n_frequency = len(arrays["log10_frequency_hz"])
    n_ratio = len(arrays["ratio_n"])
    expected_shapes = {
        "inv_f_tilde": (n_inv,),
        "abs_vprime": (n_abs,),
        "log10_frequency_hz": (n_frequency,),
        "log10_omega_gw_h2": (n_abs, n_inv, n_frequency),
        "ratio_n": (n_ratio,),
        "ratio_grad_over_kin": (n_abs, n_inv, n_ratio),
        "sampled_node": (n_abs, n_inv),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Axion array {name!r} has shape {arrays[name].shape}; "
                f"expected {shape}."
            )


def _load_arrays(filename: str) -> dict[str, np.ndarray]:
    resource = _data_resource(filename)
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    _validate_arrays(arrays)
    return arrays


def _cell_coordinate(value: ArrayLike, axis: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the plan-defined lower cell index and a safe local coordinate."""
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


class AxionInflationU1(NumericalTemplate):
    r"""Numerical axion-inflation SGWB sourced by a U(1) gauge field.

    Parameter and log-frequency interpolation are piecewise multilinear, so
    derivatives are piecewise defined and not differentiable on cell borders.
    The default data file is a development-only nearest-filled table. A
    different packaged table can be selected with ``data_filename``; its
    metadata filename is derived from the same stem. Filled values keep dense
    evaluation finite, while :meth:`is_valid` enforces four-corner sampled
    support and the strict ratio threshold.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    def __init__(
        self,
        data_filename: str = _DEFAULT_DATA_FILENAME,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.data_filename = str(data_filename)
        self.metadata_filename = _metadata_filename(self.data_filename)
        self._metadata = _load_metadata(self.metadata_filename)
        bounds = self._metadata["reference_prior_bounds"]
        default_labels = {
            "inv_f_tilde": r"$1/\tilde{f}$",
            "abs_vprime": r"$|v'|$",
        }
        default_priors = {
            name: {"min": bounds[name][0], "max": bounds[name][1]}
            for name in _PARAMETER_NAMES
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Axion Inflation (U(1))"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        """Load the selected packaged interpolation table."""
        arrays = _load_arrays(self.data_filename)
        self.inv_f_tilde_axis = jnp.asarray(arrays["inv_f_tilde"])
        self.abs_vprime_axis = jnp.asarray(arrays["abs_vprime"])
        self.log10_frequency_hz = jnp.asarray(arrays["log10_frequency_hz"])
        self.log10_omega_gw_h2_table = jnp.asarray(arrays["log10_omega_gw_h2"])
        self.ratio_n = jnp.asarray(arrays["ratio_n"])
        self.ratio_grad_over_kin_table = jnp.asarray(arrays["ratio_grad_over_kin"])
        self.sampled_node = jnp.asarray(arrays["sampled_node"])
        self.support_cell = (
            self.sampled_node[:-1, :-1]
            & self.sampled_node[1:, :-1]
            & self.sampled_node[:-1, 1:]
            & self.sampled_node[1:, 1:]
        )

        self.model_version = str(self._metadata["model_version"])
        self.validity_contract_version = str(
            self._metadata["validity_contract_version"]
        )
        self.validity_threshold = float(self._metadata["ratio_threshold"])
        self.valid_prior_mass = float(self._metadata["valid_prior_mass"])
        self.valid_prior_mass_error = float(self._metadata["valid_prior_mass_error"])
        self.log_evidence_correction = -math.log(self.valid_prior_mass)
        self.reference_prior_bounds = MappingProxyType(
            {
                name: tuple(
                    float(value)
                    for value in self._metadata["reference_prior_bounds"][name]
                )
                for name in _PARAMETER_NAMES
            }
        )
        self.frequency_bounds_hz = tuple(
            float(value) for value in 10.0 ** arrays["log10_frequency_hz"][[0, -1]]
        )
        self.validity_contract = MappingProxyType(
            {
                "parameter_order": _PARAMETER_NAMES,
                "reference_prior_family": self._metadata["reference_prior_family"],
                "reference_prior_bounds": self.reference_prior_bounds,
                "support_rule": "four_corner_sampled_node_conjunction",
                "ratio_interpolation": "bilinear_full_history_then_maximum",
                "threshold_comparison": "strict_less_than",
                "validity_threshold": self.validity_threshold,
                "valid_prior_mass": self.valid_prior_mass,
                "valid_prior_mass_error": self.valid_prior_mass_error,
                "log_evidence_correction": self.log_evidence_correction,
                "validity_contract_version": self.validity_contract_version,
                "model_version": self.model_version,
            }
        )

    def _parameter_cell(
        self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        inv_index, inv_fraction = _cell_coordinate(inv_f_tilde, self.inv_f_tilde_axis)
        abs_index, abs_fraction = _cell_coordinate(abs_vprime, self.abs_vprime_axis)
        return abs_index, inv_index, abs_fraction, inv_fraction

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        inv_f_tilde: ArrayLike,
        abs_vprime: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate the packaged :math:`\Omega_{\mathrm{GW}} h^2` spectrum."""
        abs_index, inv_index, abs_fraction, inv_fraction = self._parameter_cell(
            inv_f_tilde, abs_vprime
        )
        log_spectrum = _bilinear_parameter_interpolation(
            self.log10_omega_gw_h2_table,
            abs_index,
            inv_index,
            abs_fraction,
            inv_fraction,
        )
        frequency_array = jnp.asarray(frequency)
        safe_frequency = jnp.where(
            jnp.isfinite(frequency_array) & (frequency_array > 0.0),
            frequency_array,
            10.0 ** self.log10_frequency_hz[0],
        )
        log_frequency = jnp.log10(safe_frequency)
        frequency_index, frequency_fraction = _cell_coordinate(
            log_frequency, self.log10_frequency_hz
        )
        log_omega = (1.0 - frequency_fraction) * log_spectrum[
            frequency_index
        ] + frequency_fraction * log_spectrum[frequency_index + 1]
        in_band = (
            jnp.isfinite(frequency_array)
            & (frequency_array > 0.0)
            & (log_frequency >= self.log10_frequency_hz[0])
            & (log_frequency <= self.log10_frequency_hz[-1])
        )
        return jnp.where(in_band, 10.0**log_omega, 0.0)

    def max_grad_over_kin(
        self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike
    ) -> jax.Array:
        """Interpolate the full ratio history, then maximize over time."""
        abs_index, inv_index, abs_fraction, inv_fraction = self._parameter_cell(
            inv_f_tilde, abs_vprime
        )
        ratio_history = _bilinear_parameter_interpolation(
            self.ratio_grad_over_kin_table,
            abs_index,
            inv_index,
            abs_fraction,
            inv_fraction,
        )
        return jnp.max(ratio_history, axis=-1)

    def is_valid(self, inv_f_tilde: ArrayLike, abs_vprime: ArrayLike) -> jax.Array:
        """Evaluate the joint bounds, support, and strict ratio predicate."""
        inv_value = jnp.asarray(inv_f_tilde)
        abs_value = jnp.asarray(abs_vprime)
        abs_index, inv_index, _, _ = self._parameter_cell(inv_value, abs_value)
        finite = jnp.isfinite(inv_value) & jnp.isfinite(abs_value)
        in_bounds = (
            (inv_value >= self.inv_f_tilde_axis[0])
            & (inv_value <= self.inv_f_tilde_axis[-1])
            & (abs_value >= self.abs_vprime_axis[0])
            & (abs_value <= self.abs_vprime_axis[-1])
        )
        supported = self.support_cell[abs_index, inv_index]
        below_threshold = (
            self.max_grad_over_kin(inv_value, abs_value) < self.validity_threshold
        )
        return finite & in_bounds & supported & below_threshold
