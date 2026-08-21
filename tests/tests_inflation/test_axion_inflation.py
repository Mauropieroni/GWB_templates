"""Tests for the numerical U(1) axion-inflation template."""

from __future__ import annotations

import math
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gwb_templates import get_template_from_registry
from gwb_templates.inflation_templates import AxionInflationU1

ARTIFACT_STEM = "axion_inflation_current_default_exp7_dev_v0"


def _resource(filename: str):
    return resources.files("gwb_templates.inflation_templates").joinpath(
        "data", filename
    )


def _artifact_arrays() -> dict[str, np.ndarray]:
    with np.load(_resource(f"{ARTIFACT_STEM}.npz"), allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _numpy_cell(axis: np.ndarray, value: float) -> tuple[int, float]:
    lower = int(np.searchsorted(axis, value, side="right") - 1)
    lower = int(np.clip(lower, 0, len(axis) - 2))
    fraction = float(
        np.clip((value - axis[lower]) / (axis[lower + 1] - axis[lower]), 0, 1)
    )
    return lower, fraction


def _numpy_parameter_interpolation(
    table: np.ndarray,
    inv_axis: np.ndarray,
    abs_axis: np.ndarray,
    inv_f_tilde: float,
    abs_vprime: float,
) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    inv_index, inv_fraction = _numpy_cell(inv_axis, inv_f_tilde)
    abs_index, abs_fraction = _numpy_cell(abs_axis, abs_vprime)
    return (
        (1 - abs_fraction) * (1 - inv_fraction) * table[abs_index, inv_index]
        + abs_fraction * (1 - inv_fraction) * table[abs_index + 1, inv_index]
        + (1 - abs_fraction) * inv_fraction * table[abs_index, inv_index + 1]
        + abs_fraction * inv_fraction * table[abs_index + 1, inv_index + 1]
    )


@pytest.fixture(scope="module")
def model() -> AxionInflationU1:
    return get_template_from_registry("AxionInflationU1")


def test_registry_identity_parameters_and_u1_name(model):
    assert isinstance(model, AxionInflationU1)
    assert model.parameter_names == ("inv_f_tilde", "abs_vprime")
    assert tuple(model.prior_by_param) == model.parameter_names
    assert tuple(model.reference_prior_bounds) == model.parameter_names
    assert model.model_type == "AxionInflationU1"
    assert model.model_label == "Axion Inflation (U(1))"
    named = get_template_from_registry("AxionInflationU1", model_name="axion-u1")
    assert named.model_id == "AxionInflationU1:axion-u1"


def test_data_filename_selects_same_stem_metadata(model):
    assert model.data_filename == f"{ARTIFACT_STEM}.npz"
    assert model.metadata_filename == f"{ARTIFACT_STEM}.meta.json"
    selected = AxionInflationU1(data_filename=f"{ARTIFACT_STEM}.npz")
    assert selected.model_version == ARTIFACT_STEM
    with pytest.raises(ValueError, match="must end with '.npz'"):
        AxionInflationU1(data_filename=f"{ARTIFACT_STEM}.json")


def test_runtime_validity_contract(model):
    required_attributes = (
        "validity_contract",
        "reference_prior_bounds",
        "valid_prior_mass",
        "valid_prior_mass_error",
        "log_evidence_correction",
        "validity_threshold",
        "frequency_bounds_hz",
        "model_version",
        "validity_contract_version",
    )
    assert all(hasattr(model, name) for name in required_attributes)
    assert model.model_version == ARTIFACT_STEM
    assert model.validity_contract_version == "axion_inflation_current_default_dev_v0"
    assert model.validity_contract["parameter_order"] == model.parameter_names
    assert model.validity_contract["support_rule"] == (
        "four_corner_sampled_node_conjunction"
    )
    assert model.validity_contract["threshold_comparison"] == "strict_less_than"


def test_training_node_recovery(model):
    arrays = _artifact_arrays()
    abs_index, inv_index = 14, 30
    assert arrays["sampled_node"][abs_index, inv_index]
    frequency_indices = np.array([17, 82, 151])
    frequencies = 10.0 ** arrays["log10_frequency_hz"][frequency_indices]
    expected = (
        10.0 ** arrays["log10_omega_gw_h2"][abs_index, inv_index, frequency_indices]
    )
    actual = model.omega_gw_h2(
        frequencies,
        arrays["inv_f_tilde"][inv_index],
        arrays["abs_vprime"][abs_index],
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.0)


def test_off_grid_spectrum_matches_independent_numpy_reference(model):
    arrays = _artifact_arrays()
    inv_f_tilde, abs_vprime = 64.612, 0.1369
    log_frequency = np.array([-2.731, -0.413, 1.627])
    log_spectrum = _numpy_parameter_interpolation(
        arrays["log10_omega_gw_h2"],
        arrays["inv_f_tilde"],
        arrays["abs_vprime"],
        inv_f_tilde,
        abs_vprime,
    )
    expected_log_omega = np.interp(
        log_frequency, arrays["log10_frequency_hz"], log_spectrum
    )
    expected = 10.0**expected_log_omega
    actual = model.omega_gw_h2(10.0**log_frequency, inv_f_tilde, abs_vprime)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_scalar_and_vector_frequency_behavior(model):
    scalar = model.omega_gw_h2(1.0e-3, 64.612, 0.1369)
    vector = model.omega_gw_h2(jnp.array([1.0e-3, 2.0e-2]), 64.612, 0.1369)
    assert scalar.shape == ()
    assert vector.shape == (2,)
    assert isinstance(scalar, jax.Array)
    assert isinstance(vector, jax.Array)
    assert np.all(np.asarray(vector) >= 0.0)


def test_jit_and_vmap_match_eager(model):
    frequencies = jnp.geomspace(1.0e-5, 1.0e2, 21)
    eager = model.omega_gw_h2(frequencies, 64.612, 0.1369)
    compiled = jax.jit(model.omega_gw_h2)(frequencies, 64.612, 0.1369)
    np.testing.assert_allclose(compiled, eager, rtol=2e-13, atol=0.0)

    parameters = jnp.array([[64.612, 0.1369], [70.2, 0.1303]])
    vmapped = jax.vmap(
        lambda theta: model.omega_gw_h2(frequencies, theta[0], theta[1])
    )(parameters)
    expected = jnp.stack(
        [model.omega_gw_h2(frequencies, *theta) for theta in parameters]
    )
    np.testing.assert_allclose(vmapped, expected, rtol=2e-13, atol=0.0)


def test_autodiff_matches_central_finite_difference_inside_cell(model):
    frequencies = jnp.array([1.0e-3, 3.0e-2, 10.0])
    theta = np.array([64.612, 0.1369])
    gradient = np.asarray(model.grad_theta_omega_gw_h2(frequencies, theta))
    steps = (1.0e-4, 1.0e-7)
    for parameter_index, step in enumerate(steps):
        upper = theta.copy()
        lower = theta.copy()
        upper[parameter_index] += step
        lower[parameter_index] -= step
        finite_difference = (
            np.asarray(model.omega_gw_h2(frequencies, *upper))
            - np.asarray(model.omega_gw_h2(frequencies, *lower))
        ) / (2.0 * step)
        np.testing.assert_allclose(
            gradient[:, parameter_index],
            finite_difference,
            rtol=5e-3,
            atol=1e-18,
        )


def test_out_of_band_is_zero_and_frequency_bounds_match_data(model):
    arrays = _artifact_arrays()
    data_bounds = tuple(
        float(value) for value in 10.0 ** arrays["log10_frequency_hz"][[0, -1]]
    )
    np.testing.assert_allclose(model.frequency_bounds_hz, data_bounds, rtol=0)
    outside = jnp.array(
        [
            0.0,
            -1.0,
            data_bounds[0] * (1.0 - 1.0e-6),
            data_bounds[1] * (1.0 + 1.0e-6),
            jnp.inf,
            jnp.nan,
        ]
    )
    np.testing.assert_array_equal(model.omega_gw_h2(outside, 64.612, 0.1369), 0.0)


def test_invalid_parameter_points_still_evaluate_finitely(model):
    frequencies = jnp.geomspace(1.0e-5, 1.0e2, 31)
    for inv_f_tilde, abs_vprime in (
        (35.0, 0.12),
        (105.0, 0.18),
        (97.71428571428572, 0.10714285714285715),
        (jnp.nan, 0.12),
    ):
        assert not bool(model.is_valid(inv_f_tilde, abs_vprime))
        spectrum = model.omega_gw_h2(frequencies, inv_f_tilde, abs_vprime)
        assert np.all(np.isfinite(spectrum))
        assert np.isfinite(model.max_grad_over_kin(inv_f_tilde, abs_vprime))


def test_unsupported_cell_is_invalid_by_four_corner_rule(model):
    abs_index, inv_index = 12, 33
    assert not bool(model.support_cell[abs_index, inv_index])
    inv_f_tilde = float(
        (model.inv_f_tilde_axis[inv_index] + model.inv_f_tilde_axis[inv_index + 1]) / 2
    )
    abs_vprime = float(
        (model.abs_vprime_axis[abs_index] + model.abs_vprime_axis[abs_index + 1]) / 2
    )
    assert not bool(model.is_valid(inv_f_tilde, abs_vprime))


def test_full_history_is_interpolated_before_maximizing(model):
    arrays = _artifact_arrays()
    abs_index, inv_index = 26, 19
    inv_f_tilde = float(np.mean(arrays["inv_f_tilde"][inv_index : inv_index + 2]))
    abs_vprime = float(np.mean(arrays["abs_vprime"][abs_index : abs_index + 2]))
    corners = arrays["ratio_grad_over_kin"][
        abs_index : abs_index + 2, inv_index : inv_index + 2
    ]
    full_history_then_max = float(np.max(np.mean(corners, axis=(0, 1))))
    interpolate_node_maxima = float(np.mean(np.max(corners, axis=-1)))
    actual = float(model.max_grad_over_kin(inv_f_tilde, abs_vprime))
    assert actual == pytest.approx(full_history_then_max, rel=2e-7)
    assert interpolate_node_maxima - actual > 0.05


def test_known_points_bracket_strict_ratio_boundary(model):
    valid_point = (70.28571428571428, 0.13)
    invalid_point = (78.85714285714286, 0.12314285714285714)
    valid_ratio = float(model.max_grad_over_kin(*valid_point))
    invalid_ratio = float(model.max_grad_over_kin(*invalid_point))
    assert valid_ratio == pytest.approx(0.09868825972080231)
    assert invalid_ratio == pytest.approx(0.10009675333276391)
    assert bool(model.is_valid(*valid_point))
    assert not bool(model.is_valid(*invalid_point))


def test_threshold_comparison_is_strict():
    model = AxionInflationU1()
    point = (64.612, 0.1369)
    assert bool(model.is_valid(*point))
    model.validity_threshold = float(model.max_grad_over_kin(*point))
    assert not bool(model.is_valid(*point))


def test_valid_prior_mass_and_correction_sign(model):
    assert model.valid_prior_mass == pytest.approx(0.6280302405357361)
    assert model.valid_prior_mass_error == pytest.approx(1.5020370483398438e-05)
    assert model.log_evidence_correction > 0.0
    assert model.log_evidence_correction == pytest.approx(
        -math.log(model.valid_prior_mass), rel=1e-15
    )
