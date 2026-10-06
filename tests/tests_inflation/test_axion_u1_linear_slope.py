"""Tests for both interpolators of the U(1) linear-slope axion template."""

from __future__ import annotations

from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gwb_templates import get_template_from_registry
from gwb_templates.inflation_templates.axion_u1_linear_slope import (
    AxionU1LinearSlope,
    _DEFAULT_DATA_FILENAME,
    _validate_arrays,
)


@pytest.fixture(scope="module")
def arrays():
    resource = resources.files("gwb_templates").joinpath(
        "inflation_templates", "data", _DEFAULT_DATA_FILENAME
    )
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


@pytest.fixture(scope="module")
def model():
    return get_template_from_registry("AxionU1LinearSlope")


@pytest.fixture(scope="module")
def bilinear_model():
    return get_template_from_registry("AxionU1LinearSlope", interpolation="bilinear")


@pytest.fixture(params=["matern52", "bilinear"])
def interpolation(request):
    return request.param


@pytest.fixture(params=["model", "bilinear_model"])
def template(request):
    return request.getfixturevalue(request.param)


def _small_arrays(n_inv=2, n_abs=2, n_frequency=3):
    return {
        "inv_f_tilde": np.linspace(0.0, 2.0, n_inv),
        "abs_vprime": np.linspace(0.0, 1.0, n_abs),
        "log10_frequency_hz": np.linspace(-3.0, 1.0, n_frequency),
        "centers": np.array([[0.7, 0.4], [1.3, 0.6]]),
        "parameter_offset": np.array([0.0, 0.0]),
        "parameter_scale": np.array([2.0, 1.0]),
        "length_scale": np.array([0.8, 0.9]),
        "weights": np.linspace(-0.2, 0.3, 2 * n_frequency).reshape(2, n_frequency),
        "trend_coefficients": np.vstack(
            [np.full(n_frequency, -10.0), np.ones((2, n_frequency))]
        ),
        "ratio_grad_over_kin": np.zeros((n_abs, n_inv, 2)),
        "sampled_node": np.ones((n_abs, n_inv), dtype=bool),
        "log10_omega_gw_h2": np.linspace(
            -12.0, -8.0, n_abs * n_inv * n_frequency
        ).reshape(n_abs, n_inv, n_frequency),
    }


def _external_model(tmp_path, arrays, **kwargs):
    filename = tmp_path / "scan.npz"
    np.savez(filename, **arrays)
    return AxionU1LinearSlope(data_file=filename, **kwargs)


def _numpy_bilinear(arrays, parameters, field="log10_omega_gw_h2"):
    x, y = arrays["inv_f_tilde"], arrays["abs_vprime"]
    inv, slope = parameters
    i = np.clip(np.searchsorted(x, inv, side="right") - 1, 0, len(x) - 2)
    j = np.clip(np.searchsorted(y, slope, side="right") - 1, 0, len(y) - 2)
    u = np.clip((inv - x[i]) / (x[i + 1] - x[i]), 0, 1)
    v = np.clip((slope - y[j]) / (y[j + 1] - y[j]), 0, 1)
    table = np.asarray(arrays[field], dtype=float)
    return (
        (1 - u) * (1 - v) * table[j, i]
        + (1 - u) * v * table[j + 1, i]
        + u * (1 - v) * table[j, i + 1]
        + u * v * table[j + 1, i + 1]
    )


def _numpy_prediction(arrays, parameters):
    """Independent formula, including analytic parameter derivatives at r=0."""
    normalized = (parameters - arrays["parameter_offset"]) / arrays["parameter_scale"]
    metric = arrays["parameter_scale"] * arrays["length_scale"]
    delta = (parameters - arrays["centers"]) / metric
    radius = np.linalg.norm(delta, axis=1)
    exponential = np.exp(-np.sqrt(5.0) * radius)
    kernel = (1 + np.sqrt(5.0) * radius + 5.0 / 3.0 * radius**2) * exponential
    radial_gradient = -5.0 / 3.0 * (1 + np.sqrt(5.0) * radius) * exponential
    gradient = radial_gradient[:, None] * delta / metric
    hessian = (
        radial_gradient[:, None, None] * np.eye(2)
        + 25.0
        / 3.0
        * exponential[:, None, None]
        * np.einsum("ni,nj->nij", delta, delta)
    ) / (metric[:, None] * metric[None, :])
    return (
        kernel @ arrays["weights"]
        + np.r_[1.0, normalized] @ arrays["trend_coefficients"],
        np.einsum("ni,nf->fi", gradient, arrays["weights"])
        + arrays["trend_coefficients"][1:].T / arrays["parameter_scale"],
        np.einsum("nij,nf->fij", hessian, arrays["weights"]),
    )


def test_registry_identity_and_default_interpolation(model, bilinear_model):
    assert type(model) is type(bilinear_model) is AxionU1LinearSlope
    assert model.model_type == bilinear_model.model_type == "AxionU1LinearSlope"
    assert model.interpolation == "matern52"
    assert bilinear_model.interpolation == "bilinear"
    assert model.parameter_names == ("inv_f_tilde", "abs_vprime")
    assert model.parameter_names == bilinear_model.parameter_names
    named = get_template_from_registry("AxionU1LinearSlope", model_name="axion")
    assert named.model_id == "AxionU1LinearSlope:axion"


def test_bibliography(model):
    assert "2303.13425" in model.get_bibtex()
    assert model.get_bibtex() == "\n\n".join(model.get_bibtex(joined=False))


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        AxionU1LinearSlope(interpolation="unknown")


@pytest.mark.parametrize("shape", [(2, 2, 1), (3, 4, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_external_single_npz_accepts_different_grids_and_real_dtypes(
    tmp_path, shape, dtype, interpolation
):
    data = {name: value.astype(dtype) for name, value in _small_arrays(*shape).items()}
    # Auxiliary fields remain untouched by the loader.
    data["notes"] = np.array([{"unused": True}], dtype=object)
    custom = _external_model(tmp_path, data, interpolation=interpolation)
    as_string = AxionU1LinearSlope(
        data_file=str(tmp_path / "scan.npz"), interpolation=interpolation
    )
    numeric = {
        name: value.astype(float) for name, value in data.items() if name != "notes"
    }
    point = np.array([1.0, 0.5])
    expected = (
        _numpy_prediction(numeric, point)[0]
        if interpolation == "matern52"
        else _numpy_bilinear(numeric, point)
    )
    frequency = 10.0 ** data["log10_frequency_hz"].astype(float)
    actual = custom.omega_gw_h2(frequency, 1.0, 0.5)
    np.testing.assert_allclose(np.log10(actual), expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(actual, as_string.omega_gw_h2(frequency, 1.0, 0.5))
    assert custom.prior_by_param == {
        "inv_f_tilde": {"min": 0.0, "max": 2.0},
        "abs_vprime": {"min": 0.0, "max": 1.0},
    }


def test_integer_data_and_non_grid_centers_are_accepted(tmp_path, interpolation):
    arrays = {name: value.astype(int) for name, value in _small_arrays().items()}
    arrays["length_scale"][:] = 1
    custom = _external_model(tmp_path, arrays, interpolation=interpolation)
    assert np.all(np.isfinite(custom.omega_gw_h2(np.array([0.001, 1.0]), 1.0, 0.5)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("inv_f_tilde", np.array([0.0, 0.0])),
        ("abs_vprime", np.array([0.0])),
        ("log10_frequency_hz", np.array([])),
        ("log10_frequency_hz", np.array([1.0, -1.0, -3.0])),
        ("centers", np.ones((2, 3))),
        ("parameter_offset", np.ones((1, 2))),
        ("parameter_scale", np.array([1.0, 0.0])),
        ("length_scale", np.array([-1.0, 1.0])),
        ("weights", np.ones((3, 3))),
        ("weights", np.full((2, 3), np.nan)),
        ("weights", np.ones((2, 3), dtype=complex)),
        ("weights", np.full((2, 3), "1")),
        ("trend_coefficients", np.ones((2, 3))),
        ("ratio_grad_over_kin", np.empty((2, 2, 0))),
        ("ratio_grad_over_kin", np.ones((3, 2, 2))),
        ("sampled_node", np.ones((2, 3), dtype=bool)),
        ("sampled_node", np.full((2, 2), 2)),
    ],
)
def test_invalid_array_structure_is_rejected(field, value):
    arrays = _small_arrays()
    arrays[field] = value
    with pytest.raises(ValueError):
        _validate_arrays(arrays)


def test_missing_arrays_and_pickle_payload_are_rejected(tmp_path, interpolation):
    arrays = _validate_arrays(_small_arrays(), interpolation=interpolation)
    for name in arrays:
        with pytest.raises(ValueError):
            _validate_arrays(
                {key: value for key, value in arrays.items() if key != name},
                interpolation=interpolation,
            )
    field = "weights" if interpolation == "matern52" else "log10_omega_gw_h2"
    arrays[field] = arrays[field].astype(object)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        _external_model(tmp_path, arrays, interpolation=interpolation)


def test_only_selected_interpolator_fields_are_required_and_read(
    tmp_path, interpolation
):
    arrays = _small_arrays()
    required = _validate_arrays(arrays, interpolation=interpolation)
    minimal = _external_model(tmp_path, required, interpolation=interpolation)
    for name in arrays.keys() - required.keys():
        required[name] = np.array([{"not_read": True}], dtype=object)
    extra = _external_model(tmp_path, required, interpolation=interpolation)
    frequency = 10.0 ** arrays["log10_frequency_hz"]
    np.testing.assert_array_equal(
        minimal.omega_gw_h2(frequency, 1.0, 0.5),
        extra.omega_gw_h2(frequency, 1.0, 0.5),
    )


@pytest.mark.parametrize(
    "value",
    [np.zeros((2, 3, 3)), np.full((2, 2, 3), np.inf), np.zeros((2, 2, 3), complex)],
)
def test_invalid_bilinear_grid_is_rejected(value):
    arrays = _small_arrays()
    arrays["log10_omega_gw_h2"] = value
    with pytest.raises(ValueError):
        _validate_arrays(arrays, interpolation="bilinear")


def test_matern_predictions_match_numpy_within_1e_10_dex(model, arrays):
    low = np.array([arrays["inv_f_tilde"][0], arrays["abs_vprime"][0]])
    high = np.array([arrays["inv_f_tilde"][-1], arrays["abs_vprime"][-1]])
    points = np.vstack(
        [
            arrays["centers"][:: max(1, len(arrays["centers"]) // 5)],
            low,
            high,
            low + (high - low) * [0.405, 0.637],
        ]
    )
    for point in points:
        expected, _, _ = _numpy_prediction(arrays, point)
        np.testing.assert_allclose(
            model._log10_spectrum(*point), expected, atol=1e-10, rtol=0
        )


def test_bilinear_predictions_match_numpy_and_recover_grid_nodes(
    bilinear_model, arrays
):
    x, y = arrays["inv_f_tilde"], arrays["abs_vprime"]
    points = [
        [x[0], y[0]],
        [x[-1], y[-1]],
        [x[len(x) // 2], y[len(y) // 2]],
        [x[0] + 0.413 * np.ptp(x), y[0] + 0.637 * np.ptp(y)],
    ]
    for point in points:
        expected = _numpy_bilinear(arrays, point)
        np.testing.assert_allclose(
            bilinear_model._log10_spectrum(*point), expected, atol=1e-12, rtol=0
        )
    i, j = len(x) // 2, len(y) // 2
    np.testing.assert_array_equal(
        bilinear_model._log10_spectrum(x[i], y[j]), arrays["log10_omega_gw_h2"][j, i]
    )


def test_bilinear_parameter_derivatives_inside_cell(tmp_path):
    arrays = _small_arrays()
    x, y = np.meshgrid(arrays["inv_f_tilde"], arrays["abs_vprime"])
    arrays["log10_omega_gw_h2"][:] = (-10 + 0.3 * x + 0.2 * y + x * y)[:, :, None]
    custom = _external_model(tmp_path, arrays, interpolation="bilinear")
    theta = np.array([0.7, 0.4])

    def evaluate(point):
        return custom._log10_spectrum(*point)

    expected_gradient = np.broadcast_to([0.3 + theta[1], 0.2 + theta[0]], (3, 2))
    expected_hessian = np.broadcast_to([[0.0, 1.0], [1.0, 0.0]], (3, 2, 2))
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(evaluate))(theta), expected_gradient, atol=1e-14
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(evaluate))(theta), expected_hessian, atol=1e-14
    )


@pytest.mark.parametrize("at_center", [False, True])
def test_analytic_derivatives_including_exact_kernel_center(tmp_path, at_center):
    arrays = _small_arrays()
    custom = _external_model(tmp_path, arrays)
    theta = arrays["centers"][0] if at_center else np.array([0.91, 0.47])
    expected, gradient, hessian = _numpy_prediction(arrays, theta)

    def evaluate(point):
        return custom._log10_spectrum(point[0], point[1])

    np.testing.assert_allclose(evaluate(theta), expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(evaluate))(theta), gradient, rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(evaluate))(theta), hessian, rtol=1e-12
    )

    frequencies = 10.0 ** arrays["log10_frequency_hz"]
    spectrum = 10.0**expected
    expected_gradient = np.log(10.0) * spectrum[:, None] * gradient
    expected_hessian = spectrum[:, None, None] * (
        np.log(10.0) * hessian
        + np.log(10.0) ** 2 * np.einsum("fi,fj->fij", gradient, gradient)
    )
    np.testing.assert_allclose(
        custom.grad_theta_omega_gw_h2(frequencies, theta), expected_gradient, rtol=1e-12
    )
    np.testing.assert_allclose(
        custom.hess_theta_omega_gw_h2(frequencies, theta), expected_hessian, rtol=1e-12
    )


def test_packaged_center_hessian_matches_analytic_formula(model, arrays):
    interior = np.all(
        (arrays["centers"] > [arrays["inv_f_tilde"][0], arrays["abs_vprime"][0]])
        & (arrays["centers"] < [arrays["inv_f_tilde"][-1], arrays["abs_vprime"][-1]]),
        axis=1,
    )
    theta = arrays["centers"][interior][len(arrays["centers"][interior]) // 2]
    _, expected_gradient, expected_hessian = _numpy_prediction(arrays, theta)

    def evaluate(point):
        return model._log10_spectrum(point[0], point[1])

    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(evaluate))(theta), expected_gradient, rtol=1e-9, atol=1e-8
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(evaluate))(theta), expected_hessian, rtol=1e-9, atol=1e-7
    )


def test_stored_frequency_axis_and_log_linear_interpolation(template, arrays):
    theta = np.array([np.mean(arrays["inv_f_tilde"]), np.mean(arrays["abs_vprime"])])
    log_spectrum = (
        _numpy_prediction(arrays, theta)[0]
        if template.interpolation == "matern52"
        else _numpy_bilinear(arrays, theta)
    )
    log_axis = arrays["log10_frequency_hz"]
    log_frequency = np.r_[log_axis, 0.37 * log_axis[:-1] + 0.63 * log_axis[1:]]
    actual = template.omega_gw_h2(10.0**log_frequency, *theta)
    np.testing.assert_allclose(
        np.log10(actual),
        np.interp(log_frequency, log_axis, log_spectrum),
        atol=1e-10,
        rtol=0,
    )
    low, high = 10.0 ** log_axis[[0, -1]]
    np.testing.assert_array_equal(template.frequency_bounds_hz, [low, high])
    outside = np.array(
        [np.nextafter(low, 0), np.nextafter(high, np.inf), 0, -1, np.nan, np.inf]
    )
    np.testing.assert_array_equal(template.omega_gw_h2(outside, *theta), 0)


def test_jit_vmap_and_scalar_frequency(template, arrays):
    frequencies = jnp.geomspace(*template.frequency_bounds_hz, 19)
    low = np.array([arrays["inv_f_tilde"][0], arrays["abs_vprime"][0]])
    high = np.array([arrays["inv_f_tilde"][-1], arrays["abs_vprime"][-1]])
    parameters = jnp.asarray(low + (high - low) * [[0.2, 0.3], [0.4, 0.6], [0.7, 0.8]])

    def evaluate(theta):
        return template.omega_gw_h2(frequencies, theta[0], theta[1])

    eager = np.stack([evaluate(theta) for theta in parameters])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(parameters), eager, rtol=1e-12
    )
    assert template.omega_gw_h2(float(frequencies[1]), *parameters[0]).shape == ()
    assert np.all(np.isfinite(eager)) and np.all(eager >= 0)


def test_original_invalid_parameters_are_only_clipped_for_spectrum(template, arrays):
    low = np.array([arrays["inv_f_tilde"][0], arrays["abs_vprime"][0]])
    high = np.array([arrays["inv_f_tilde"][-1], arrays["abs_vprime"][-1]])
    middle = (low + high) / 2
    frequencies = jnp.geomspace(*template.frequency_bounds_hz, 7)
    for index in range(2):
        for value in (low[index] - 1, high[index] + 1, np.nan, -np.inf, np.inf):
            point = middle.copy()
            point[index] = value
            safe = point.copy()
            safe[index] = np.clip(
                np.nan_to_num(
                    value, nan=low[index], neginf=low[index], posinf=high[index]
                ),
                low[index],
                high[index],
            )
            assert not bool(template.is_valid(*point))
            actual = template.omega_gw_h2(frequencies, *point)
            assert np.all(np.isfinite(actual))
            np.testing.assert_array_equal(
                actual, template.omega_gw_h2(frequencies, *safe)
            )


def test_validity_and_full_histories_match_bilinear(model, bilinear_model, arrays):
    for name in ("ratio_grad_over_kin_table", "sampled_node", "support_cell"):
        np.testing.assert_array_equal(
            getattr(model, name), getattr(bilinear_model, name)
        )
    x, y = arrays["inv_f_tilde"], arrays["abs_vprime"]
    x = np.sort(np.r_[x, (x[:-1] + x[1:]) / 2])
    y = np.sort(np.r_[y, (y[:-1] + y[1:]) / 2])
    points = jnp.asarray(np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2))
    for method in ("is_valid", "max_grad_over_kin"):

        def evaluate(template):
            return jax.jit(jax.vmap(lambda p: getattr(template, method)(p[0], p[1])))(
                points
            )

        np.testing.assert_array_equal(evaluate(model), evaluate(bilinear_model))


def test_full_history_before_maximum_and_strict_configurable_threshold(
    tmp_path, interpolation
):
    arrays = _small_arrays()
    arrays["ratio_grad_over_kin"][:] = [[[0.16, 0.0], [0.0, 0.16]]]
    custom = _external_model(tmp_path, arrays, interpolation=interpolation)
    assert float(custom.max_grad_over_kin(1.0, 0.5)) == pytest.approx(0.08)
    assert bool(custom.is_valid(1.0, 0.5))
    strict = AxionU1LinearSlope(
        data_file=tmp_path / "scan.npz",
        interpolation=interpolation,
        ratio_threshold=0.08,
    )
    assert not bool(strict.is_valid(1.0, 0.5))
    with pytest.raises(AttributeError):
        strict.validity_threshold = 0.2


def test_four_corner_support_and_right_hand_cell_selection(tmp_path, interpolation):
    arrays = _small_arrays(n_inv=3)
    arrays["sampled_node"][0, 2] = False
    custom = _external_model(tmp_path, arrays, interpolation=interpolation)
    assert bool(custom.is_valid(0.5, 0.5))
    assert not bool(custom.is_valid(1.0, 0.5))
    assert not bool(custom.is_valid(1.5, 0.5))
    assert np.isfinite(custom.omega_gw_h2(0.01, 1.5, 0.5))


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_nonfinite_ratio_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError):
        _external_model(tmp_path, _small_arrays(), ratio_threshold=threshold)


@pytest.mark.parametrize(
    "prior_bounds,expected_mass",
    [(None, 0.25), ((0.0, 0.5), 1.0), ((-2.0, 2.0), 0.125)],
)
def test_explicit_evidence_correction_uses_configured_prior_and_threshold(
    tmp_path, prior_bounds, expected_mass
):
    arrays = _small_arrays()
    arrays["ratio_grad_over_kin"][:, 1, :] = 1.0
    priors = (
        None
        if prior_bounds is None
        else {
            "inv_f_tilde": {"min": prior_bounds[0], "max": prior_bounds[1]},
            "abs_vprime": {"min": 0.0, "max": 1.0},
        }
    )
    custom = _external_model(
        tmp_path, arrays, ratio_threshold=0.25, prior_by_param=priors
    )
    assert custom.valid_prior_mass is None
    assert custom.valid_prior_mass_error is None
    assert custom.log_evidence_correction is None
    correction = custom.compute_evidence_correction()
    assert custom.valid_prior_mass == pytest.approx(expected_mass, abs=1e-8)
    assert 0 <= custom.valid_prior_mass_error <= 1e-8
    assert (
        correction
        == custom.log_evidence_correction
        == pytest.approx(-np.log(expected_mass))
    )
    assert custom.reference_prior_bounds["inv_f_tilde"] == (prior_bounds or (0.0, 2.0))


def test_evidence_correction_respects_support_and_all_ratio_times(tmp_path):
    arrays = _small_arrays(n_inv=3)
    arrays["sampled_node"][0, 2] = False
    arrays["ratio_grad_over_kin"][:, 0, :] = [0.0, 1.0]
    arrays["ratio_grad_over_kin"][:, 1, :] = [1.0, 0.0]
    custom = _external_model(tmp_path, arrays, ratio_threshold=0.75)
    custom.compute_evidence_correction()
    assert custom.valid_prior_mass == pytest.approx(0.25, abs=1e-8)


def test_evidence_correction_integrates_curved_boundary_with_restricted_prior(tmp_path):
    arrays = _small_arrays()
    arrays["ratio_grad_over_kin"][1, 1, :] = 1.0
    custom = _external_model(
        tmp_path,
        arrays,
        ratio_threshold=0.25,
        prior_by_param={
            "inv_f_tilde": {"min": 0.0, "max": 2.0},
            "abs_vprime": {"min": 0.25, "max": 0.75},
        },
    )
    custom.compute_evidence_correction()
    assert custom.valid_prior_mass == pytest.approx(0.5 * np.log(3.0), abs=1e-8)


def test_evidence_correction_rejects_zero_mass_and_nonuniform_prior(tmp_path):
    arrays = _small_arrays()
    zero_mass = _external_model(tmp_path, arrays, ratio_threshold=0.0)
    with pytest.raises(ValueError):
        zero_mass.compute_evidence_correction()
    assert zero_mass.log_evidence_correction is None
    nonuniform = _external_model(
        tmp_path,
        arrays,
        prior_by_param={
            "inv_f_tilde": {"mean": 1.0, "std": 0.2},
            "abs_vprime": {"min": 0.0, "max": 1.0},
        },
    )
    assert np.isfinite(nonuniform.omega_gw_h2(0.01, 1.0, 0.5))
    with pytest.raises(ValueError):
        nonuniform.compute_evidence_correction()


def test_packaged_prior_and_normalization_are_interpolator_independent(arrays):
    custom = AxionU1LinearSlope()
    bilinear_model = AxionU1LinearSlope(interpolation="bilinear")
    assert custom.prior_by_param == bilinear_model.prior_by_param
    assert custom.prior_by_param == {
        name: {"min": arrays[name][0], "max": arrays[name][-1]}
        for name in custom.parameter_names
    }
    assert custom.validity_threshold == bilinear_model.validity_threshold
    custom.compute_evidence_correction()
    bilinear_model.compute_evidence_correction()
    assert custom.valid_prior_mass == bilinear_model.valid_prior_mass
    assert custom.valid_prior_mass_error == bilinear_model.valid_prior_mass_error
    assert custom.log_evidence_correction == pytest.approx(
        -np.log(custom.valid_prior_mass)
    )


def test_matern_derivative_is_smooth_across_bilinear_ridge(tmp_path):
    arrays = _small_arrays(n_inv=3)
    arrays["log10_omega_gw_h2"][:] = np.array([-10.0, -9.0, -10.0])[None, :, None]
    smooth = _external_model(tmp_path, arrays)
    linear = AxionU1LinearSlope(tmp_path / "scan.npz", interpolation="bilinear")
    boundary = arrays["inv_f_tilde"][1]
    epsilon = 1e-7
    for template, continuous in ((smooth, True), (linear, False)):
        derivative = jax.jacfwd(lambda x: template._log10_spectrum(x, 0.5))
        left = np.asarray(derivative(boundary - epsilon))
        right = np.asarray(derivative(boundary + epsilon))
        if continuous:
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_allclose(left - right, 2.0, atol=1e-14)
