"""Analytic checks of ratio-history prior normalization."""

import numpy as np
import pytest

from gwb_templates.inflation_templates._axion_validity import _valid_prior_mass


def _unit_square_mass(ratios, threshold, bounds=((0.0, 1.0), (0.0, 1.0))):
    axis = np.array([0.0, 1.0])
    return _valid_prior_mass(
        axis, axis, ratios, np.ones((1, 1), dtype=bool), bounds, threshold, 1e-10
    )


def test_triangle_constraint():
    ratios = np.array([[[0.0], [1.0]], [[1.0], [2.0]]])
    mass, error = _unit_square_mass(ratios, 1.0)
    assert mass == pytest.approx(0.5, abs=1e-12)
    assert 0.0 < error < 1e-10


def test_product_constraint():
    ratios = np.array([[[0.0], [0.0]], [[0.0], [1.0]]])
    threshold = 0.25
    expected = threshold * (1.0 - np.log(threshold))
    mass, error = _unit_square_mass(ratios, threshold)
    assert mass == pytest.approx(expected, abs=1e-12)
    assert 0.0 < error < 1e-9


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (((0.25, 0.75), (-0.5, 1.5)), 0.25),
        (((-1.0, 2.0), (-1.0, 2.0)), 0.5 / 9.0),
        (((2.0, 3.0), (0.0, 1.0)), 0.0),
    ],
)
def test_prior_clipping_uses_full_prior_area(bounds, expected):
    ratios = np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
    mass, _ = _unit_square_mass(ratios, 0.5, bounds)
    assert mass == pytest.approx(expected, abs=1e-12)


def test_unsupported_cells_do_not_contribute():
    mass, _ = _valid_prior_mass(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0]),
        np.zeros((2, 3, 1)),
        np.array([[True, False]]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        0.1,
        1e-10,
    )
    assert mass == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("value, expected", [(0.0, 1.0), (0.1, 0.0), (0.2, 0.0)])
def test_constant_ratio_uses_strict_threshold(value, expected):
    mass, _ = _unit_square_mass(np.full((2, 2, 1), value), 0.1)
    assert mass == expected


def test_narrow_interior_region_from_multiple_histories():
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    center, half_width = 0.51234, 0.005
    ratios = np.stack(
        (
            0.1 + 0.05 * (x * y - 2 * center * x + center**2 - half_width**2),
            0.1 + 0.05 * (x - y),
        ),
        axis=-1,
    )
    # Validity requires x < y < 2*c - (c*c-e*e)/x, hence c-e < x < c+e.
    left, right = center - half_width, center + half_width
    expected = (
        2 * center * (right - left)
        - (right**2 - left**2) / 2
        - (center**2 - half_width**2) * np.log(right / left)
    )
    assert np.all(ratios > 0.0)
    mass, _ = _unit_square_mass(ratios, 0.1)
    assert mass == pytest.approx(expected, rel=1e-8, abs=1e-16)
