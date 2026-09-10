"""Prior mass of a bilinearly interpolated ratio-history constraint."""

import numpy as np
from scipy.integrate import quad


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
    estimates quadrature error, with a floating-point floor; it does not estimate
    the physical or interpolation uncertainty of the input ratios.
    """
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
