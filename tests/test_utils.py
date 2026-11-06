import unittest

import jax.numpy as jnp

from gwb_templates.utils import (
    finite_difference_d2f2,
    finite_difference_d2f_dtheta,
    finite_difference_df,
    finite_difference_grad_theta,
    finite_difference_hess_theta,
)

# Smooth test function with nonzero first/second derivatives in both frequency
# and parameters, and nonzero mixed derivatives, so every finite-difference
# routine below is exercised against a nontrivial analytic ground truth.
#
#   f(freq, [a, b, c]) = a * sin(b * freq) + c * freq**2


def _f(freq, params):
    a, b, c = params
    return a * jnp.sin(b * freq) + c * freq**2


FREQ = jnp.geomspace(1e-3, 1.0, 20)
SCALAR_FREQ = jnp.array(0.5)
PARAMS = jnp.array([2.0, 3.0, 1.5])
A, B, C = PARAMS


def _analytic_grad_theta(freq):
    """d f / d(a, b, c) at each frequency, shape (..., 3)."""
    return jnp.stack(
        [jnp.sin(B * freq), A * freq * jnp.cos(B * freq), freq**2], axis=-1
    )


def _analytic_hess_theta(freq):
    """d^2 f / d(a, b, c)^2 at each frequency, shape (..., 3, 3)."""
    hess = jnp.zeros(freq.shape + (3, 3))
    hess = hess.at[..., 1, 1].set(-A * freq**2 * jnp.sin(B * freq))
    cross = freq * jnp.cos(B * freq)
    hess = hess.at[..., 0, 1].set(cross)
    hess = hess.at[..., 1, 0].set(cross)
    return hess


def _analytic_df(freq):
    """d f / d(freq)."""
    return A * B * jnp.cos(B * freq) + 2.0 * C * freq


def _analytic_d2f2(freq):
    """d^2 f / d(freq)^2."""
    return -A * B**2 * jnp.sin(B * freq) + 2.0 * C


def _analytic_d2f_dtheta(freq):
    """d^2 f / d(freq) d(a, b, c), shape (..., 3)."""
    return jnp.stack(
        [
            B * jnp.cos(B * freq),
            A * jnp.cos(B * freq) - A * B * freq * jnp.sin(B * freq),
            2.0 * freq,
        ],
        axis=-1,
    )


class TestFiniteDifferenceGradTheta(unittest.TestCase):
    """First-order central differences (step=1e-6) w.r.t. parameters."""

    def test_vector_frequency(self):
        grad = finite_difference_grad_theta(_f, FREQ, PARAMS)
        self.assertEqual(grad.shape, FREQ.shape + (3,))
        self.assertTrue(jnp.allclose(grad, _analytic_grad_theta(FREQ), atol=1e-6))

    def test_scalar_frequency(self):
        grad = finite_difference_grad_theta(_f, SCALAR_FREQ, PARAMS)
        self.assertTrue(
            jnp.allclose(grad, _analytic_grad_theta(SCALAR_FREQ), atol=1e-6)
        )


class TestFiniteDifferenceHessTheta(unittest.TestCase):
    """Second-order central differences (step=1e-5) w.r.t. parameters."""

    def test_vector_frequency(self):
        hess = finite_difference_hess_theta(_f, FREQ, PARAMS)
        self.assertEqual(hess.shape, FREQ.shape + (3, 3))
        self.assertTrue(jnp.allclose(hess, _analytic_hess_theta(FREQ), atol=1e-4))

    def test_symmetric(self):
        hess = finite_difference_hess_theta(_f, FREQ, PARAMS)
        self.assertTrue(jnp.allclose(hess, jnp.swapaxes(hess, -1, -2)))


class TestFiniteDifferenceDf(unittest.TestCase):
    """First-order central differences (step=1e-6) w.r.t. frequency."""

    def test_vector_frequency(self):
        df = finite_difference_df(_f, FREQ, PARAMS)
        self.assertEqual(df.shape, FREQ.shape)
        self.assertTrue(jnp.allclose(df, _analytic_df(FREQ), atol=1e-6))

    def test_scalar_frequency(self):
        df = finite_difference_df(_f, SCALAR_FREQ, PARAMS)
        self.assertTrue(jnp.allclose(df, _analytic_df(SCALAR_FREQ), atol=1e-6))


class TestFiniteDifferenceD2f2(unittest.TestCase):
    """Second-order central differences (step=1e-5) w.r.t. frequency."""

    def test_vector_frequency(self):
        d2f2 = finite_difference_d2f2(_f, FREQ, PARAMS)
        self.assertEqual(d2f2.shape, FREQ.shape)
        self.assertTrue(jnp.allclose(d2f2, _analytic_d2f2(FREQ), atol=1e-4))

    def test_scalar_frequency(self):
        d2f2 = finite_difference_d2f2(_f, SCALAR_FREQ, PARAMS)
        self.assertTrue(jnp.allclose(d2f2, _analytic_d2f2(SCALAR_FREQ), atol=1e-4))


class TestFiniteDifferenceD2fDtheta(unittest.TestCase):
    """Mixed frequency/parameter central differences (step_f=1e-5, step_theta=1e-6)."""

    def test_vector_frequency(self):
        mixed = finite_difference_d2f_dtheta(_f, FREQ, PARAMS)
        self.assertEqual(mixed.shape, FREQ.shape + (3,))
        self.assertTrue(jnp.allclose(mixed, _analytic_d2f_dtheta(FREQ), atol=1e-3))

    def test_scalar_frequency(self):
        mixed = finite_difference_d2f_dtheta(_f, SCALAR_FREQ, PARAMS)
        self.assertTrue(
            jnp.allclose(mixed, _analytic_d2f_dtheta(SCALAR_FREQ), atol=1e-3)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
