"""
Cosmic String Model II with non-SM degree-of-freedom changes (3 parameters).

Extends :class:`CosmicStringModelII` with two extra parameters describing a BSM
species that becomes relativistic at temperature ``T_delta``, adding ``delta_g``
degrees of freedom. Backed by a precomputed 4D data grid over (log_Gmu,
log_T_delta, delta_g, log10_frequency), evaluated via JAX quadrilinear
interpolation so that JAX automatic differentiation works.

Reference: arXiv:1909.00819 (BOS P_n loop distribution);
           arXiv:2405.03740 (GW from cosmic strings in LISA).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from gwb_templates.template import NumericalTemplate, DifferentiationBackend

ArrayLike: TypeAlias = float | int | np.ndarray | jax.Array

_DEFAULT_DATA_FILENAME = "DOF-Model-II_BOS-arrays.npz"
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_grid_4d(
    filename: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Load the precomputed (log_Gmu, log_T_delta, delta_g, log10_f) grid."""
    path = os.path.join(_DATA_DIR, filename)
    data = np.load(path)
    gmu_axis = jnp.array(data["log_Gmu"])
    t_delta_axis = jnp.array(data["log_T_delta"])
    delta_g_axis = jnp.array(data["delta_g"])
    freq_axis = jnp.array(data["log_freq"])
    log10_omega = jnp.array(data["cdf_grid"])
    return gmu_axis, t_delta_axis, delta_g_axis, freq_axis, log10_omega


# ── JAX quadrilinear interpolation ────────────────────────────────────────────


def _to_frac_ix(val: ArrayLike, axis: jax.Array) -> jax.Array:
    """Convert a physical value to a fractional grid index along ``axis``.

    Unlike ``cosmic_string_model_ii._to_frac_ix``, this doesn't assume ``axis``
    is uniformly spaced: the delta_g axis here isn't (checked -- [0, 8.16, 23.09,
    42.43, 65.32, 91.29, 120]), matching how scipy.interpolate.RegularGridInterpolator
    (what the reference uses) handles it. jnp.searchsorted finds the bracketing
    interval; the fractional position within it is then a plain linear interpolation.
    """
    n = axis.shape[0]
    idx = jnp.clip(jnp.searchsorted(axis, val, side="right") - 1, 0, n - 2)
    x0 = axis[idx]
    x1 = axis[idx + 1]
    return idx.astype(x0.dtype) + (val - x0) / (x1 - x0)


def _quadrilinear_eval(
    i0: jax.Array,
    i1: jax.Array,
    i2: jax.Array,
    i3: jax.Array,
    grid: jax.Array,
    n0: int,
    n1: int,
    n2: int,
    n3: int,
) -> jax.Array:
    """4D generalization of ``cosmic_string_model_ii._bilinear_eval``.

    i0, i1, i2 are scalar (log_Gmu, log_T_delta, delta_g); i3 may be an array (log10
    frequency), so the result matches i3's shape. Interpolates via the 16-corner
    weighted sum, same construction as the 2D case's 4 corners.
    """

    def _floor_clip(i: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
        k = jnp.clip(jnp.floor(jnp.clip(i, 0.0, n - 1.0)).astype(jnp.int32), 0, n - 2)
        t = jnp.clip(i - k, 0.0, 1.0)
        return k, t

    k0, t0 = _floor_clip(i0, n0)
    k1, t1 = _floor_clip(i1, n1)
    k2, t2 = _floor_clip(i2, n2)
    k3, t3 = _floor_clip(i3, n3)

    result = jnp.zeros_like(t3)
    for b0 in (0, 1):
        w0 = t0 if b0 else 1.0 - t0
        for b1 in (0, 1):
            w1 = w0 * (t1 if b1 else 1.0 - t1)
            for b2 in (0, 1):
                w2 = w1 * (t2 if b2 else 1.0 - t2)
                for b3 in (0, 1):
                    w3 = w2 * (t3 if b3 else 1.0 - t3)
                    corner = grid[k0 + b0, k1 + b1, k2 + b2, k3 + b3]
                    result = result + w3 * corner
    return result


# ── Template class ────────────────────────────────────────────────────────────


class CosmicStringModelIIDof(NumericalTemplate):
    r"""
    Cosmic String Model II with non-SM DOF changes (arXiv:1909.00819, BOS).

    Adds a BSM species that becomes relativistic at temperature ``T_delta``,
    contributing ``delta_g`` extra degrees of freedom, on top of the
    :class:`CosmicStringModelII` spectrum.

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`. Grid range
        :math:`-10.1 \le \log G\mu \le -9.9`.
    log_T_delta
        :math:`\log_{10}` of the BSM transition temperature (GeV).
    delta_g
        Number of extra relativistic degrees of freedom.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = ()  # TODO: cite

    def __init__(
        self,
        data_filename: str = _DEFAULT_DATA_FILENAME,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            data_filename: Filename of the precomputed 4D grid inside the
                ``cosmic_string_templates/data`` directory.
        """
        self.data_filename: str = str(data_filename)

        gmu_axis, t_delta_axis, delta_g_axis, _, _ = _load_grid_4d(
            self.data_filename
        )

        default_labels = {
            "log_Gmu": r"$\log_{10}(G\mu)$",
            "log_T_delta": r"$\log_{10}(T_\Delta/\mathrm{GeV})$",
            "delta_g": r"$\Delta g$",
        }
        default_priors = {
            "log_Gmu": {"min": float(gmu_axis[0]), "max": float(gmu_axis[-1])},
            "log_T_delta": {
                "min": float(t_delta_axis[0]),
                "max": float(t_delta_axis[-1]),
            },
            "delta_g": {
                "min": float(delta_g_axis[0]),
                "max": float(delta_g_axis[-1]),
            },
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Cosmic String Model II (DOF)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        """Load the precomputed 4D grid into JAX arrays."""
        gmu_axis, t_delta_axis, delta_g_axis, freq_axis, log10_omega = _load_grid_4d(
            self.data_filename
        )
        self.gmu_axis: jax.Array = gmu_axis
        self.t_delta_axis: jax.Array = t_delta_axis
        self.delta_g_axis: jax.Array = delta_g_axis
        self.freq_axis: jax.Array = freq_axis
        self.log10_omega: jax.Array = log10_omega
        self.n_gmu: int = int(gmu_axis.shape[0])
        self.n_t_delta: int = int(t_delta_axis.shape[0])
        self.n_delta_g: int = int(delta_g_axis.shape[0])
        self.n_freq_grid: int = int(freq_axis.shape[0])

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_Gmu: ArrayLike,
        log_T_delta: ArrayLike,
        delta_g: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate :math:`\Omega_{\mathrm{GW}} h^2(f)` for Model II (DOF)."""
        log10_f = jnp.log10(jnp.asarray(frequency))
        i0 = _to_frac_ix(log_Gmu, self.gmu_axis)
        i1 = _to_frac_ix(log_T_delta, self.t_delta_axis)
        i2 = _to_frac_ix(delta_g, self.delta_g_axis)
        i3 = _to_frac_ix(log10_f, self.freq_axis)
        return 10.0 ** _quadrilinear_eval(
            i0,
            i1,
            i2,
            i3,
            self.log10_omega,
            self.n_gmu,
            self.n_t_delta,
            self.n_delta_g,
            self.n_freq_grid,
        )
