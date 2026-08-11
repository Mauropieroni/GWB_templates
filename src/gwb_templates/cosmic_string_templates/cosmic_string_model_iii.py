"""
Cosmic String Model III (1 parameter).

Model III uses a precomputed 2D data grid over (log_Gmu, log10_frequency) and
evaluates h^2 * Omega_GW via JAX bilinear interpolation so that JAX automatic
differentiation works.

Reference: arXiv:astro-ph/0511646 (Ringeval, Sakellariadou & Bouchet — Nambu-Goto simulations);
           arXiv:1006.0931 (Lorenz, Ringeval & Sakellariadou — LRS Loop distribution);
           arXiv:1903.06685 (Auclair et al. — Extension of the model).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np

from gwb_templates.template import NumericalTemplate

ArrayLike: TypeAlias = jtp.ArrayLike

_DEFAULT_DATA_FILENAME = "Model-III_LRS-loggrid.dat"
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_grid(filename: str) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Load the precomputed Model III data grid.

    Layout of the .dat file:
      * Row 0, cols 1..  → log10_frequency axis
      * Col 0, rows 1..  → log_Gmu axis
      * Submatrix [1:, 1:] → log10(h^2 Omega_GW)

    Returns:
        (gmu_axis, freq_axis, log10_omega) as JAX arrays.
    """
    path = os.path.join(_DATA_DIR, filename)
    data_np = np.loadtxt(path)
    gmu_axis = jnp.array(data_np[1:, 0])
    freq_axis = jnp.array(data_np[0, 1:])
    log10_omega = jnp.array(data_np[1:, 1:])
    return gmu_axis, freq_axis, log10_omega


# ── JAX bilinear interpolation primitives ────────────────────────────────────


def _to_frac_ix(val: ArrayLike, axis: jax.Array) -> jax.Array:
    """Convert a physical value to a fractional grid index along ``axis``."""
    n = axis.shape[0]
    return (val - axis[0]) / (axis[-1] - axis[0]) * (n - 1)


def _bilinear_eval(
    ix: jax.Array,
    iy: jax.Array,
    log10_omega: jax.Array,
    n_gmu: int,
    n_freq: int,
) -> jax.Array:
    """Bilinear interpolation of log10(h^2 Omega) at fractional indices."""
    kx = jnp.clip(
        jnp.floor(jnp.clip(ix, 0.0, n_gmu - 1.0)).astype(jnp.int32),
        0,
        n_gmu - 2,
    )
    tx = jnp.clip(ix - kx, 0.0, 1.0)

    ky = jnp.clip(
        jnp.floor(jnp.clip(iy, 0.0, n_freq - 1.0)).astype(jnp.int32),
        0,
        n_freq - 2,
    )
    ty = jnp.clip(iy - ky, 0.0, 1.0)

    row0 = log10_omega[kx]
    row1 = log10_omega[kx + 1]
    g00 = row0[ky]
    g01 = row0[ky + 1]
    g10 = row1[ky]
    g11 = row1[ky + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * g00
        + tx * (1.0 - ty) * g10
        + (1.0 - tx) * ty * g01
        + tx * ty * g11
    )


def _bilinear_dS_dix(
    ix: jax.Array,
    iy: jax.Array,
    log10_omega: jax.Array,
    n_gmu: int,
    n_freq: int,
) -> jax.Array:
    """Analytical dS/d(ix) for the bilinear interpolation."""
    kx = jnp.clip(
        jnp.floor(jnp.clip(ix, 0.0, n_gmu - 1.0)).astype(jnp.int32),
        0,
        n_gmu - 2,
    )
    ky = jnp.clip(
        jnp.floor(jnp.clip(iy, 0.0, n_freq - 1.0)).astype(jnp.int32),
        0,
        n_freq - 2,
    )
    ty = jnp.clip(iy - ky, 0.0, 1.0)

    row0 = log10_omega[kx]
    row1 = log10_omega[kx + 1]
    g00 = row0[ky]
    g01 = row0[ky + 1]
    g10 = row1[ky]
    g11 = row1[ky + 1]

    return (1.0 - ty) * (g10 - g00) + ty * (g11 - g01)


# ── Template classes ──────────────────────────────────────────────────────────


class CosmicStringModelIII(NumericalTemplate):
    r"""
    Cosmic String Model III (arXiv:1006.0931, Cusp dominated :math:`P_n`).

    1-parameter model evaluated from a precomputed data grid via bilinear
    interpolation. The template is JAX-differentiable since the interpolation
    is implemented in pure JAX.

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`. Grid range
        :math:`-18 \le \log G\mu \le -9.5`.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Ringeval:2005kr,
    author = "Ringeval, Christophe and Sakellariadou, Mairi and Bouchet, Francois",
    title = "{Cosmological evolution of cosmic string loops}",
    eprint = "astro-ph/0511646",
    archivePrefix = "arXiv",
    doi = "10.1088/1475-7516/2007/02/023",
    journal = "JCAP",
    volume = "02",
    pages = "023",
    year = "2007"
}
""",
        r"""
@article{Lorenz:2010sm,
    author = "Lorenz, Larissa and Ringeval, Christophe and Sakellariadou, Mairi",
    title = "{Cosmic string loop distribution on all length scales and at any redshift}",
    eprint = "1006.0931",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2010/10/003",
    journal = "JCAP",
    volume = "10",
    pages = "003",
    year = "2010"
}
""",
        r"""
@article{Auclair:2019zoz,
    author = "Auclair, Pierre and Ringeval, Christophe and Sakellariadou, Mairi and Steer, Daniele",
    title = "{Cosmic string loop production functions}",
    eprint = "1903.06685",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "KCL-PH-TH/2019-19",
    doi = "10.1088/1475-7516/2019/06/015",
    journal = "JCAP",
    volume = "06",
    pages = "015",
    year = "2019"
}
""",
    )

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
            data_filename: Filename of the precomputed grid inside the
                ``cosmic_string_templates/data`` directory.
        """
        self.data_filename: str = str(data_filename)

        # setup() will populate these; we need them after super().__init__
        # to be able to read the grid extrema for defaulting priors. So we
        # load the grid once eagerly here just to peek at the gmu range.
        gmu_axis, _, _ = _load_grid(self.data_filename)
        log_gmu_min = float(gmu_axis[0])
        log_gmu_max = float(gmu_axis[-1])

        default_labels = {"log_Gmu": r"$\log_{10}(G\mu)$"}
        default_priors = {"log_Gmu": {"min": log_gmu_min, "max": log_gmu_max}}

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Cosmic String Model III"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        """Load the precomputed (log_Gmu, log10_f) grid into JAX arrays."""
        gmu_axis, freq_axis, log10_omega = _load_grid(self.data_filename)
        self.gmu_axis: jax.Array = gmu_axis
        self.freq_axis: jax.Array = freq_axis
        self.log10_omega: jax.Array = log10_omega
        self.n_gmu: int = int(gmu_axis.shape[0])
        self.n_freq_grid: int = int(freq_axis.shape[0])

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_Gmu: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate :math:`\Omega_{\mathrm{GW}} h^2(f)` for Model II."""
        log10_f = jnp.log10(jnp.asarray(frequency))
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)
        return 10.0 ** _bilinear_eval(
            ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        r"""Analytical :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`."""
        log_Gmu = theta[0]
        log10_f = jnp.log10(jnp.asarray(frequency))
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)

        S = _bilinear_eval(ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid)
        h2_omega = 10.0**S

        d_ix_d_log_Gmu = (self.n_gmu - 1) / (self.gmu_axis[-1] - self.gmu_axis[0])
        dS_dix = _bilinear_dS_dix(
            ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid
        )

        grad = jnp.log(10.0) * h2_omega * dS_dix * d_ix_d_log_Gmu
        return grad[..., None]
