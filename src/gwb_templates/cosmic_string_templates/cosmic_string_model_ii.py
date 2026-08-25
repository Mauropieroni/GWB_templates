"""
Cosmic String Model II (1 parameter) and Abelian Higgs Model II (2 parameters).

Model II uses a precomputed 2D data grid over (log_Gmu, log10_frequency) and
evaluates h^2 * Omega_GW via JAX bilinear interpolation so that JAX automatic
differentiation works.

Reference: arXiv:1309.6637 (Blanco-Pillado, Olum & Shlaer — original BOS
           loop-number-density distribution);
           arXiv:1909.00819 (Auclair et al. — BOS P_n distribution applied
           to LISA cosmic-string forecasts);
           arXiv:2405.03740 (GW from cosmic strings in LISA: reconstruction
           pipeline and physics interpretation).
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
from gwb_templates.utils import bilinear_interp as _bilinear_interp
from gwb_templates.utils import to_frac_ix as _to_frac_ix

ArrayLike: TypeAlias = jtp.ArrayLike

_DEFAULT_DATA_FILENAME = "Model-II_BOS-loggrid.dat"
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_grid(filename: str) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Load the precomputed Model II data grid.

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


# ── Template classes ──────────────────────────────────────────────────────────


class CosmicStringModelII(NumericalTemplate):
    r"""
    Cosmic String Model II (arXiv:1909.00819, BOS :math:`P_n`).

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
@article{Auclair:2019wcv,
    author = "Auclair, Pierre and others",
    title = "{Probing the gravitational wave background from cosmic strings with LISA}",
    eprint = "1909.00819",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2020/04/034",
    journal = "JCAP",
    volume = "04",
    pages = "034",
    year = "2020"
}
""",
        r"""
@article{Blanco-Pillado:2024aca,
    author = "Blanco-Pillado, Jose J. and Cui, Yanou and Kuroyanagi, Sachiko and
        Lewicki, Marek and Nardini, Germano and Pieroni, Mauro and Rybak, Ivan Yu. and
        Sousa, Lara and Wachter, Jeremy M.",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from cosmic strings in LISA: reconstruction pipeline
        and physics interpretation}",
    eprint = "2405.03740",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-02, CERN-TH-2024-085",
    doi = "10.1088/1475-7516/2025/05/006",
    journal = "JCAP",
    volume = "05",
    pages = "006",
    year = "2025"
}
""",
        r"""
@article{Blanco-Pillado:2013qja,
    author = "Blanco-Pillado, Jose J. and Olum, Ken D. and Shlaer, Benjamin",
    title = "{The number of cosmic string loops}",
    eprint = "1309.6637",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.89.023512",
    journal = "Phys. Rev. D",
    volume = "89",
    number = "2",
    pages = "023512",
    year = "2014"
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
                model_label if model_label is not None else "Cosmic String Model II"
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

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_Gmu: ArrayLike,
    ) -> jax.Array:
        r"""Evaluate :math:`\Omega_{\mathrm{GW}} h^2(f)` for Model II."""
        log10_f = jnp.log10(frequency)
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)
        return 10.0 ** _bilinear_interp(ix, iy, self.log10_omega)


class AbelianHiggsModelII(NumericalTemplate):
    r"""
    Abelian Higgs Model II.

    Scales the BOS Model II spectrum by an overall amplitude
    :math:`10^{\log_{10} f_{\mathrm{NG}}}`, allowing a continuous
    interpolation between the Nambu-Goto and Abelian Higgs limits.

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`.
    logf
        :math:`\log_{10}` of the Nambu-Goto-vs-Abelian-Higgs amplitude
        scaling :math:`f_{\mathrm{NG}}`.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Auclair:2019wcv,
    author = "Auclair, Pierre and others",
    title = "{Probing the gravitational wave background from cosmic strings with LISA}",
    eprint = "1909.00819",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2020/04/034",
    journal = "JCAP",
    volume = "04",
    pages = "034",
    year = "2020"
}
""",
        r"""
@article{Blanco-Pillado:2024aca,
    author = "Blanco-Pillado, Jose J. and Cui, Yanou and Kuroyanagi, Sachiko and
        Lewicki, Marek and Nardini, Germano and Pieroni, Mauro and Rybak, Ivan Yu. and
        Sousa, Lara and Wachter, Jeremy M.",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from cosmic strings in LISA: reconstruction pipeline
        and physics interpretation}",
    eprint = "2405.03740",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-02, CERN-TH-2024-085",
    doi = "10.1088/1475-7516/2025/05/006",
    journal = "JCAP",
    volume = "05",
    pages = "006",
    year = "2025"
}
""",
        r"""
@article{Blanco-Pillado:2013qja,
    author = "Blanco-Pillado, Jose J. and Olum, Ken D. and Shlaer, Benjamin",
    title = "{The number of cosmic string loops}",
    eprint = "1309.6637",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.89.023512",
    journal = "Phys. Rev. D",
    volume = "89",
    number = "2",
    pages = "023512",
    year = "2014"
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
        self.data_filename: str = str(data_filename)

        gmu_axis, _, _ = _load_grid(self.data_filename)
        log_gmu_min = float(gmu_axis[0])
        log_gmu_max = float(gmu_axis[-1])

        default_labels = {
            "log_Gmu": r"$\log_{10}(G\mu)$",
            "logf": r"$\log_{10}(f_\mathrm{NG})$",
        }
        default_priors = {
            "log_Gmu": {"min": log_gmu_min, "max": log_gmu_max},
            "logf": {"min": -3.0, "max": 3.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Abelian Higgs Model II"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def setup(self) -> None:
        gmu_axis, freq_axis, log10_omega = _load_grid(self.data_filename)
        self.gmu_axis: jax.Array = gmu_axis
        self.freq_axis: jax.Array = freq_axis
        self.log10_omega: jax.Array = log10_omega

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_Gmu: ArrayLike,
        logf: ArrayLike,
    ) -> jax.Array:
        log10_f = jnp.log10(frequency)
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)
        spectrum = 10.0 ** _bilinear_interp(ix, iy, self.log10_omega)
        return 10.0**logf * spectrum
