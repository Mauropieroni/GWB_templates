"""
Abelian Higgs Model II (2 parameters).

Scales the :class:`CosmicStringModelII` (BOS) spectrum by an overall amplitude,
interpolating continuously between the Nambu-Goto and Abelian Higgs limits.
Reuses that model's precomputed 2D data grid and JAX bilinear interpolation
primitives, so JAX automatic differentiation works here too.

Reference: arXiv:1309.6637 (Blanco-Pillado, Olum & Shlaer — original BOS
           loop-number-density distribution);
           arXiv:1909.00819 (Auclair et al. — BOS P_n distribution applied
           to LISA cosmic-string forecasts);
           arXiv:2405.03740 (GW from cosmic strings in LISA: reconstruction
           pipeline and physics interpretation).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import NumericalTemplate, DifferentiationBackend
from gwb_templates.cosmic_string_templates.cosmic_string_model_ii import (
    ArrayLike,
    _DEFAULT_DATA_FILENAME,
    _bilinear_dS_dix,
    _bilinear_eval,
    _load_grid,
    _to_frac_ix,
)


class AbelianHiggsModelII(NumericalTemplate):
    r"""
    Abelian Higgs Model II.

    Scales the BOS Model II spectrum by an overall amplitude
    :math:`10^{\log_{10} f_{\mathrm{NG}}}`, allowing a continuous interpolation between
    the Nambu-Goto and Abelian Higgs limits.

    Free parameters
    ---------------
    log_Gmu
        :math:`\log_{10}` of the string tension :math:`G\mu`.
    logf
        :math:`\log_{10}` of the Nambu-Goto-vs-Abelian-Higgs amplitude
        scaling :math:`f_{\mathrm{NG}}`.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
    @article{Auclair:2019wcv,
        author = "Auclair, Pierre and others",
        title = "{Probing the gravitational wave background from cosmic strings with
            LISA}",
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
            Lewicki, Marek and Nardini, Germano and Pieroni, Mauro and Rybak, Ivan Yu.
            and Sousa, Lara and Wachter, Jeremy M.",
        collaboration = "LISA Cosmology Working Group",
        title = "{Gravitational waves from cosmic strings in LISA: reconstruction
            pipeline and physics interpretation}",
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
        self.n_gmu: int = int(gmu_axis.shape[0])
        self.n_freq_grid: int = int(freq_axis.shape[0])

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_Gmu: ArrayLike,
        logf: ArrayLike,
    ) -> jax.Array:
        log10_f = jnp.log10(jnp.asarray(frequency))
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)
        spectrum = 10.0 ** _bilinear_eval(
            ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid
        )
        return jnp.asarray(10.0**logf * spectrum)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        r"""Analytical :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`."""
        log_Gmu = theta[0]
        logf = theta[1]
        log10_f = jnp.log10(jnp.asarray(frequency))
        ix = _to_frac_ix(log_Gmu, self.gmu_axis)
        iy = _to_frac_ix(log10_f, self.freq_axis)

        S = _bilinear_eval(ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid)
        h2_omega_ii = 10.0**S
        h2_omega_ah = 10.0**logf * h2_omega_ii

        # d/d(log_Gmu)
        d_ix_d_log_Gmu = (self.n_gmu - 1) / (self.gmu_axis[-1] - self.gmu_axis[0])
        dS_dix = _bilinear_dS_dix(
            ix, iy, self.log10_omega, self.n_gmu, self.n_freq_grid
        )
        d_d_log_Gmu = 10.0**logf * jnp.log(10.0) * h2_omega_ii * dS_dix * d_ix_d_log_Gmu
        # d/d(logf)
        d_d_logf = jnp.log(10.0) * h2_omega_ah

        return jnp.stack([d_d_log_Gmu, d_d_logf], axis=-1)
