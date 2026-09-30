"""Parity-even scalar-induced gravitational-wave template."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.scalar_induced_templates.base import ScalarInducedTemplate
from gwb_templates.template import NumericalTemplate

from gwb_templates.scalar_induced_templates.parity_violation.helper_functions import (
    evaluate_components_jax,
    interpolate_components_jax,
    normalization_factor_gradients_jax,
)


class even_sigw(ScalarInducedTemplate, NumericalTemplate):
    r"""
    Non-parity-violating SIGW arising from gaussian power spectrum, bispectrum,
    even trispectrum (arxiv:2507.02733,2607.16162).
    Primordial power spectrum set as broken power law, with IR index equal to 4,
    (steepest possible growth in canonical single-field inflation - arxiv:1811.11158).
    Partially jaxed interpolation of precomputed grid through helper functions.

    Free parameters
    ---------------
    target_f_peak
        Peak frequency in Hz.
    target_n2
        UV spectral index.
    log10_A_zeta
        :math:`\log_{10}` amplitude of the primordial scalar spectrum.
    log10_f_NL
        :math:`\log_{10}` of the bispectrum coefficient :math:`f_{NL}`.
    log10_tau_NL
        :math:`\log_{10}` of the even trispectrum coefficient :math:`\tau_{NL}`.
    log10_tilde_tau_NL
        :math:`\log_{10}` of the odd trispectrum coefficient.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Ragavendra:2025svk,
    author = "Ragavendra, H. V. and Bartolo, Nicola",
    title = "{Twisted echoes of an odd quartet: Scalar-induced gravitational waves
    as a probe of primordial parity-violation}",
    eprint = "2507.02733",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "7",
    year = "2025"
},
@article{Caporali:2026qhc,
    author = "Caporali, Ilaria and Ragavendra, H. V. and Ricciardone, Angelo and
    Bartolo, Nicola",
    title = "{Constraining primordial non-Gaussianity and parity-violation through
    Scalar-Induced Gravitational Waves with next-generation ground-based
    interferometers}",
    eprint = "2607.16162",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "7",
    year = "2026"
},

""",
    )

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[str] = "autodiff"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        labels = {
            "target_f_peak": r"$f_*\,[\mathrm{Hz}]$",
            "target_n2": r"$n_2$",
            "log10_A_zeta": r"$\log_{10}A_\zeta$",
            "log10_f_NL": r"$\log_{10}f_{\rm NL}$",
            "log10_tau_NL": r"$\log_{10}\tau_{\rm NL}$",
        }
        priors = {
            "target_f_peak": {"min": 1e-6, "max": 1e4},
            "target_n2": {"min": 0.1, "max": 1.0},
            "log10_A_zeta": {"min": -4.0, "max": 0.0},
            "log10_f_NL": {"min": -4.0, "max": 2.0},
            "log10_tau_NL": {"min": -4.0, "max": 4.0},
        }
        super().__init__(
            model_name=model_name,
            model_label=model_label or "Parity-even SIGW",
            parameter_labels=parameter_labels or labels,
            prior_by_param=prior_by_param or priors,
        )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        target_f_peak: jax.Array,
        target_n2: jax.Array,
        log10_A_zeta: jax.Array,
        log10_f_NL: jax.Array,
        log10_tau_NL: jax.Array,
        log10_tilde_tau_NL: jax.Array,
    ) -> jax.Array:
        components = evaluate_components_jax(
            jnp.asarray(frequency, dtype=jnp.float64),
            target_f_peak,
            target_n2,
            log10_A_zeta,
            log10_f_NL,
            log10_tau_NL,
            log10_tilde_tau_NL,
        )
        return jnp.sum(components[:3], axis=0)

    def _grad_theta_omega_gw_h2_analytical(
        self, frequency: jax.Array, theta: jax.Array, *args: Any, **kwargs: Any
    ) -> jax.Array:
        autodiff_gradient = jax.jacfwd(self._omega_from_parameter_vector, argnums=1)(
            frequency, theta, *args, **kwargs
        )
        shape_components = interpolate_components_jax(
            jnp.asarray(frequency, dtype=jnp.float64),
            theta[0],
            theta[1],
        )
        normalization_gradient = normalization_factor_gradients_jax(
            theta[2], theta[3], theta[4], theta[5]
        )
        scale_gradient = jnp.einsum(
            "cf,pc->fp", shape_components[:3], normalization_gradient[:, :3]
        )
        return autodiff_gradient.at[..., 2:].set(scale_gradient)
