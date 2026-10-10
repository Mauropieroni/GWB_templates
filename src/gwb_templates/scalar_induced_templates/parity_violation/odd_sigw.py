"""Parity-odd scalar-induced gravitational-wave template."""

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
    normalization_factors_jax,
    normalization_factor_gradients_jax,
)


class OddSIGW(ScalarInducedTemplate, NumericalTemplate):
    r"""
    Parity-violating SIGW arising from odd trispectrum (arxiv:2507.02733,2607.16162).
    Primordial power spectrum set as broken power law, with IR index equal to 4,
    steepest possible growth in canonical single-field inflation (arxiv:1811.11158).
    Partially jaxed interpolation of precomputed grid through helper functions.

    Free parameters
    ---------------
    target_f_peak
        Peak frequency in Hz.
    target_n2
        UV spectral index.
    log10_A_zeta
        :math:`\log_{10}` amplitude of the primordial scalar spectrum.
    log10_tilde_tau_NL
        :math:`\log_{10}` of the odd trispectrum coefficient
        :math:`\tilde\tau_{NL}`.
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
 """,
        r"""
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
            "log10_tilde_tau_NL": r"$\log_{10}\tilde{\tau}_{\rm NL}$",
        }
        priors = {
            "target_f_peak": {"min": 1e-6, "max": 1e4},
            "target_n2": {"min": 0.1, "max": 1.0},
            "log10_A_zeta": {"min": -4.0, "max": 0.0},
            "log10_tilde_tau_NL": {"min": -4.0, "max": 4.0},
        }
        super().__init__(
            model_name=model_name,
            model_label=(model_label if model_label is not None else "Parity-odd SIGW"),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else labels
            ),
            prior_by_param=(prior_by_param if prior_by_param is not None else priors),
        )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        target_f_peak: jax.Array,
        target_n2: jax.Array,
        log10_A_zeta: jax.Array,
        log10_tilde_tau_NL: jax.Array,
    ) -> jax.Array:
        components = evaluate_components_jax(
            jnp.asarray(frequency, dtype=jnp.float64),
            target_f_peak,
            target_n2,
            log10_A_zeta,
            0.0,
            0.0,
            log10_tilde_tau_NL,
        )
        return jnp.abs(components[3])  # to have positive energy density

    def _grad_theta_omega_gw_h2_analytical(
        self, frequency: jax.Array, theta: jax.Array, *args: Any, **kwargs: Any
    ) -> jax.Array:
        target_frequencies = jnp.asarray(frequency, dtype=jnp.float64)
        normalization_factors = normalization_factors_jax(theta[2], 0.0, 0.0, theta[3])

        def shape_only(shape_parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
            components = interpolate_components_jax(
                target_frequencies,
                shape_parameters[0],
                shape_parameters[1],
            )
            return jnp.abs(components[3] * normalization_factors[3]), components

        shape_gradient, shape_components = jax.jacfwd(shape_only, has_aux=True)(
            theta[:2]
        )
        normalization_gradient = normalization_factor_gradients_jax(
            theta[2], 0.0, 0.0, theta[3]
        )
        scale_gradient = jnp.stack(
            [
                jnp.abs(shape_components[3]) * normalization_gradient[0, 3],
                jnp.abs(shape_components[3]) * normalization_gradient[3, 3],
            ],
            axis=-1,
        )
        return jnp.concatenate((shape_gradient, scale_gradient), axis=-1)
