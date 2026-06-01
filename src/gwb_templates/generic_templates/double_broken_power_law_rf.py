r"""
Double broken power-law with frequency-ratio reparametrisation (8 parameters).

Variant of :class:`DoubleBrokenPowerLaw` where ``log_f_1`` is replaced by
:math:`\log_{10}(f_2/f_1)`. This reparametrisation decouples the overall
frequency scale from the internal frequency ratio, which can improve
sampling efficiency when the ratio is well-constrained by the physics.

Reference: arXiv:2403.03723.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.generic_templates.double_broken_power_law import (
    DoubleBrokenPowerLaw,
)
from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class DoubleBrokenPowerLawRf(AnalyticTemplate):
    r"""
    Double broken power law reparameterised with a frequency ratio.

    Identical to :class:`DoubleBrokenPowerLaw` but replaces ``log_f_1``
    with :math:`\log r_f = \log_{10}(f_2 / f_1)`, so that
    :math:`\log f_1 = \log f_2 - \log r_f`.

    Free parameters
    ---------------
    log_amplitude
        :math:`\log_{10}` amplitude at :math:`f_2`.
    log_f_2
        :math:`\log_{10}` of the second break frequency in Hz.
    log_r_f
        :math:`\log_{10}(f_2/f_1)`.
    n_1, n_2, n_3
        Spectral indices in the three regimes.
    a_1, a_2
        Smoothness parameters at the two breaks.
    """

    #: TODO: cite
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        # Underlying single-DBPL helper used to delegate the analytic
        # gradient via the chain rule.
        self._dbpl = DoubleBrokenPowerLaw()

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_*)$",
            "log_f_2": r"$\log_{10}(f_2/\mathrm{Hz})$",
            "log_r_f": r"$\log_{10}(f_2/f_1)$",
            "n_1": r"$n_1$",
            "n_2": r"$n_2$",
            "n_3": r"$n_3$",
            "a_1": r"$a_1$",
            "a_2": r"$a_2$",
        }
        default_priors = {
            "log_amplitude": {
                "prior_type": "uniform",
                "minimum": -20.0,
                "maximum": -1.0,
            },
            "log_f_2": {
                "prior_type": "uniform",
                "minimum": -10.0,
                "maximum": 0.0,
            },
            "log_r_f": {
                "prior_type": "uniform",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_3": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
            "a_2": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Double Broken Power Law (ratio freq.)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        log_amplitude: ArrayLike,
        log_f_2: ArrayLike,
        log_r_f: ArrayLike,
        n_1: ArrayLike,
        n_2: ArrayLike,
        n_3: ArrayLike,
        a_1: ArrayLike,
        a_2: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the frequency-ratio reparametrised DBPL at ``frequency``.
        """
        log_f_1 = log_f_2 - log_r_f
        amplitude = 10.0**log_amplitude
        ratio = 10.0 ** (log_f_1 - log_f_2)  # f_1 / f_2
        x_1 = frequency / 10.0**log_f_1
        x_2 = frequency / 10.0**log_f_2

        norm = (
            ratio**n_1
            * (1.0 + ratio ** (-a_1)) ** ((n_1 - n_2) / a_1)
            * 2.0 ** ((n_2 - n_3) / a_2)
        )

        return (
            norm
            * amplitude
            * x_1**n_1
            * (1.0 + x_1**a_1) ** ((n_2 - n_1) / a_1)
            * (1.0 + x_2**a_2) ** ((n_3 - n_2) / a_2)
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian via the chain rule from
        :class:`DoubleBrokenPowerLaw`.

        Since :math:`\log f_1 = \log f_2 - \log r_f`,

        - :math:`\partial/\partial(\log f_2) = J_{\log f_1} + J_{\log f_2}`
        - :math:`\partial/\partial(\log r_f) = -J_{\log f_1}`

        All other columns pass through unchanged.
        """
        log_amplitude, log_f_2, log_r_f, n_1, n_2, n_3, a_1, a_2 = theta
        log_f_1 = log_f_2 - log_r_f
        theta_dbpl = jnp.stack(
            [log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2]
        )
        # J columns: [logA, logf1, logf2, n1, n2, n3, a1, a2]
        J = self._dbpl._grad_theta_omega_gw_h2_analytical(frequency, theta_dbpl)

        d_logA = J[..., 0]
        d_logf2 = J[..., 1] + J[..., 2]  # chain rule
        d_logrf = -J[..., 1]  # chain rule
        d_n1 = J[..., 3]
        d_n2 = J[..., 4]
        d_n3 = J[..., 5]
        d_a1 = J[..., 6]
        d_a2 = J[..., 7]

        return jnp.stack(
            [d_logA, d_logf2, d_logrf, d_n1, d_n2, d_n3, d_a1, d_a2], axis=-1
        )
