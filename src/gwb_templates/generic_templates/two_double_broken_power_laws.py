r"""
Sum of two double broken power-law spectra (16 parameters).

Models scenarios where two independent FOPT sources contribute to the GWB
simultaneously — for example, sound waves and bubble collisions from the
same transition, or two independent transitions at different temperatures.
The second spectrum parameters are expressed as ratios relative to the
first spectrum's parameters to reduce prior volume.

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


class TwoDoubleBrokenPowerLaws(AnalyticTemplate):
    r"""
    Sum of two double broken power laws (16 parameters).

    The second amplitude and break frequencies are expressed as ratios
    relative to the first spectrum's parameters.

    The first DBPL uses (with :math:`\log f_{11} = \log f_{12} - \log r_{f,12}`):
    ``[log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12]``.

    The second DBPL uses (with
    :math:`\log A_2 = \log A_1 + \log r_{A,2}`,
    :math:`\log f_{21} = \log f_{11} + \log r_{f,21}`,
    :math:`\log f_{22} = \log f_{11} + \log r_{f,22}`):
    ``[log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22]``.
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
        # Underlying single-DBPL helper used inside omega_gw_h2.
        self._dbpl = DoubleBrokenPowerLaw()

        default_labels = {
            "log_amp_1": r"$\log_{10}(h^2\,\Omega_{*,1})$",
            "log_r_amp_2": r"$\log_{10}(\Omega_{*,2}/\Omega_{*,1})$",
            "log_f_12": r"$\log_{10}(f_{12}/\mathrm{Hz})$",
            "log_r_f_12": r"$\log_{10}(f_{12}/f_{11})$",
            "log_r_f_21": r"$\log_{10}(f_{21}/f_{11})$",
            "log_r_f_22": r"$\log_{10}(f_{22}/f_{11})$",
            "n_11": r"$n_{11}$",
            "n_12": r"$n_{12}$",
            "n_13": r"$n_{13}$",
            "a_11": r"$a_{11}$",
            "a_12": r"$a_{12}$",
            "n_21": r"$n_{21}$",
            "n_22": r"$n_{22}$",
            "n_23": r"$n_{23}$",
            "a_21": r"$a_{21}$",
            "a_22": r"$a_{22}$",
        }
        default_priors = {
            "log_amp_1": {
                "prior_type": "uniform",
                "minimum": -20.0,
                "maximum": -1.0,
            },
            "log_r_amp_2": {
                "prior_type": "uniform",
                "minimum": -5.0,
                "maximum": 5.0,
            },
            "log_f_12": {
                "prior_type": "uniform",
                "minimum": -10.0,
                "maximum": 0.0,
            },
            "log_r_f_12": {
                "prior_type": "uniform",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "log_r_f_21": {
                "prior_type": "uniform",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "log_r_f_22": {
                "prior_type": "uniform",
                "minimum": -3.0,
                "maximum": 3.0,
            },
            "n_11": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_12": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_13": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "a_11": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
            "a_12": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
            "n_21": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_22": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "n_23": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
            "a_21": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
            "a_22": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Two Double Broken Power Laws"
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
        log_amp_1: ArrayLike,
        log_r_amp_2: ArrayLike,
        log_f_12: ArrayLike,
        log_r_f_12: ArrayLike,
        log_r_f_21: ArrayLike,
        log_r_f_22: ArrayLike,
        n_11: ArrayLike,
        n_12: ArrayLike,
        n_13: ArrayLike,
        a_11: ArrayLike,
        a_12: ArrayLike,
        n_21: ArrayLike,
        n_22: ArrayLike,
        n_23: ArrayLike,
        a_21: ArrayLike,
        a_22: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the sum of two DBPL spectra at ``frequency``.
        """
        log_amp_2 = log_amp_1 + log_r_amp_2
        log_f_11 = log_f_12 - log_r_f_12
        log_f_21 = log_f_11 + log_r_f_21
        log_f_22 = log_f_11 + log_r_f_22

        dbpl_1 = self._dbpl.omega_gw_h2(
            frequency,
            log_amp_1,
            log_f_11,
            log_f_12,
            n_11,
            n_12,
            n_13,
            a_11,
            a_12,
        )
        dbpl_2 = self._dbpl.omega_gw_h2(
            frequency,
            log_amp_2,
            log_f_21,
            log_f_22,
            n_21,
            n_22,
            n_23,
            a_21,
            a_22,
        )
        return jnp.asarray(dbpl_1 + dbpl_2)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian via the chain rule from
        :class:`DoubleBrokenPowerLaw`.

        Reparametrisation:
        :math:`\log A_2 = \log A_1 + \log r_{A,2}`,
        :math:`\log f_{11} = \log f_{12} - \log r_{f,12}`,
        :math:`\log f_{21} = \log f_{11} + \log r_{f,21}`,
        :math:`\log f_{22} = \log f_{11} + \log r_{f,22}`.
        """
        (
            log_amp_1,
            log_r_amp_2,
            log_f_12,
            log_r_f_12,
            log_r_f_21,
            log_r_f_22,
            n_11,
            n_12,
            n_13,
            a_11,
            a_12,
            n_21,
            n_22,
            n_23,
            a_21,
            a_22,
        ) = theta

        log_amp_2 = log_amp_1 + log_r_amp_2
        log_f_11 = log_f_12 - log_r_f_12
        log_f_21 = log_f_11 + log_r_f_21
        log_f_22 = log_f_11 + log_r_f_22

        theta1 = jnp.stack(
            [log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12]
        )
        theta2 = jnp.stack(
            [log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22]
        )

        # (..., 8) columns: [logA, logf1, logf2, n1, n2, n3, a1, a2]
        J1 = self._dbpl._grad_theta_omega_gw_h2_analytical(frequency, theta1)
        J2 = self._dbpl._grad_theta_omega_gw_h2_analytical(frequency, theta2)

        d_logamp1 = J1[..., 0] + J2[..., 0]
        d_logr_amp2 = J2[..., 0]
        d_logf12 = J1[..., 1] + J1[..., 2] + J2[..., 1] + J2[..., 2]
        d_logrf12 = -J1[..., 1] - J2[..., 1] - J2[..., 2]
        d_logrf21 = J2[..., 1]
        d_logrf22 = J2[..., 2]

        return jnp.stack(
            [
                d_logamp1,
                d_logr_amp2,
                d_logf12,
                d_logrf12,
                d_logrf21,
                d_logrf22,
                J1[..., 3],
                J1[..., 4],
                J1[..., 5],
                J1[..., 6],
                J1[..., 7],
                J2[..., 3],
                J2[..., 4],
                J2[..., 5],
                J2[..., 6],
                J2[..., 7],
            ],
            axis=-1,
        )
