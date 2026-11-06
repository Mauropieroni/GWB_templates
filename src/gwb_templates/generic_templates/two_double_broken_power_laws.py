r"""
Sum of two double broken power-law spectra (16 parameters).

Models scenarios where two independent FOPT sources contribute to the GWB simultaneously
— for example, sound waves and bubble collisions from the same transition, or two
independent transitions at different temperatures. The second spectrum parameters are
expressed as ratios relative to the first spectrum's parameters to reduce prior volume.

Reference: arXiv:2403.03723.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.double_broken_power_law import (
    DoubleBrokenPowerLaw,
)
from gwb_templates.template import AnalyticTemplate


class TwoDoubleBrokenPowerLaws(AnalyticTemplate):
    r"""
    Sum of two double broken power laws (16 parameters).

    The second amplitude and break frequencies are expressed as ratios relative to the
    first spectrum's parameters.

    The first DBPL uses (with :math:`\log f_{11} = \log f_{12} - \log r_{f,12}`):
    ``[log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12]``.

    The second DBPL uses (with
    :math:`\log A_2 = \log A_1 + \log r_{A,2}`,
    :math:`\log f_{21} = \log f_{11} + \log r_{f,21}`,
    :math:`\log f_{22} = \log f_{11} + \log r_{f,22}`):
    ``[log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22]``.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2024hue,
    author = "Caprini, Chiara and Jinno, Ryusuke and Lewicki, Marek and Madge, Eric and
        Merchand, Marco and Nardini, Germano and Pieroni, Mauro and Roper Pol, Alberto
        and Vaskonen, Ville",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from first-order phase transitions in LISA:
        reconstruction pipeline and physics interpretation}",
    eprint = "2403.03723",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-01, CERN-TH-2024-029",
    doi = "10.1088/1475-7516/2024/10/020",
    journal = "JCAP",
    volume = "10",
    pages = "020",
    year = "2024"
}
""",
    )

    DEFAULT_MODEL_NAME: ClassVar[str] = "two_double_broken_power_laws"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Two Double Broken Power Laws"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
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
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
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

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Caprini:2024hue,
    author = "Caprini, Chiara and Jinno, Ryusuke and Lewicki, Marek and Madge, Eric and
        Merchand, Marco and Nardini, Germano and Pieroni, Mauro and Roper Pol, Alberto
        and Vaskonen, Ville",
    collaboration = "LISA Cosmology Working Group",
    title = "{Gravitational waves from first-order phase transitions in LISA:
        reconstruction pipeline and physics interpretation}",
    eprint = "2403.03723",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "LISA-COSWG-24-01, CERN-TH-2024-029",
    doi = "10.1088/1475-7516/2024/10/020",
    journal = "JCAP",
    volume = "10",
    pages = "020",
    year = "2024"
}
""",
    )

    def __init__(self, **kwargs: Any) -> None:
        # Underlying single-DBPL helper used inside omega_gw_h2.
        self._dbpl = DoubleBrokenPowerLaw()
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amp_1: jax.Array,
        log_r_amp_2: jax.Array,
        log_f_12: jax.Array,
        log_r_f_12: jax.Array,
        log_r_f_21: jax.Array,
        log_r_f_22: jax.Array,
        n_11: jax.Array,
        n_12: jax.Array,
        n_13: jax.Array,
        a_11: jax.Array,
        a_12: jax.Array,
        n_21: jax.Array,
        n_22: jax.Array,
        n_23: jax.Array,
        a_21: jax.Array,
        a_22: jax.Array,
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
        return dbpl_1 + dbpl_2

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian via the chain rule from :class:`DoubleBrokenPowerLaw`.

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
