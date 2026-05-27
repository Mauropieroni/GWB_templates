"""
Sum of two double broken power-law spectra (16 parameters).

Models scenarios where two independent FOPT sources contribute to the GWB
simultaneously — for example, sound waves and bubble collisions from the
same transition, or two independent transitions at different temperatures.
The second spectrum parameters are expressed as ratios relative to the
first spectrum's parameters to reduce prior volume.

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline
and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.generic_templates.double_broken_power_law import (
    double_broken_power_law,
    d1double_broken_power_law,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def two_double_broken_power_laws(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Sum of two double broken power laws (16 parameters).

    The second amplitude and break frequencies are expressed as ratios
    relative to the first spectrum's parameters.

    The first DBPL uses:
        log_f_11 = log_f_12 - log_r_f_12
        [log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12]

    The second DBPL uses:
        log_amp_2 = log_amp_1 + log_r_amp_2
        log_f_21 = log_f_11 + log_r_f_21
        log_f_22 = log_f_11 + log_r_f_22
        [log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22]

    Args:
        freq: Frequency grid.
        pars: [log_amp_1, log_r_amp_2,
               log_f_12, log_r_f_12, log_r_f_21, log_r_f_22,
               n_11, n_12, n_13, a_11, a_12,
               n_21, n_22, n_23, a_21, a_22].
    Returns:
        jax.Array of shape (N_freq,).
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
    ) = pars[:16]

    log_amp_2 = log_amp_1 + log_r_amp_2
    log_f_11 = log_f_12 - log_r_f_12
    log_f_21 = log_f_11 + log_r_f_21
    log_f_22 = log_f_11 + log_r_f_22

    dbpl_1 = double_broken_power_law(
        freq,
        jnp.array([log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12]),
    )
    dbpl_2 = double_broken_power_law(
        freq,
        jnp.array([log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22]),
    )
    return dbpl_1 + dbpl_2


def d1two_double_broken_power_laws(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``two_double_broken_power_laws`` w.r.t. pars.

    Args:
        freq: Frequency grid.
        pars: [log_amp_1, log_r_amp_2,
               log_f_12, log_r_f_12, log_r_f_21, log_r_f_22,
               n_11, n_12, n_13, a_11, a_12,
               n_21, n_22, n_23, a_21, a_22].
    Returns:
        jax.Array of shape (N_freq, 16).
    """
    pars = jnp.asarray(pars)
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
    ) = pars[:16]

    log_amp_2 = log_amp_1 + log_r_amp_2
    log_f_11 = log_f_12 - log_r_f_12
    log_f_21 = log_f_11 + log_r_f_21
    log_f_22 = log_f_11 + log_r_f_22

    pars1 = jnp.array([log_amp_1, log_f_11, log_f_12, n_11, n_12, n_13, a_11, a_12])
    pars2 = jnp.array([log_amp_2, log_f_21, log_f_22, n_21, n_22, n_23, a_21, a_22])

    J1 = d1double_broken_power_law(
        freq, pars1
    )  # (N, 8): [logA, logf1, logf2, n1..n3, a1, a2]
    J2 = d1double_broken_power_law(freq, pars2)  # (N, 8)

    # Chain rule for the 16 parameters of two_double_broken_power_laws:
    # log_amp_1  -> pars1[0] (direct) and pars2[0] (via log_amp_2 = log_amp_1 + r)
    d_logamp1 = J1[:, 0] + J2[:, 0]
    # log_r_amp_2 -> pars2[0] only
    d_logr_amp2 = J2[:, 0]
    # log_f_12 -> pars1[1] (via log_f_11=log_f_12-r12), pars1[2] (direct),
    # pars2[1] (via log_f_21=log_f_11+r21), pars2[2] (via log_f_22=log_f_11+r22)
    d_logf12 = J1[:, 1] + J1[:, 2] + J2[:, 1] + J2[:, 2]
    # log_r_f_12 -> pars1[1] (-1), pars2[1] (-1), pars2[2] (-1)
    d_logrf12 = -J1[:, 1] - J2[:, 1] - J2[:, 2]
    # log_r_f_21 -> pars2[1] only
    d_logrf21 = J2[:, 1]
    # log_r_f_22 -> pars2[2] only
    d_logrf22 = J2[:, 2]

    return jnp.stack(
        [
            d_logamp1,
            d_logr_amp2,
            d_logf12,
            d_logrf12,
            d_logrf21,
            d_logrf22,
            J1[:, 3],
            J1[:, 4],
            J1[:, 5],
            J1[:, 6],
            J1[:, 7],
            J2[:, 3],
            J2[:, 4],
            J2[:, 5],
            J2[:, 6],
            J2[:, 7],
        ],
        axis=1,
    )


two_double_broken_power_laws_model = ut.Signal_model(
    "two_double_broken_power_laws",
    two_double_broken_power_laws,
    dtemplate=d1two_double_broken_power_laws,
    model_label="Two Double Broken Power Laws",
    parameter_names=[
        "log_amp_1",
        "log_r_amp_2",
        "log_f_12",
        "log_r_f_12",
        "log_r_f_21",
        "log_r_f_22",
        "n_11",
        "n_12",
        "n_13",
        "a_11",
        "a_12",
        "n_21",
        "n_22",
        "n_23",
        "a_21",
        "a_22",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_{*,1})$",
        r"$\log_{10}(\Omega_{*,2}/\Omega_{*,1})$",
        r"$\log_{10}(f_{12}/\mathrm{Hz})$",
        r"$\log_{10}(f_{12}/f_{11})$",
        r"$\log_{10}(f_{21}/f_{11})$",
        r"$\log_{10}(f_{22}/f_{11})$",
        r"$n_{11}$",
        r"$n_{12}$",
        r"$n_{13}$",
        r"$a_{11}$",
        r"$a_{12}$",
        r"$n_{21}$",
        r"$n_{22}$",
        r"$n_{23}$",
        r"$a_{21}$",
        r"$a_{22}$",
    ],
    prior={
        "log_amp_1": {"prior_type": "uniform", "minimum": -20.0, "maximum": -1.0},
        "log_r_amp_2": {"prior_type": "uniform", "minimum": -5.0, "maximum": 5.0},
        "log_f_12": {"prior_type": "uniform", "minimum": -10.0, "maximum": 0.0},
        "log_r_f_12": {"prior_type": "uniform", "minimum": -3.0, "maximum": 3.0},
        "log_r_f_21": {"prior_type": "uniform", "minimum": -3.0, "maximum": 3.0},
        "log_r_f_22": {"prior_type": "uniform", "minimum": -3.0, "maximum": 3.0},
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
    },
)
