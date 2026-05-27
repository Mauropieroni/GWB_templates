"""
Double broken power-law with frequency-ratio reparametrisation (8 parameters).

Variant of ``double_broken_power_law`` where ``log_f_1`` is replaced by
``log_r_f = log10(f_2 / f_1)``.  This reparametrisation decouples the overall
frequency scale from the internal frequency ratio, which can improve sampling
efficiency when the ratio is well-constrained by the physics.

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline
and physics interpretation).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates import utils as ut
from gwb_templates.FOPT_templates.double_broken_power_law import (
    double_broken_power_law,
    d1double_broken_power_law,
)

ParamLike = jax.Array | Sequence[float]
ArrayLike = jtp.ArrayLike

jax.config.update("jax_enable_x64", True)


def double_broken_power_law_rf(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Double broken power law reparameterised with a frequency ratio (8 parameters).

    Identical to ``double_broken_power_law`` but replaces ``log_f_1`` with
    ``log_r_f = log10(f_2 / f_1)``, so that ``log_f_1 = log_f_2 - log_r_f``.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude at f_2, log10 f_2, log10(f_2/f_1),
               n_1, n_2, n_3, a_1, a_2].
    Returns:
        jax.Array of shape (N_freq,).
    """
    log_amplitude, log_f_2, log_r_f, n_1, n_2, n_3, a_1, a_2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
        pars[7],
    )
    log_f_1 = log_f_2 - log_r_f
    return double_broken_power_law(
        freq,
        jnp.array([log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2]),
    )


def d1double_broken_power_law_rf(freq: ArrayLike, pars: ParamLike) -> jax.Array:
    """
    Analytical Jacobian of ``double_broken_power_law_rf`` w.r.t. pars.

    Since log_f_1 = log_f_2 - log_r_f, the chain rule gives:
        d/d(log_f_2)  = DBPL[1] + DBPL[2]   (∂f_1/∂f_2=1, ∂f_2/∂f_2=1)
        d/d(log_r_f)  = -DBPL[1]             (∂f_1/∂r_f=-1)
    All other columns are taken directly from d1double_broken_power_law.

    Args:
        freq: Frequency grid.
        pars: [log10 amplitude at f_2, log10 f_2, log10(f_2/f_1),
               n_1, n_2, n_3, a_1, a_2].
    Returns:
        jax.Array of shape (N_freq, 8).
    """
    log_amplitude, log_f_2, log_r_f, n_1, n_2, n_3, a_1, a_2 = (
        pars[0],
        pars[1],
        pars[2],
        pars[3],
        pars[4],
        pars[5],
        pars[6],
        pars[7],
    )
    log_f_1 = log_f_2 - log_r_f
    pars_dbpl = jnp.array([log_amplitude, log_f_1, log_f_2, n_1, n_2, n_3, a_1, a_2])
    J = d1double_broken_power_law(freq, pars_dbpl)

    # J columns: [logA, logf1, logf2, n1, n2, n3, a1, a2]
    d_logA = J[:, 0]
    d_logf2 = J[:, 1] + J[:, 2]  # chain rule
    d_logrf = -J[:, 1]  # chain rule
    d_n1 = J[:, 3]
    d_n2 = J[:, 4]
    d_n3 = J[:, 5]
    d_a1 = J[:, 6]
    d_a2 = J[:, 7]

    return jnp.stack([d_logA, d_logf2, d_logrf, d_n1, d_n2, d_n3, d_a1, d_a2], axis=1)


double_broken_power_law_rf_model = ut.Signal_model(
    "double_broken_power_law_rf",
    double_broken_power_law_rf,
    dtemplate=d1double_broken_power_law_rf,
    model_label="Double Broken Power Law (ratio freq.)",
    parameter_names=[
        "log_amplitude",
        "log_f_2",
        "log_r_f",
        "n_1",
        "n_2",
        "n_3",
        "a_1",
        "a_2",
    ],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_2/\mathrm{Hz})$",
        r"$\log_{10}(f_2/f_1)$",
        r"$n_1$",
        r"$n_2$",
        r"$n_3$",
        r"$a_1$",
        r"$a_2$",
    ],
    prior={
        "log_amplitude": {"prior_type": "uniform", "minimum": -20.0, "maximum": -1.0},
        "log_f_2": {"prior_type": "uniform", "minimum": -10.0, "maximum": 0.0},
        "log_r_f": {"prior_type": "uniform", "minimum": -3.0, "maximum": 3.0},
        "n_1": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "n_2": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "n_3": {"prior_type": "uniform", "minimum": -7.0, "maximum": 7.0},
        "a_1": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
        "a_2": {"prior_type": "uniform", "minimum": 0.1, "maximum": 10.0},
    },
)
