"""
FOPT broken power law template (4 parameters).

Inherits the broken_power_law_fixed_smoothness geometry from generic_templates
and relabels parameters for a first-order phase-transition context.

Reference: arXiv:2403.03723 (GW from FOPT in LISA: reconstruction pipeline
and physics interpretation).
"""

from gwb_templates.generic_templates.broken_power_law_fixed_smoothness import (
    broken_power_law_fixed_smoothness,
    d1broken_power_law_fixed_smoothness,
)
from gwb_templates import utils as ut

fopt_broken_power_law_model = ut.Signal_model(
    "fopt_broken_power_law",
    broken_power_law_fixed_smoothness,
    dtemplate=d1broken_power_law_fixed_smoothness,
    model_label="FOPT Broken Power Law",
    parameter_names=["log_amplitude", "log_f_star", "n_IR", "n_UV"],
    parameter_labels=[
        r"$\log_{10}(h^2\,\Omega_*)$",
        r"$\log_{10}(f_*/\mathrm{Hz})$",
        r"$n_{\mathrm{IR}}$",
        r"$n_{\mathrm{UV}}$",
    ],
    prior={
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "log_f_star": {"min": -5.0, "max": 0.0},
        "n_IR": {"min": 0.0, "max": 10.0},
        "n_UV": {"min": -10.0, "max": 0.0},
    },
)
