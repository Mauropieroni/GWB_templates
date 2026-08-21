"""
GWB template library.

Importing this package eagerly imports every template module, which
triggers ``Template.__init_subclass__`` for each concrete template and
populates :attr:`Template._registry`. After ``import gwb_templates``
you can look up any template by class name via
:func:`get_template_from_registry`.
"""

from gwb_templates.template import (
    AnalyticTemplate,
    NumericalTemplate,
    Template,
    get_template_from_registry,
)

# ── Generic / geometric ───────────────────────────────────────────────────────
from gwb_templates.generic_templates import amplitude  # noqa: F401
from gwb_templates.generic_templates import broken_power_law  # noqa: F401
from gwb_templates.generic_templates import broken_power_law_a1  # noqa: F401
from gwb_templates.generic_templates import (  # noqa: F401
    broken_power_law_fixed_smoothness,
)
from gwb_templates.generic_templates import double_broken_power_law  # noqa: F401
from gwb_templates.generic_templates import double_broken_power_law_rf  # noqa: F401
from gwb_templates.generic_templates import lognormal_bump  # noqa: F401
from gwb_templates.generic_templates import power_law  # noqa: F401
from gwb_templates.generic_templates import two_double_broken_power_laws  # noqa: F401

# ── First-order phase transitions ─────────────────────────────────────────────
from gwb_templates.FOPT_templates import fopt_broken_power_law  # noqa: F401
from gwb_templates.FOPT_templates import fopt_broken_power_law_old  # noqa: F401
from gwb_templates.FOPT_templates import fopt_old_model  # noqa: F401
from gwb_templates.FOPT_templates import pt_collision  # noqa: F401
from gwb_templates.FOPT_templates import pt_plasma  # noqa: F401
from gwb_templates.FOPT_templates import pt_sound_waves  # noqa: F401
from gwb_templates.FOPT_templates import pt_turbulence  # noqa: F401

# ── Inflation ─────────────────────────────────────────────────────────────────
from gwb_templates.inflation_templates import axion_inflation  # noqa: F401
from gwb_templates.inflation_templates import double_peak  # noqa: F401
from gwb_templates.inflation_templates import double_peak_sharp  # noqa: F401
from gwb_templates.inflation_templates import excited_states  # noqa: F401
from gwb_templates.inflation_templates import flat_resonant  # noqa: F401
from gwb_templates.inflation_templates import lognormal_bump_sharp  # noqa: F401
from gwb_templates.inflation_templates import resonant_feature  # noqa: F401
from gwb_templates.inflation_templates import sharp_feature  # noqa: F401

# ── Astrophysical foregrounds ─────────────────────────────────────────────────
from gwb_templates.astrophysical_templates import extragalactic_sobbh_bns  # noqa: F401
from gwb_templates.astrophysical_templates import extragalactic_wd  # noqa: F401
from gwb_templates.astrophysical_templates import extragalactic_wd2  # noqa: F401
from gwb_templates.astrophysical_templates import galactic_binaries  # noqa: F401
from gwb_templates.astrophysical_templates import galactic_binaries_old  # noqa: F401

# ── Cosmic strings ────────────────────────────────────────────────────────────
from gwb_templates.cosmic_string_templates import cosmic_string_model_i  # noqa: F401
from gwb_templates.cosmic_string_templates import (  # noqa: F401
    cosmic_string_model_i_edf,
    cosmic_string_model_i_eos,
)
from gwb_templates.cosmic_string_templates import cosmic_string_model_ii  # noqa: F401

# ── Scalar-induced GWs ────────────────────────────────────────────────────────
# The physics-category marker base is dependency-free and always available.
from gwb_templates.scalar_induced_templates.base import (  # noqa: F401
    ScalarInducedTemplate,
)

# SIGWAY backend (optional extra: ``pip install gwb_templates[sigway]``; needs a
# JAX new enough for ``jax.scipy.special.sici``). Import the SIGWAY templates
# only if the dependency actually imports; otherwise ``import gwb_templates``
# still works and the SIGWAY templates simply do not register.
# ``SIGWAY_AVAILABLE`` is set True only once every SIGWAY module has imported
# successfully — a bare ``import sigway`` is not enough, since ``sigway``'s
# top-level package imports even when its numerical submodules cannot.
SIGWAY_AVAILABLE: bool = False
try:
    from gwb_templates.scalar_induced_templates.sigway.base import (  # noqa: F401
        SIGWAYTemplate,
        sigway_template,
    )
    from gwb_templates.scalar_induced_templates.sigway import (  # noqa: F401
        broken_power_law as sigway_broken_power_law,
        early_matter_domination as sigway_early_matter_domination,
        lognormal as sigway_lognormal,
        multifield_oscillations as sigway_multifield_oscillations,
        single_field_usr as sigway_single_field_usr,
    )

    SIGWAY_AVAILABLE = True
except ImportError:
    # SIGWAY (or a compatible JAX) is not installed: skip the SIGWAY templates.
    pass

__all__ = [
    "Template",
    "AnalyticTemplate",
    "NumericalTemplate",
    "get_template_from_registry",
    "ScalarInducedTemplate",
    "SIGWAY_AVAILABLE",
]

if SIGWAY_AVAILABLE:
    __all__ += ["SIGWAYTemplate", "sigway_template"]
