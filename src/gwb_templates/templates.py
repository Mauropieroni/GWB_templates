from gwb_templates.utils import Signal_model

# ── Generic / geometric ───────────────────────────────────────────────────────
from gwb_templates.generic_templates.amplitude import amplitude_model
from gwb_templates.generic_templates.power_law import power_law_model
from gwb_templates.generic_templates.lognormal_bump import lognormal_bump_model
from gwb_templates.generic_templates.broken_power_law import (
    broken_power_law_model,
)
from gwb_templates.generic_templates.broken_power_law_fixed_smoothness import (
    broken_power_law_fixed_smoothness_model,
)

# ── First-order phase transitions ─────────────────────────────────────────────
from gwb_templates.FOPT_templates.fopt_broken_power_law import (
    fopt_broken_power_law_model,
)
from gwb_templates.FOPT_templates.fopt_broken_power_law_old import (
    fopt_broken_power_law_old_model,
)
from gwb_templates.FOPT_templates.broken_power_law_a1 import (
    broken_power_law_a1_model,
)
from gwb_templates.FOPT_templates.double_broken_power_law import (
    double_broken_power_law_model,
)
from gwb_templates.FOPT_templates.double_broken_power_law_rf import (
    double_broken_power_law_rf_model,
)
from gwb_templates.generic_templates.two_double_broken_power_laws import (
    two_double_broken_power_laws_model,
)
from gwb_templates.FOPT_templates.pt_sound_waves import pt_sound_waves_model
from gwb_templates.FOPT_templates.pt_turbulence import pt_turbulence_model
from gwb_templates.FOPT_templates.pt_collision import pt_collision_model
from gwb_templates.FOPT_templates.pt_plasma import pt_plasma_model

# ── Inflation ─────────────────────────────────────────────────────────────────
from gwb_templates.inflation_templates.double_peak import double_peak_model
from gwb_templates.inflation_templates.double_peak_sharp import (
    double_peak_sharp_model,
    double_peak_sharp_log_model,
)
from gwb_templates.inflation_templates.excited_states import excited_states_model
from gwb_templates.inflation_templates.flat_resonant import (
    flat_resonant_model,
    flat_resonant_log_model,
)
from gwb_templates.inflation_templates.lognormal_bump_sharp import (
    lognormal_bump_sharp_model,
    lognormal_bump_sharp_log_model,
)
from gwb_templates.inflation_templates.sharp_feature import (
    sharp_feature_model,
    sharp_feature_log_model,
)
from gwb_templates.inflation_templates.resonant_feature import (
    resonant_feature_model,
    resonant_feature_log_model,
)

# ── Astrophysical foregrounds ─────────────────────────────────────────────────
from gwb_templates.astrophysical_templates.galactic_binaries import (
    galactic_binaries_model,
    galactic_binaries_A_model,
)
from gwb_templates.astrophysical_templates.galactic_binaries_old import (
    galactic_binaries_old_model,
    galactic_binaries_old_A_model,
)
from gwb_templates.astrophysical_templates.extragalactic_sobbh_bns import (
    extragalactic_sobbh_bns_model,
    extragalactic_sobbh_bns_A_model,
)
from gwb_templates.astrophysical_templates.extragalactic_wd import (
    extragalactic_wd_model,
    extragalactic_wd_A_model,
)
from gwb_templates.astrophysical_templates.extragalactic_wd2 import (
    extragalactic_wd2_model,
    extragalactic_wd2_A_model,
)

# ── Cosmic strings ────────────────────────────────────────────────────────────
from gwb_templates.cosmic_string_templates.cosmic_string_model_i import (
    cosmic_string_model_i_model,
)
from gwb_templates.cosmic_string_templates.cosmic_string_model_i_edf import (
    cosmic_string_model_i_edf_model,
)
from gwb_templates.cosmic_string_templates.cosmic_string_model_i_eos import (
    cosmic_string_model_i_eos_model,
)
from gwb_templates.cosmic_string_templates.cosmic_string_model_ii import (
    abelian_higgs_model_ii_model,
    cosmic_string_model_ii_model,
)

_REGISTRY: dict[str, Signal_model] = {
    # generic
    "amplitude": amplitude_model,
    "power_law": power_law_model,
    "lognormal_bump": lognormal_bump_model,
    "broken_power_law": broken_power_law_model,
    "broken_power_law_fixed_smoothness": broken_power_law_fixed_smoothness_model,
    # FOPT
    "fopt_broken_power_law": fopt_broken_power_law_model,
    "fopt_old": fopt_broken_power_law_old_model,
    # phase transitions
    "broken_power_law_a1": broken_power_law_a1_model,
    "double_broken_power_law": double_broken_power_law_model,
    "double_broken_power_law_rf": double_broken_power_law_rf_model,
    "two_double_broken_power_laws": two_double_broken_power_laws_model,
    "pt_sound_waves": pt_sound_waves_model,
    "pt_turbulence": pt_turbulence_model,
    "pt_collision": pt_collision_model,
    "pt_plasma": pt_plasma_model,
    # inflation
    "double_peak": double_peak_model,
    "double_peak_sharp": double_peak_sharp_model,
    "double_peak_sharp_log": double_peak_sharp_log_model,
    "excited_states": excited_states_model,
    "flat_resonant": flat_resonant_model,
    "flat_resonant_log": flat_resonant_log_model,
    "lognormal_bump_sharp": lognormal_bump_sharp_model,
    "lognormal_bump_sharp_log": lognormal_bump_sharp_log_model,
    "sharp_feature": sharp_feature_model,
    "sharp_feature_log": sharp_feature_log_model,
    "resonant_feature": resonant_feature_model,
    "resonant_feature_log": resonant_feature_log_model,
    # foregrounds
    "galactic_binaries": galactic_binaries_model,
    "galactic_binaries_A": galactic_binaries_A_model,
    "galactic_binaries_old": galactic_binaries_old_model,
    "galactic_binaries_old_A": galactic_binaries_old_A_model,
    "extragalactic_sobbh_bns": extragalactic_sobbh_bns_model,
    "extragalactic_sobbh_bns_A": extragalactic_sobbh_bns_A_model,
    "extragalactic_wd": extragalactic_wd_model,
    "extragalactic_wd_A": extragalactic_wd_A_model,
    "extragalactic_wd2": extragalactic_wd2_model,
    "extragalactic_wd2_A": extragalactic_wd2_A_model,
    # cosmic strings
    "cosmic_string_model_i": cosmic_string_model_i_model,
    "cosmic_string_model_i_edf": cosmic_string_model_i_edf_model,
    "cosmic_string_model_i_eos": cosmic_string_model_i_eos_model,
    "cosmic_string_model_ii": cosmic_string_model_ii_model,
    "abelian_higgs_model_ii": abelian_higgs_model_ii_model,
}


def get_template(signal_label: str) -> Signal_model:
    """
    Return a configured signal model by label.

    Args:
        signal_label: Name of the signal model.

    Returns:
        A Signal_model instance exposing template, dtemplate, and d2template.

    Raises:
        ValueError: If signal_label is not in the registry.
    """
    try:
        return _REGISTRY[signal_label]
    except KeyError:
        raise ValueError(
            f"Unknown signal label '{signal_label}'. "
            f"Available labels: {sorted(_REGISTRY)}"
        )
