r"""
Extragalactic compact binary merger foreground template.

Power-law spectrum with the 2/3 spectral index expected from GW emission
during the inspiral phase of compact binary mergers integrated over redshift.

Two variants are provided:

* :class:`ExtragalacticSobbhBns` — 2-parameter model (log amplitude + tilt).
* :class:`ExtragalacticSobbhBnsA` — 1-parameter amplitude-only model with
  tilt fixed at 2/3.

References:
  arXiv:2304.06368 (Babak et al. — SOBBH SGWB in LISA; the fixed
  :math:`f^{2/3}` tilt is the standard inspiral-dominated prediction
  used there).
  arXiv:1809.10360 (Chen, Huang & Huang — combined BBH + BNS SGWB
  and implications for LISA).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jtp.ArrayLike


class ExtragalacticSobbhBns(AnalyticTemplate):
    r"""
    Extragalactic SOBBH+BNS foreground with free spectral index.

    .. math::

        \Omega_{\mathrm{GW}} h^2(f) =
            10^{\log_{10} A}\,(f / f_{\mathrm{ref}})^{\alpha}

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude at the reference frequency.
    tilt
        Spectral index.

    Configuration
    -------------
    ref_freq
        Reference frequency (Hz). Defaults to 1 mHz.
    """

    #: Default reference frequency in Hz.
    DEFAULT_REF_FREQ: ClassVar[float] = 1e-3

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Babak:2023lro,
    author = "Babak, Stanislav and Caprini, Chiara and Figueroa, Daniel G. and Karnesis,
        Nikolaos and Marcoccia, Paolo and Nardini, Germano and Pieroni, Mauro and
        Ricciardone, Angelo and Sesana, Alberto and Torrado, Jes\'us",
    title = "{Stochastic gravitational wave background from stellar origin binary black
        holes in LISA}",
    eprint = "2304.06368",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2023/08/034",
    journal = "JCAP",
    volume = "08",
    pages = "034",
    year = "2023"
}
""",
        r"""
@article{Chen:2018rzo,
    author = "Chen, Zu-Cheng and Huang, Fan and Huang, Qing-Guo",
    title = "{Stochastic Gravitational-wave Background from Binary Black Holes and
        Binary Neutron Stars and Implications for LISA}",
    eprint = "1809.10360",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.3847/1538-4357/aaf581",
    journal = "Astrophys. J.",
    volume = "871",
    number = "1",
    pages = "97",
    year = "2019"
}
""",
    )

    def __init__(
        self,
        ref_freq: float = DEFAULT_REF_FREQ,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.ref_freq: float = float(ref_freq)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
            "tilt": r"$\alpha_{\rm EG}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "tilt": {"min": -3.0, "max": 5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Extragalactic SOBBH+BNS"
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
        frequency: Array,
        log_amplitude: Array,
        tilt: Array,
    ) -> Array:
        x = jnp.asarray(frequency) / self.ref_freq
        return 10.0**log_amplitude * x**tilt

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: Array,
        theta: Array,
    ) -> Array:
        """Analytic Jacobian: columns [log_amplitude, tilt]."""
        fvec = jnp.asarray(frequency)
        model = self.omega_gw_h2(fvec, theta[0], theta[1])
        d_logA = model * jnp.log(10.0)
        d_tilt = model * jnp.log(fvec / self.ref_freq)
        return jnp.stack([d_logA, d_tilt], axis=-1)


class ExtragalacticSobbhBnsA(AnalyticTemplate):
    r"""
    Extragalactic SOBBH+BNS foreground with tilt fixed to 2/3.

    Free parameter: ``log_amplitude`` only.
    """

    DEFAULT_REF_FREQ: ClassVar[float] = 1e-3
    DEFAULT_TILT: ClassVar[float] = 2.0 / 3.0

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Babak:2023lro,
    author = "Babak, Stanislav and Caprini, Chiara and Figueroa, Daniel G. and Karnesis,
        Nikolaos and Marcoccia, Paolo and Nardini, Germano and Pieroni, Mauro and
        Ricciardone, Angelo and Sesana, Alberto and Torrado, Jes\'us",
    title = "{Stochastic gravitational wave background from stellar origin binary black
        holes in LISA}",
    eprint = "2304.06368",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2023/08/034",
    journal = "JCAP",
    volume = "08",
    pages = "034",
    year = "2023"
}
""",
        r"""
@article{Chen:2018rzo,
    author = "Chen, Zu-Cheng and Huang, Fan and Huang, Qing-Guo",
    title = "{Stochastic Gravitational-wave Background from Binary Black Holes and
        Binary Neutron Stars and Implications for LISA}",
    eprint = "1809.10360",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.3847/1538-4357/aaf581",
    journal = "Astrophys. J.",
    volume = "871",
    number = "1",
    pages = "97",
    year = "2019"
}
""",
    )

    def __init__(
        self,
        ref_freq: float = DEFAULT_REF_FREQ,
        tilt: float = DEFAULT_TILT,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.ref_freq: float = float(ref_freq)
        self.tilt: float = float(tilt)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Extragalactic SOBBH+BNS (amplitude only)"
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
        frequency: Array,
        log_amplitude: Array,
    ) -> Array:
        x = jnp.asarray(frequency) / self.ref_freq
        return 10.0**log_amplitude * x**self.tilt

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: Array,
        theta: Array,
    ) -> Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        model = self.omega_gw_h2(jnp.asarray(frequency), theta[0])
        d_logA = model * jnp.log(10.0)
        return d_logA[..., None]
