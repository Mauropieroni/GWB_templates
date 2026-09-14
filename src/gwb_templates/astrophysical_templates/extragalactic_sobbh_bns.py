r"""
Extragalactic compact binary merger foreground template.

Power-law spectrum with the 2/3 spectral index expected from GW emission
during the inspiral phase of compact binary mergers integrated over redshift.

Two variants are provided:

* :class:`ExtragalacticSobbhBns` — 2-parameter model (log amplitude + tilt).
* :class:`ExtragalacticSobbhBnsA` — 1-parameter amplitude-only model with
  tilt fixed at 2/3.

References:
  arXiv:2304.06368 (Babak et al. — SOBBH SGWB in LISA; the fixed :math:`f^{2/3}` tilt is
  the standard inspiral-dominated prediction used there). arXiv:1809.10360 (Chen, Huang
  & Huang — combined BBH + BNS SGWB and implications for LISA).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class ExtragalacticSobbhBns(AnalyticTemplate):
    r"""
    Extragalactic SOBBH+BNS foreground with free spectral index.

    .. math::

        \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,(f / f_{\mathrm{ref}})^{\alpha}

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
    DEFAULT_REF_FREQ: ClassVar[jax.Array] = jnp.array(1e-3)
    DEFAULT_MODEL_NAME: ClassVar[str] = "sobbh_bns"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Extragalactic SOBBH+BNS"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
        "tilt": r"$\alpha_{\rm EG}$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "tilt": {"min": -3.0, "max": 5.0},
    }

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
        ref_freq: jax.Array = DEFAULT_REF_FREQ,
        **kwargs: Any,
    ) -> None:
        self.ref_freq: jax.Array = ref_freq
        super().__init__(**kwargs)

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        tilt: jax.Array,
    ) -> jax.Array:
        x = frequency / self.ref_freq
        return 10.0**log_amplitude * x**tilt

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: columns [log_amplitude, tilt]."""
        model = ExtragalacticSobbhBns.omega_gw_h2(self, frequency, theta[0], theta[1])
        d_logA = model * jnp.log(10.0)
        d_tilt = model * jnp.log(frequency / self.ref_freq)
        return jnp.stack([d_logA, d_tilt], axis=-1)


class ExtragalacticSobbhBnsA(ExtragalacticSobbhBns):
    r"""
    Extragalactic SOBBH+BNS foreground with tilt fixed to 2/3.

    Free parameter: ``log_amplitude`` only. Inherits :attr:`bibtex_entries` and the
    underlying spectral shape from :class:`ExtragalacticSobbhBns`, fixing ``tilt`` to
    the value passed at construction.
    """

    DEFAULT_TILT: ClassVar[jax.Array] = jnp.array(2.0 / 3.0)

    DEFAULT_MODEL_NAME: ClassVar[str] = "sobbh_bns_a"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Extragalactic SOBBH+BNS (amplitude only)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EG})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
    }

    def __init__(
        self,
        ref_freq: jax.Array = ExtragalacticSobbhBns.DEFAULT_REF_FREQ,
        tilt: jax.Array = DEFAULT_TILT,
        **kwargs: Any,
    ) -> None:
        self.tilt: jax.Array = tilt
        super().__init__(ref_freq=ref_freq, **kwargs)

    def omega_gw_h2(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
    ) -> jax.Array:
        return super().omega_gw_h2(frequency, log_amplitude, self.tilt)

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        full_theta = jnp.array([theta[0], self.tilt])
        return super()._grad_theta_omega_gw_h2_analytical(frequency, full_theta)[:, :1]
