r"""
Extragalactic white-dwarf binary foreground template.

Implements Eq. B1 of Boileau et al. 2025 (arXiv:2506.18390):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,x^{-\alpha_1}\,
        H^{(\alpha_1 - \alpha_2)\delta}

with :math:`x = f / f_{\mathrm{knee}}` and :math:`H = \tfrac{1}{2}(1 + x^{1/\delta})`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.template import AnalyticTemplate


class ExtragalacticWd(AnalyticTemplate):
    r"""
    Extragalactic WD-binary foreground (5-parameter form).

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude.
    f_knee
        Knee frequency (Hz).
    delta
        Transition width parameter.
    alpha1
        Low-frequency slope (model goes as :math:`f^{-\alpha_1}`).
    alpha2
        High-frequency slope (model goes as :math:`f^{-\alpha_2}`).
    """

    DEFAULT_MODEL_NAME: ClassVar[str] = "ewd"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Extragalactic WD Binaries"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD})$",
        "f_knee": r"$f_{\rm knee}$",
        "delta": r"$\delta$",
        "alpha1": r"$\alpha_1$",
        "alpha2": r"$\alpha_2$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
        "f_knee": {"min": 1e-5, "max": 0.1},
        "delta": {"min": 0.1, "max": 10.0},
        "alpha1": {"min": -5.0, "max": 5.0},
        "alpha2": {"min": -5.0, "max": 5.0},
    }

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Boileau:2025jkv,
    author = "Boileau, Guillaume and Bruel, Tristan and Toubiana, Alexandre and
        Lamberts, Astrid and Christensen, Nelson",
    title = "{Gravitational-wave background from extragalactic double white dwarfs for
        LISA}",
    eprint = "2506.18390",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1051/0004-6361/202556052",
    journal = "Astron. Astrophys.",
    volume = "702",
    pages = "A246",
    year = "2025"
}
""",
    )

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        f_knee: jax.Array,
        delta: jax.Array,
        alpha1: jax.Array,
        alpha2: jax.Array,
    ) -> jax.Array:
        x = frequency / f_knee
        low_pl = x ** (-alpha1)
        H = 0.5 * (1.0 + x ** (1.0 / delta))
        power = (alpha1 - alpha2) * delta
        return 10.0**log_amplitude * low_pl * H**power

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; columns [log_amplitude, f_knee, delta, alpha1, alpha2]."""
        log_amplitude, f_knee, delta, alpha1, alpha2 = (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            theta[4],
        )
        x = frequency / f_knee
        x_inv_delta = x ** (1.0 / delta)
        factor = 1.0 + x_inv_delta
        H = 0.5 * factor
        power = (alpha1 - alpha2) * delta
        model = ExtragalacticWd.omega_gw_h2(
            self, frequency, log_amplitude, f_knee, delta, alpha1, alpha2
        )

        d_logA = model * jnp.log(10.0)
        d_fknee = model / f_knee * (alpha1 - (power / H) * (0.5 / delta) * x_inv_delta)
        d_delta = (
            model
            * (alpha1 - alpha2)
            * (jnp.log(H) - x_inv_delta * jnp.log(x) / (delta * factor))
        )
        d_alpha1 = model * (-jnp.log(x) + delta * jnp.log(H))
        d_alpha2 = model * delta * (jnp.log(2.0) - jnp.log(factor))
        return jnp.stack([d_logA, d_fknee, d_delta, d_alpha1, d_alpha2], axis=-1)


class ExtragalacticWdA(ExtragalacticWd):
    r"""
    Extragalactic WD-binary foreground — amplitude-only variant.

    Free parameter: ``log_amplitude`` only. Inherits :attr:`bibtex_entries` and the
    underlying spectral shape from :class:`ExtragalacticWd`, fixing ``f_knee``,
    ``delta``, ``alpha1``, ``alpha2`` at the fiducial values from arXiv:2506.18390 (or
    values passed at construction).
    """

    DEFAULT_F_KNEE: ClassVar[jax.Array] = jnp.array(7e-3)
    DEFAULT_DELTA: ClassVar[jax.Array] = jnp.array(0.24)
    DEFAULT_ALPHA1: ClassVar[jax.Array] = jnp.array(-0.72)
    DEFAULT_ALPHA2: ClassVar[jax.Array] = jnp.array(2.43)

    DEFAULT_MODEL_NAME: ClassVar[str] = "ewd_a"
    DEFAULT_MODEL_LABEL: ClassVar[str] = "Extragalactic WD Binaries (amplitude only)"
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = {
        "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD})$",
    }
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = {
        "log_amplitude": {"min": -20.0, "max": -5.0},
    }

    def __init__(
        self,
        f_knee: jax.Array = DEFAULT_F_KNEE,
        delta: jax.Array = DEFAULT_DELTA,
        alpha1: jax.Array = DEFAULT_ALPHA1,
        alpha2: jax.Array = DEFAULT_ALPHA2,
        **kwargs: Any,
    ) -> None:
        self.f_knee: jax.Array = f_knee
        self.delta: jax.Array = delta
        self.alpha1: jax.Array = alpha1
        self.alpha2: jax.Array = alpha2

        super().__init__(**kwargs)

    def omega_gw_h2(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
    ) -> jax.Array:
        return super().omega_gw_h2(
            frequency, log_amplitude, self.f_knee, self.delta, self.alpha1, self.alpha2
        )

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian: single column d/d(log_amplitude)."""
        full_theta = jnp.array(
            [theta[0], self.f_knee, self.delta, self.alpha1, self.alpha2]
        )
        return super()._grad_theta_omega_gw_h2_analytical(frequency, full_theta)[:, :1]
