r"""
Extragalactic white-dwarf binary foreground template.

Implements Eq. B1 of Boileau et al. 2025 (arXiv:2506.18390):

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\log_{10} A}\,x^{-\alpha_1}\,
        H^{(\alpha_1 - \alpha_2)\delta}

with :math:`x = f / f_{\mathrm{knee}}` and
:math:`H = \tfrac{1}{2}(1 + x^{1/\delta})`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


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

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD})$",
            "f_knee": r"$f_{\rm knee}$",
            "delta": r"$\delta$",
            "alpha1": r"$\alpha_1$",
            "alpha2": r"$\alpha_2$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "f_knee": {"min": 1e-5, "max": 0.1},
            "delta": {"min": 0.1, "max": 10.0},
            "alpha1": {"min": -5.0, "max": 5.0},
            "alpha2": {"min": -5.0, "max": 5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label if model_label is not None else "Extragalactic WD Binaries"
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
        log_amplitude: ArrayLike,
        f_knee: ArrayLike,
        delta: ArrayLike,
        alpha1: ArrayLike,
        alpha2: ArrayLike,
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        x = fvec / f_knee
        low_pl = x ** (-alpha1)
        H = 0.5 * (1.0 + x ** (1.0 / delta))
        power = (alpha1 - alpha2) * delta
        return 10.0**log_amplitude * low_pl * H**power

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
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
        fvec = jnp.asarray(frequency)
        x = fvec / f_knee
        x_inv_delta = x ** (1.0 / delta)
        factor = 1.0 + x_inv_delta
        H = 0.5 * factor
        power = (alpha1 - alpha2) * delta
        model = self.omega_gw_h2(fvec, log_amplitude, f_knee, delta, alpha1, alpha2)

        d_logA = model * jnp.log(10.0)
        d_fknee = model / f_knee * (alpha1 - (power / H) * (0.5 / delta) * x_inv_delta)
        d_delta = model * (alpha1 - alpha2) * (
            jnp.log(H) - x_inv_delta * jnp.log(x) / (delta * factor)
        )
        d_alpha1 = model * (-jnp.log(x) + delta * jnp.log(H))
        d_alpha2 = model * delta * (jnp.log(2.0) - jnp.log(factor))
        return jnp.stack(
            [d_logA, d_fknee, d_delta, d_alpha1, d_alpha2], axis=-1
        )


class ExtragalacticWdA(AnalyticTemplate):
    r"""
    Extragalactic WD-binary foreground — amplitude-only variant.

    Shape parameters are fixed at the fiducial values from
    arXiv:2506.18390 but can be overridden via constructor kwargs.
    """

    DEFAULT_F_KNEE: ClassVar[float] = 7e-3
    DEFAULT_DELTA: ClassVar[float] = 0.24
    DEFAULT_ALPHA1: ClassVar[float] = -0.72
    DEFAULT_ALPHA2: ClassVar[float] = 2.43

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

    def __init__(
        self,
        f_knee: float = DEFAULT_F_KNEE,
        delta: float = DEFAULT_DELTA,
        alpha1: float = DEFAULT_ALPHA1,
        alpha2: float = DEFAULT_ALPHA2,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self.f_knee: float = float(f_knee)
        self.delta: float = float(delta)
        self.alpha1: float = float(alpha1)
        self.alpha2: float = float(alpha2)

        default_labels = {
            "log_amplitude": r"$\log_{10}(h^2\,\Omega_{\rm EWD})$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Extragalactic WD Binaries (amplitude only)"
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
        log_amplitude: ArrayLike,
    ) -> jax.Array:
        fvec = jnp.asarray(frequency)
        x = fvec / self.f_knee
        low_pl = x ** (-self.alpha1)
        H = 0.5 * (1.0 + x ** (1.0 / self.delta))
        power = (self.alpha1 - self.alpha2) * self.delta
        return 10.0**log_amplitude * low_pl * H**power

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: ArrayLike,
        theta: jax.Array,
    ) -> jax.Array:
        """Analytic Jacobian; single column d/d(log_amplitude)."""
        model = self.omega_gw_h2(jnp.asarray(frequency), theta[0])
        return (model * jnp.log(10.0))[..., None]
