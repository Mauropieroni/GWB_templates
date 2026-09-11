from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar,TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
#import math
from gwb_templates.template import AnalyticTemplate
ArrayLike: TypeAlias = jtp.ArrayLike

# Constant parameters
m_Pl = 1.22 * 10**(19.0)  # non-reduced Planck mass in GeV
M_Pl = m_Pl / jnp.sqrt(8.0 * jnp.pi)  # reduced Planck Mass in GeV
c_grams_to_GeV = 5.6 * (10.**(23.0))

class PBH_double_peak_non_Gaussian(AnalyticTemplate):
     bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
        @article{Papanikolaou:2024kjb,
        author = "Papanikolaou, Theodoros and He, Xin-Chen and Ma, Xiao-Han and Cai, Yi-Fu and Saridakis, Emmanuel N. and Sasaki, Misao",
        title = "{New probe of non-Gaussianities with primordial black hole induced gravitational waves}",
        eprint = "2403.00660",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        reportNumber = "YITP-24-22",
        doi = "10.1016/j.physletb.2024.138997",
        journal = "Phys. Lett. B",
        volume = "857",
        pages = "138997",
        year = "2024"
        }
        """,
        r"""
        @article{He:2024luf,
        author = "He, Xin-Chen and Cai, Yi-Fu and Ma, Xiao-Han and Papanikolaou, Theodoros and Saridakis, Emmanuel N. and Sasaki, Misao",
        title = "{Gravitational waves from primordial black hole isocurvature: the effect of non-Gaussianities}",
        eprint = "2409.11333",
        archivePrefix = "arXiv",
        primaryClass = "astro-ph.CO",
        reportNumber = "YITP-24-93",
        doi = "10.1088/1475-7516/2024/12/039",
        journal = "JCAP",
        volume = "12",
        pages = "039",
        year = "2024"
        }
        """
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
            "M_PBH": r"$M_\mathrm{PBH}\;\mathrm{in}\;\mathrm{grams}$",
            "Omega_f": r"$\Omega_\mathrm{PBH,f}$",
            "tau_NL": r"$\tau_\mathrm{NL}$",
        }
        default_priors = {
            "M_PBH": {"min": 10.0, "max": 10.**(8.0)},
            "Omega_f": {"min": 10.**(-12.0), "max": 10.**(-4.0)},
            "tau_NL": {"min": 10.**(-11.0), "max": 10.**(-2.0)},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "PBH double peak non Gaussian GW signal",
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )
    
    def f_UV(M_PBH: jax.Array) -> jax.Array:
        return 1.7 * (10**(3.0)) * (M_PBH / (10.**4.0))**(-5.0 / 6.0)

    def f_d(self, M_PBH: jax.Array, Omega_f: jax.Array) -> jax.Array:
        return (4.0 * 10.**(-7.0)) * self.f_UV(M_PBH) * (Omega_f / 10.**(-10.0))**(2.0 / 3.0)

    def f_evap(self, M_PBH: jax.Array) -> jax.Array:
        return 0.75 * ((M_Pl / c_grams_to_GeV / M_PBH)**(2.0 / 3.0)) * self.f_UV(M_PBH)

    def f_c(self, M_PBH: jax.Array, tau_NL: jax.Array) -> jax.Array:
        return 7 * 10.**(-3.0) * ((tau_NL / 10.**(-3.0))**(1.0 / 3.0)) * self.f_UV(M_PBH)

    def f_L(self, M_PBH: jax.Array, Omega_f: jax.Array) -> jax.Array:
        return (5.0 * 10.**(-7.0)) * self.f_UV(M_PBH) * ((Omega_f / 10.**(-10.0))**(11.0 / 21.0)) * ((M_PBH / 10.**4.0)**(-1.0 / 7.0))

    def Omega_GW_non_Gaussian_peak(self, f: jax.Array, M_PBH: jax.Array, Omega_f: jax.Array, tau_NL: jax.Array) -> jax.Array:
        return ((tau_NL / 10.**(-3.0))**(2.0)) * (10.**(10.0)) * ((M_PBH / 10.**4.0)**(34.0 / 9.0)) * ((f / self.f_UV(M_PBH))**(17.0 / 3.0)) / (1.0 + (f / self.f_d(M_PBH, Omega_f))**(24.0 / 3.0))

    def Omega_GW_non_Gaussian(self, f: jax.Array, M_PBH: jax.Array, Omega_f: jax.Array, tau_NL: jax.Array) -> jax.Array:
        # Python 'if/else' statements break JAX compilation. 
        # Using jnp.where evaluates both conditions element-wise.
        f_L_val = self.f_L(M_PBH, Omega_f)
        peak_at_f = self.Omega_GW_non_Gaussian_peak(f, M_PBH, Omega_f, tau_NL)
        peak_at_f_L = self.Omega_GW_non_Gaussian_peak(f_L_val, M_PBH, Omega_f, tau_NL)
        
        return jnp.where(f >= f_L_val, peak_at_f, peak_at_f_L * (f / f_L_val))

    def Omega_GW_Gaussian(self, f: jax.Array, M_PBH: jax.Array, Omega_f: jax.Array, tau_NL: jax.Array) -> jax.Array:
        # Replaced Python conditional with jnp.where
        f_UV_val = self.f_UV(M_PBH)
        gaussian_val = (10.**(-28.0)) * ((f / f_UV_val)**(11.0 / 3.0)) * ((M_PBH / (10.**4.0))**(34.0 / 9.0)) * ((Omega_f / (10.**(-10.0)))**(16.0 / 3.0))
        
        return jnp.where(f <= 2.0 * f_UV_val, gaussian_val, 0.0)

    def omega_gw_h2(self, f: jax.Array, M_PBH: jax.Array, Omega_f: jax.Array, tau_NL: jax.Array) -> jax.Array:
        return self.Omega_GW_non_Gaussian(f, M_PBH, Omega_f, tau_NL) + self.Omega_GW_Gaussian(f, M_PBH, Omega_f, tau_NL)

    def _grad_theta_omega_gw_h2_analytical(
            self,
            frequency: jax.Array,
            theta: jax.Array,
        ) -> jax.Array:
            r"""
            Jacobian via JAX forward-mode autodiff of :meth:`omega_gw_h2`.
    
            We do not hand-derive a closed-form Jacobian here: with 3 scalar
            parameters and many chained power-law/piecewise terms, a manual
            derivation would be long and error-prone, and mathematically
            offers no advantage over automatic differentiation of the
            (already analytic) forward formula. ``jacfwd`` is used rather than
            the reverse-mode default.
            """
            M_PBH,Omega_f,tau_NL = (
            theta[0], theta[1], theta[2]
            )
    
            def Omega_GW_signal(
                M_PBH: jax.Array,
                Omega_f: jax.Array,
                tau_NL: jax.Array,
            ) -> jax.Array:
                return self.omega_gw_h2(
                    frequency, M_PBH, Omega_f, tau_NL
                )
    
            jac = jax.jacfwd(Omega_GW_signal, argnums=(0, 1, 2))(
                 M_PBH, Omega_f,tau_NL)
            return jnp.stack(jac, axis=-1) 