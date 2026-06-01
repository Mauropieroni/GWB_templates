r"""
Standard power-law spectrum template.

Two-parameter model for a GWB:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha_{\mathrm{PL}}}
        \left(\frac{f}{f_{\mathrm{pivot}}}\right)^{n_T}

The default pivot frequency is 3 mHz (roughly the centre of the LISA
band). This is the simplest phenomenological GWB model and is widely
used as a baseline in stochastic-background searches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp

from gwb_templates.template import AnalyticTemplate

ArrayLike: TypeAlias = jtp.ArrayLike


class PowerLaw(AnalyticTemplate):
    r"""
    Standard power-law GWB spectrum.

    Free parameters
    ---------------
    log_amplitude
        Base-10 logarithm of the amplitude at the pivot frequency.
    tilt
        Spectral index.

    Configuration
    -------------
    pivot
        Reference frequency (Hz) used to normalize the power law.
        Defaults to :attr:`DEFAULT_PIVOT` (3 mHz). Stored as an instance
        attribute set at construction time; instantiate two ``PowerLaw``
        objects to sweep across pivots.
    """

    #: Default pivot frequency in Hz (LISA-band centre).
    DEFAULT_PIVOT: ClassVar[float] = 3e-3

    #: TODO: populate with the canonical PL-template references once we
    #: settle on which papers to cite by default.
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        pivot: float = DEFAULT_PIVOT,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            pivot: Reference frequency in Hz.
            model_name: Override instance identifier (see
                :class:`~gwb_templates.template.Template`).
            model_label: Override display label. Defaults to
                ``"Power Law Model"``.
            parameter_labels: Sparse override map for parameter display
                labels. Defaults provide LaTeX-friendly labels.
            prior_by_param: Sparse override map for parameter priors.
                Defaults to a broad uniform prior on each parameter.
        """
        self.pivot: float = float(pivot)

        default_labels = {
            "log_amplitude": r"$\alpha_{\mathrm{PL}}$",
            "tilt": r"$n_{T}$",
        }
        default_priors = {
            "log_amplitude": {"min": -20.0, "max": -5.0},
            "tilt": {"min": -10.0, "max": 10.0},
        }

        super().__init__(
            model_name=model_name,
            model_label=model_label if model_label is not None else "Power Law Model",
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
        tilt: ArrayLike,
    ) -> jax.Array:
        r"""
        Evaluate the power-law spectrum at ``frequency``.

        Args:
            frequency: Frequency value(s) in Hz.
            log_amplitude: :math:`\log_{10}` amplitude at the pivot.
            tilt: Spectral index.

        Returns:
            Spectrum :math:`\Omega_{\mathrm{GW}} h^2(f)` at each input
            frequency.
        """
        x = frequency / self.pivot
        return 10.0**log_amplitude * x**tilt

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian of the power-law spectrum.

        :math:`\partial/\partial(\log_{10}A) = \text{model} \cdot \ln 10`,
        :math:`\partial/\partial(\text{tilt}) = \text{model} \cdot \ln(f/f_{\rm pivot})`.
        """
        log_amplitude, tilt = theta
        model = self.omega_gw_h2(frequency, log_amplitude, tilt)
        d_logA = model * jnp.log(10.0)
        d_tilt = model * jnp.log(frequency / self.pivot)
        return jnp.stack((d_logA, d_tilt), axis=-1)
