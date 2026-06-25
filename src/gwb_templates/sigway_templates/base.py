r"""
SIGWAY-backed template base class and generic factory.

This module wires the public `SIGWAY <https://github.com/jonaselgammal/SIGWAY>`_
package into the :class:`~gwb_templates.template.Template` hierarchy. SIGWAY
computes the scalar-induced gravitational-wave (SIGW) energy-density spectrum
:math:`\Omega_{\mathrm{GW}}(f)` from a primordial curvature power spectrum
:math:`\mathcal{P}_\zeta(k)` by composing three ingredients:

* a :class:`sigway.perturbations.ScalarPerturbations` (the source
  :math:`\mathcal{P}_\zeta(k)`, analytic or solved from the
  Mukhanov-Sasaki equation),
* a :class:`sigway.kernels.Kernel` (the cosmological transfer kernel, which
  also carries the overall normalisation), and
* a :class:`sigway.integrators.Integrator` (the :math:`(s, t)` quadrature;
  Simpson by default).

:class:`SIGWAYTemplate` is a *new sibling* of
:class:`~gwb_templates.template.AnalyticTemplate` /
:class:`~gwb_templates.template.NumericalTemplate`: it owns a configured
:class:`sigway.spectrum.OmegaGW` instance and exposes it through the standard
``omega_gw_h2`` / derivative API. Concrete templates only have to implement
:meth:`SIGWAYTemplate.build_model`, returning a fully composed ``OmegaGW``.
This keeps the *full* compositional flexibility of SIGWAY — any perturbation
source, any kernel, any integrator, any grids — rather than hard-coding one
particular combination.

For ad-hoc spectra there is :func:`sigway_template`, a factory that builds (and
registers) a :class:`SIGWAYTemplate` subclass from the SIGWAY pieces directly.

.. note::
   SIGWAY is an **optional** dependency. Importing this module requires
   ``sigway`` (and a JAX new enough for ``jax.scipy.special.sici``, used by
   the EMD kernel). The package ``__init__`` imports the SIGWAY templates
   inside a ``try/except`` so ``import gwb_templates`` keeps working without
   SIGWAY installed; the templates simply do not register in that case.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, ClassVar

import jax.numpy as jnp

# SIGWAY is an optional dependency; importing this module is what gates
# registration of the SIGWAY templates (see gwb_templates/__init__.py).
from sigway.spectrum import OmegaGW
from sigway.perturbations import AnalyticPerturbations, ScalarPerturbations
from sigway.kernels import Kernel, RadiationKernel

from gwb_templates.template import (
    Array,
    ArrayLike,
    DifferentiationBackend,
    Template,
)

# Citation for the template suite tested in arXiv:2501.11320 and the SIGWAY
# code. Subclasses default to this; override per-template where a different
# primary reference applies.
SIGWAY_BIBTEX: str = r"""
@article{LISACosmologyWorkingGroup:2025sigw,
  title         = {Reconstructing Primordial Curvature Perturbations via
                   Scalar-Induced Gravitational Waves with LISA},
  author        = {El Gammal, Jonas and Ghaleb, Aya and Franciolini, Gabriele
                   and Papanikolaou, Theodoros and Peloso, Marco and Perna,
                   Gabriele and Pieroni, Mauro and Ricciardone, Angelo and
                   Rosati, Robert and Tasinato, Gianmassimo and others},
  collaboration = {LISA Cosmology Working Group},
  year          = {2025},
  eprint        = {2501.11320},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}
""".strip()


class SIGWAYTemplate(Template):
    r"""
    Base class for templates whose spectrum is computed by SIGWAY.

    A :class:`SIGWAYTemplate` wraps a single configured
    :class:`sigway.spectrum.OmegaGW` instance, built once in
    :meth:`build_model`. The free parameters are taken straight from the
    SIGWAY model (``perturbations.param_names + kernel.param_names``), so the
    GWB-template parameter vector is automatically consistent with what SIGWAY
    expects.

    Differentiation:

    * When the underlying scalar perturbations are *jittable* (analytic
      :math:`\mathcal{P}_\zeta`), parameter gradients are delegated to
      :meth:`sigway.spectrum.OmegaGW.jacobian`. That uses forward-mode autodiff
      for smooth parameters and central finite differences for any parameter
      flagged ``nonsmooth`` by SIGWAY (e.g. a hard cutoff :math:`k_{\max}`),
      which is exactly the right thing for a Heaviside-type feature.
    * When the perturbations are *not* jittable (the Mukhanov-Sasaki solver
      calls SciPy ODE routines), the instance transparently downgrades to
      finite-difference derivatives via the base class, and advertises
      ``jittable = False``.

    Subclasses implement :meth:`build_model`; everything else is inherited.
    """

    # Analytic-P_zeta SIGWAY models are jittable and autodiff-friendly. The
    # Mukhanov-Sasaki path overrides both of these *per instance* at
    # construction time (see __init__).
    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    bibtex_entries: ClassVar[tuple[str, ...]] = (SIGWAY_BIBTEX,)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    def build_model(self) -> OmegaGW:
        """
        Build and return the configured :class:`sigway.spectrum.OmegaGW`.

        Called once during ``__init__`` (before parameter-name inference).
        Subclasses set any configuration they need (grids, kernel, norm) on
        ``self`` *before* calling ``super().__init__`` and read it back here —
        mirroring how :class:`~gwb_templates.generic_templates.power_law.PowerLaw`
        stores ``self.pivot``.

        Declared abstract so that :class:`SIGWAYTemplate` itself is not
        registered as a concrete template.
        """
        raise NotImplementedError

    # mark abstract without importing abc.abstractmethod semantics on the
    # concrete omega_gw_h2: build_model is the abstract hook that keeps this
    # base class out of the registry.
    build_model.__isabstractmethod__ = True  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_type: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        # Build the SIGWAY model first: it is the source of truth for the
        # parameter names that Template.__init__ will record.
        self._model: OmegaGW = self.build_model()

        # SIGWAY tells us whether its compiled / differentiable path is usable.
        self._sigway_jittable: bool = bool(
            getattr(self._model.perturbations, "jittable", True)
        )
        if not self._sigway_jittable:
            # Mukhanov-Sasaki source: not traceable by jit/autodiff. Shadow the
            # class-level flags on this instance so the base-class derivative
            # helpers fall back to finite differences.
            self.jittable = False
            self.differentiation_backend = "finite_difference"

        super().__init__(
            model_name=model_name,
            model_type=model_type,
            model_label=model_label,
            parameter_labels=parameter_labels,
            prior_by_param=prior_by_param,
        )

    # ------------------------------------------------------------------
    # Parameter names come from the SIGWAY model, not the call signature
    # ------------------------------------------------------------------
    def _infer_parameter_names_from_signature(self) -> list[str]:
        """Read the free-parameter names straight off the SIGWAY model."""
        return list(self._model.parameter_names)

    # ------------------------------------------------------------------
    # Spectrum
    # ------------------------------------------------------------------
    def omega_gw_h2(self, frequency: Array, *theta: Any) -> Array:
        r"""
        Evaluate :math:`\Omega_{\mathrm{GW}} h^2(f)` via SIGWAY.

        The parameters are forwarded positionally to the SIGWAY model in
        ``parameter_names`` order (perturbation parameters first, then kernel
        parameters). The overall normalisation — and therefore whether the
        result is :math:`\Omega_{\mathrm{GW}} h^2` or a dimensionless ratio —
        is set by the kernel passed to :meth:`build_model`.
        """
        return self._model(jnp.asarray(frequency), *theta)

    # ------------------------------------------------------------------
    # Parameter gradient: use SIGWAY's own Jacobian when available
    # ------------------------------------------------------------------
    def grad_theta_omega_gw_h2(
        self,
        frequency: Array,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""
        :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`.

        For jittable (analytic) sources this delegates to
        :meth:`sigway.spectrum.OmegaGW.jacobian`, which mixes autodiff with
        finite differences for any SIGWAY-flagged non-smooth parameter. For
        the Mukhanov-Sasaki source it falls back to the base-class
        finite-difference path. The returned shape is
        ``frequency.shape + (n_params,)``, matching the base class.
        """
        if self._sigway_jittable:
            theta = self._to_parameter_vector(parameters)
            return self._model.jacobian(jnp.asarray(frequency), theta)
        return super().grad_theta_omega_gw_h2(frequency, parameters, *args, **kwargs)


# =============================================================================
# Generic factory
# =============================================================================
def sigway_template(
    name: str,
    *,
    perturbations: ScalarPerturbations | None = None,
    pzeta: Callable[..., Any] | None = None,
    parameter_names: Sequence[str] | None = None,
    nonsmooth_params: Sequence[str] = (),
    kernel: Kernel | None = None,
    integrator: Any = None,
    s: Any = None,
    t: Any = None,
    f: Any = None,
    upsample: bool = False,
    bibtex_entries: Sequence[str] = (SIGWAY_BIBTEX,),
    model_label: str | None = None,
    differentiation_backend: DifferentiationBackend | None = None,
    jittable: bool | None = None,
    register: bool = True,
) -> type[SIGWAYTemplate]:
    r"""
    Build (and register) a :class:`SIGWAYTemplate` subclass from SIGWAY pieces.

    This is the general-purpose entry point: it exposes SIGWAY's full
    composition surface so you can pair *any* perturbation source with *any*
    kernel, integrator and grids — exactly as you would when using SIGWAY
    directly — while getting a first-class GWB ``Template`` back.

    Provide the source either as a ready
    :class:`sigway.perturbations.ScalarPerturbations` (``perturbations=``) or
    as a closed-form callable plus its parameter names (``pzeta=`` and
    ``parameter_names=``), in which case it is wrapped in
    :class:`sigway.perturbations.AnalyticPerturbations`.

    Args:
        name: Class name; also the registry key used by
            :func:`~gwb_templates.template.get_template_from_registry`.
        perturbations: A SIGWAY perturbation object. Mutually exclusive with
            ``pzeta``.
        pzeta: Closed-form ``pzeta(k, *params)`` (JAX). Requires
            ``parameter_names``.
        parameter_names: Ordered names of ``pzeta``'s parameters.
        nonsmooth_params: Subset of ``parameter_names`` entering a sharp
            feature (e.g. a hard cutoff), so SIGWAY uses finite differences
            for their Jacobian columns.
        kernel: A SIGWAY kernel. Defaults to
            :class:`sigway.kernels.RadiationKernel` (``"RD"`` norm, i.e.
            output already in :math:`\Omega_{\mathrm{GW}} h^2`).
        integrator: Optional explicit :class:`sigway.integrators.Integrator`.
            When omitted, SIGWAY builds a Simpson integrator from ``s`` and
            ``t``.
        s, t: Quadrature grids (arrays or ``(k, *theta)`` callables). Required
            when ``integrator`` is not given.
        f: Optional fixed internal frequency grid for ``upsample``.
        upsample: If ``True``, integrate on ``f`` and interpolate to the
            call-time frequencies (requires ``f``).
        bibtex_entries: Citation(s) for this template.
        model_label: Optional display label.
        differentiation_backend: Force a backend; by default inferred from the
            perturbation source.
        jittable: Force the ``jittable`` flag; by default inferred.
        register: If ``False``, remove the class from the registry after
            creation (useful for throwaway / notebook spectra).

    Returns:
        The freshly created :class:`SIGWAYTemplate` subclass.
    """
    if (perturbations is None) == (pzeta is None):
        raise ValueError(
            "Provide exactly one of `perturbations` or `pzeta` "
            "(with `parameter_names`)."
        )
    if pzeta is not None and parameter_names is None:
        raise ValueError("`pzeta` requires `parameter_names`.")

    resolved_kernel = RadiationKernel() if kernel is None else kernel
    pz_names = None if parameter_names is None else tuple(parameter_names)
    ns_params = tuple(nonsmooth_params)

    def build_model(self: SIGWAYTemplate) -> OmegaGW:
        if perturbations is not None:
            pert: ScalarPerturbations = perturbations
        else:
            pert = AnalyticPerturbations(pzeta, pz_names, nonsmooth_params=ns_params)
        return OmegaGW(
            pert,
            resolved_kernel,
            integrator=integrator,
            s=s,
            t=t,
            f=f,
            upsample=upsample,
        )

    attrs: dict[str, Any] = {
        "build_model": build_model,
        "bibtex_entries": tuple(bibtex_entries),
        "__doc__": f"SIGWAY template '{name}' built via sigway_template().",
    }
    if jittable is not None:
        attrs["jittable"] = jittable
    if differentiation_backend is not None:
        attrs["differentiation_backend"] = differentiation_backend

    cls = type(name, (SIGWAYTemplate,), attrs)

    if model_label is not None:
        # Stash a default label without forcing it through __init__ kwargs.
        original_init = cls.__init__

        def _init(self, **kwargs):  # type: ignore[no-redef]
            kwargs.setdefault("model_label", model_label)
            original_init(self, **kwargs)

        cls.__init__ = _init  # type: ignore[assignment]

    if not register:
        Template._registry.pop(name, None)

    return cls
