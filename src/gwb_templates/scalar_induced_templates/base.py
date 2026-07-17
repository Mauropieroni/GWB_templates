r"""
Scalar-induced gravitational-wave (SIGW) templates — package root.

This subpackage collects templates for the **scalar-induced** GW background:
the stochastic signal sourced at second order by a primordial curvature power
spectrum :math:`\mathcal{P}_\zeta(k)`. The layout reflects the physics, not one
particular code:

* :class:`ScalarInducedTemplate` (this module) is the backend-agnostic marker
  base. Every SIGW template — however its spectrum is computed — derives from
  it, so ``isinstance(model, ScalarInducedTemplate)`` selects the whole family.
  It is pure (no optional dependencies), so importing it never pulls in a
  numerical backend.

* **Direct / analytic templates** live at this package root, one module each
  (e.g. a *geometric* closed-form approximation of :math:`\Omega_{\mathrm{GW}}`
  for a log-normal peak in :math:`\mathcal{P}_\zeta`). These are ordinary
  autodiff templates with no heavy dependency.

* **Backend subpackages** live in their own folders — currently
  :mod:`gwb_templates.scalar_induced_templates.sigway`, which computes the
  spectrum numerically with the optional `SIGWAY
  <https://github.com/jonaselgammal/SIGWAY>`_ package. Additional backends
  (other solvers, emulators, …) can be added as sibling subpackages without
  touching this base.

Naming conventions:

* Direct analytic templates: descriptive module name + a class name that names
  the approximation and shape, e.g. ``geometric_lognormal.py`` →
  ``GeometricLognormalSIGW``.
* Backend templates: plain shape-named modules inside the backend package
  (the package path already qualifies them, e.g.
  ``...sigway.lognormal`` → ``SIGWAYLognormal``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from gwb_templates.template import Array, DifferentiationBackend, Template


class ScalarInducedTemplate(Template):
    r"""
    Abstract marker base for scalar-induced GW background templates.

    Subclass this (directly, for analytic templates, or via a backend base
    such as :class:`~gwb_templates.scalar_induced_templates.sigway.base.SIGWAYTemplate`)
    so that all SIGW templates share a common ancestor for discovery and
    typing. Defaults match the analytic case (jittable, autodiff); backend
    subclasses override as needed (e.g. a numerical solver may set
    ``jittable = False`` and use finite differences).

    This class carries no optional dependencies, so importing it is always
    safe regardless of which backends are installed.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: Array,
        *args: Any,
        **kwargs: Any,
    ) -> Array:  # pragma: no cover - re-declared to keep this base abstract
        ...
