r"""
Ultra-slow-roll single-field scalar-induced GW template (SIGWAY, Mukhanov-Sasaki).

Mirrors the ``usr_ms`` configuration tested in arXiv:2501.11320. Unlike the
analytic templates, the source :math:`\mathcal{P}_\zeta(k)` is **not** a closed
form: it is obtained by numerically solving the Mukhanov-Sasaki equation for a
single-field, quasi-inflection-point potential

.. math::

    V(\phi) = \frac{\lambda v^4}{12}\,
        \frac{f^2\,(6 - 4 a f + 3 f^2)}{(1 + b f^2)^2}, \qquad f = \phi/v,

with :math:`b = (1+n_{\mathrm{fac}})\bigl(1 - a^2/3
+ \tfrac{a^2}{3}(9/(2a^2) - 1)^{2/3}\bigr)`, via
:class:`sigway.ms_solver.SingleFieldSolver`. The induced GWs are then evolved
through the radiation-domination kernel.

Because the solver calls SciPy ODE routines, this path is **not** jittable and
**not** autodiff-differentiable: the template advertises ``jittable = False``
and computes parameter derivatives by finite differences (handled
transparently by :class:`~gwb_templates.sigway_templates.base.SIGWAYTemplate`).

Free parameters
---------------
``a``, ``lam``, ``v``, ``nfac`` — the potential parameters forwarded to the
solver (in this order).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from sigway.perturbations import SingleFieldPerturbations
from sigway.kernels import RadiationKernel
from sigway.ms_solver import SingleFieldSolver
from sigway.spectrum import OmegaGW

from gwb_templates.sigway_templates.base import SIGWAYTemplate


def usr_potential(
    phi: jax.Array,
    a: jax.Array,
    lam: jax.Array,
    v: jax.Array,
    nfac: jax.Array,
) -> jax.Array:
    r"""Quasi-inflection-point USR potential :math:`V(\phi)`."""
    b = (1 + nfac) * (1 - a**2 / 3 + a**2 / 3 * (9 / (2 * a**2) - 1) ** (2 / 3))
    f = phi / v
    return lam * v**4 / 12 * f**2 * (6 - 4 * a * f + 3 * f**2) / (1 + b * f**2) ** 2


def usr_t_grid(nlo: int = 200, nhi: int = 800, nf: int = 200) -> jax.Array:
    r"""
    Broadcast ``(nlo + nhi, nf)`` :math:`t`-grid for the USR run.

    A pure log grid starves the :math:`t \lesssim 1` region and leaves the
    integral well above the converged value; this linear-low-:math:`t` +
    geometric-high-:math:`t` grid (same total point count) reproduces the
    converged spectrum to sub-percent — it is the *spacing*, not the count,
    that matters.
    """
    t = jnp.concatenate(
        [jnp.linspace(1e-5, 0.999, nlo), jnp.geomspace(1.0, 1e3, nhi)]
    )
    return jnp.repeat(jnp.expand_dims(t, axis=-1), nf, axis=-1)


class SIGWAYSingleFieldUSR(SIGWAYTemplate):
    r"""
    Ultra-slow-roll single-field SIGW spectrum (Mukhanov-Sasaki), via SIGWAY.

    The paper's quasi-inflection-point potential is fixed; the free parameters
    are its coefficients ``(a, lam, v, nfac)``. For a *different* potential,
    build one with
    :func:`~gwb_templates.sigway_templates.base.sigway_template`, passing a
    :class:`sigway.perturbations.SingleFieldPerturbations` wrapping your own
    :class:`sigway.ms_solver.SingleFieldSolver`.

    Configuration (constructor)
    ---------------------------
    phi0, pi0, N_CMB_to_end
        Solver background settings (initial field value / velocity, e-folds
        from the CMB pivot to the end of inflation).
    k_solver
        Wavenumber grid (:math:`\mathrm{s}^{-1}`) on which the MS equation is
        solved before interpolation.
    s, f
        SIGWAY quadrature / internal-frequency grids.
    """

    jittable: ClassVar[bool] = False
    differentiation_backend: ClassVar[str] = "finite_difference"

    DEFAULT_F: ClassVar[tuple[float, float, int]] = (1e-5, 1.0, 200)
    DEFAULT_NS: ClassVar[int] = 10

    def __init__(
        self,
        *,
        phi0: float = 3.0,
        pi0: float = 0.0,
        N_CMB_to_end: float = 58.0,
        k_solver: Any = None,
        s: Any = None,
        f: Any = None,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        self._phi0 = phi0
        self._pi0 = pi0
        self._N_CMB_to_end = N_CMB_to_end
        self._k_solver = (
            jnp.geomspace(1e-5, 10.0, 200) if k_solver is None else k_solver
        )
        self._s = jnp.linspace(0.0, 1.0, self.DEFAULT_NS) if s is None else s
        lo, hi, n = self.DEFAULT_F
        self._f = jnp.geomspace(lo, hi, n) if f is None else f

        default_labels = {
            "a": r"$a$",
            "lam": r"$\lambda$",
            "v": r"$v$",
            "nfac": r"$n_{\mathrm{fac}}$",
        }
        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Single-field USR SIGW (Mukhanov-Sasaki)"
            ),
            parameter_labels=(
                parameter_labels if parameter_labels is not None else default_labels
            ),
            prior_by_param=prior_by_param,
        )

    def build_model(self) -> OmegaGW:
        solver = SingleFieldSolver(
            usr_potential,
            phi0=self._phi0,
            pi0=self._pi0,
            N_CMB_to_end=self._N_CMB_to_end,
            k=jnp.asarray(self._k_solver),
        )
        perturbations = SingleFieldPerturbations(solver, ("a", "lam", "v", "nfac"))
        return OmegaGW(
            perturbations,
            RadiationKernel(),
            s=self._s,
            t=usr_t_grid(nf=len(self._f)),
            f=self._f,
            upsample=True,
        )
