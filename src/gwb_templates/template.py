r"""
Abstract base class hierarchy for GW background template spectra.

Three layers:

* :class:`Template` — minimal ABC. Defines identity (``model_type`` /
  ``model_name`` / ``model_label`` / ``model_id``), parameter bookkeeping
  (names, labels, priors), the abstract computational entry point
  :meth:`omega_gw_h2`, and the family of derived quantities (gradients,
  Hessian, frequency derivatives, mixed derivative). Subclass registration
  and signature validation happen via ``__init_subclass__``.
* :class:`AnalyticTemplate` — for templates whose ``omega_gw_h2`` is a
  pure JAX computation. Derivatives go through ``jax`` autodiff.
* :class:`NumericalTemplate` — for templates backed by a numerical solver
  (e.g. SIGWAY). Carries a ``setup()`` lifecycle and a ``context`` payload,
  defaults to finite-difference derivatives, and exposes a
  :meth:`register_custom_derivatives` hook for templates that *can* provide
  hand-rolled JVP/VJP rules.

Design notes:

* Templates carry only *static config*. Hot-path computation is done via
  :meth:`_omega_from_parameter_vector`, which calls :meth:`omega_gw_h2`
  with parameters spread as positional arguments — a JAX-friendly shape
  that callers can wrap in :func:`jax.jit` themselves. No auto-jit is
  performed at construction.
* ``try/except`` around autodiff is gone. The differentiation backend is
  declared per-class via :attr:`Template.differentiation_backend` and is
  the single source of truth.
* Subclasses are registered in :attr:`Template._registry` and can be
  instantiated by name via :meth:`Template.from_name`. Instance
  ``model_name`` collisions are either auto-suffixed (when the user
  didn't pass an explicit name) or rejected (when they did).
"""

# Global imports
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np

from gwb_templates.utils import (
    finite_difference_d2f2,
    finite_difference_d2f_dtheta,
    finite_difference_df,
    finite_difference_grad_theta,
    finite_difference_hess_theta,
    gradient_autodiff,
    hessian_autodiff,
)

Array: TypeAlias = jax.Array
AnyArray: TypeAlias = jax.Array | np.ndarray
ArrayLike: TypeAlias = jtp.ArrayLike

DifferentiationBackend = Literal["autodiff", "finite_difference"]

# Change jax config to use double precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Base ABC
# =============================================================================


class Template(ABC):
    r"""
    Abstract base class for GW background template spectra.

    Concrete templates should inherit from :class:`AnalyticTemplate` (pure
    JAX implementation, autodiff-friendly) or :class:`NumericalTemplate`
    (numerical solver under the hood, finite-difference by default), not
    from :class:`Template` directly.

    Subclasses are expected to implement :meth:`omega_gw_h2` with a signature
    of the form::

        omega_gw_h2(self, frequency, param1, param2, ..., kwarg1=..., ...)

    Required positional parameters after ``frequency`` are inferred as the
    model's parameter names. Optional keyword arguments are treated as
    template-level configuration, not free parameters.
    """

    # ── Class-level configuration. Override in subclasses where appropriate. ──

    #: Whether ``omega_gw_h2`` is safe to wrap in :func:`jax.jit`. Analytic
    #: templates should leave this ``True``; numerical templates with
    #: non-JAX internals should set it to ``False``.
    jittable: ClassVar[bool] = True

    #: How parameter / frequency derivatives are computed. ``"autodiff"``
    #: uses :mod:`jax`; ``"finite_difference"`` uses the helpers in
    #: :mod:`gwb_templates.utils`.
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    #: BibTeX entries the template originates from. Subclasses should
    #: override with a tuple of one or more raw BibTeX strings (use a
    #: triple-quoted raw string per entry). Used by :meth:`get_bibtex`.
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    # ── Registries (class-level state) ──

    #: Concrete subclass registry, keyed by class name. Populated in
    #: :meth:`__init_subclass__`.
    _registry: ClassVar[dict[str, type["Template"]]] = {}

    # ------------------------------------------------------------------
    # Subclass registration & validation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register concrete subclasses and validate their ``omega_gw_h2``
        signature shape. Abstract intermediate classes (e.g.
        :class:`AnalyticTemplate`, :class:`NumericalTemplate`) are skipped.
        """
        super().__init_subclass__(**kwargs)

        # Always validate the signature shape, even for abstract intermediates,
        # so mistakes surface as early as possible.
        cls._validate_omega_signature()

        if inspect.isabstract(cls):
            return

        # Overwrite on re-definition. This is the common notebook case
        # (re-running the cell that defines a template class) and almost
        # always reflects "same logical template, reloaded". Cross-file
        # accidental collisions are rare enough that silently winning
        # beats raising in notebooks.
        Template._registry[cls.__name__] = cls

    @classmethod
    def _validate_omega_signature(cls) -> None:
        """
        Check that ``omega_gw_h2`` looks like ``(self, frequency, ...)``.
        """
        signature = inspect.signature(cls.omega_gw_h2)
        params = list(signature.parameters.values())
        if len(params) < 2:
            raise TypeError(
                f"{cls.__name__}.omega_gw_h2 must accept at least (self, frequency); "
                f"got signature {signature}."
            )
        if params[0].name != "self":
            raise TypeError(
                f"{cls.__name__}.omega_gw_h2 first parameter must be 'self'; "
                f"got {params[0].name!r}."
            )

    @classmethod
    def from_name(cls, name: str, *args: Any, **kwargs: Any) -> "Template":
        """
        Instantiate a registered concrete subclass by class name.

        Args:
            name: Concrete subclass name as registered in
                :attr:`Template._registry` (typically ``cls.__name__``).
            *args: Forwarded to the subclass constructor.
            **kwargs: Forwarded to the subclass constructor.

        Returns:
            New instance of the requested subclass.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        try:
            klass = cls._registry[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown Template subclass {name!r}. "
                f"Registered: {sorted(cls._registry)}."
            ) from e
        return klass(*args, **kwargs)

    @classmethod
    def registered_templates(cls) -> Mapping[str, type["Template"]]:
        """Return a read-only view of the concrete-subclass registry."""
        return MappingProxyType(Template._registry)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str | None = None,
        model_type: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize a GW template wrapper.

        Args:
            model_name: Optional runtime identifier for *this instance*. If
                omitted, defaults to ``model_type`` (i.e. the class name).
                No uniqueness check is performed — if you intend to use
                multiple instances of the same template simultaneously
                (e.g. low-freq vs. high-freq fits), pass ``model_name``
                explicitly so you can tell them apart in plots / logs.
            model_type: Optional model family / type label. Defaults to the
                class name.
            model_label: Optional display label (e.g. LaTeX-formatted) for
                plotting. Defaults to ``model_name``.
            parameter_labels: Optional sparse override map from parameter
                name to display label. Unspecified parameters fall back to
                the parameter name itself. Unknown keys raise.
            prior_by_param: Optional mapping of parameter priors. Stored as
                an immutable view. Unknown keys raise.

        Raises:
            ValueError: If ``parameter_labels`` or ``prior_by_param``
                contain keys not in ``parameter_names``.
        """
        self.parameter_names: tuple[str, ...] = tuple(
            self._infer_parameter_names_from_signature()
        )

        # Identity
        self.model_type: str = (
            model_type if model_type is not None else self.__class__.__name__
        )
        self.model_name: str = (
            model_name if model_name is not None else self.model_type
        )
        self.model_label: str = (
            model_label if model_label is not None else self.model_name
        )

        # Sparse labels with defaults
        label_map: dict[str, str] = {name: name for name in self.parameter_names}
        if parameter_labels is not None:
            unknown = set(parameter_labels) - set(self.parameter_names)
            if unknown:
                raise ValueError(
                    f"parameter_labels has unknown keys: {sorted(unknown)}. "
                    f"Allowed: {sorted(self.parameter_names)}."
                )
            label_map.update(parameter_labels)
        self.parameter_labels: Mapping[str, str] = MappingProxyType(label_map)

        # Priors (immutable view to avoid cross-template mutation surprises)
        priors = dict(prior_by_param) if prior_by_param is not None else {}
        unknown_prior_keys = [n for n in priors if n not in self.parameter_names]
        if unknown_prior_keys:
            raise ValueError(
                f"prior_by_param has unknown keys: {sorted(unknown_prior_keys)}. "
                f"Allowed: {sorted(self.parameter_names)}."
            )
        self.prior_by_param: Mapping[str, Any] = MappingProxyType(priors)

    # ------------------------------------------------------------------
    # Identity & dunder
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Stable identifier combining model type and instance name."""
        return f"{self.model_type}:{self.model_name}"

    @property
    def n_params(self) -> int:
        """Number of free parameters for this template."""
        return len(self.parameter_names)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} model_id={self.model_id!r} "
            f"params={list(self.parameter_names)}>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Template):
            return NotImplemented
        return (
            self.model_id == other.model_id
            and self.parameter_names == other.parameter_names
        )

    def __hash__(self) -> int:
        return hash((self.model_id, self.parameter_names))

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    @classmethod
    def get_bibtex(cls, *, joined: bool = True) -> str | tuple[str, ...]:
        """
        Return the BibTeX entries associated with this template.

        Subclasses populate the source list by setting
        :attr:`bibtex_entries` to a tuple of raw BibTeX strings (one per
        ``@article{...}`` block). The intended usage is to paste the
        result straight into a ``.bib`` file.

        Args:
            joined: If True (default), entries are concatenated into a
                single string separated by a blank line — ready to drop
                into a ``.bib`` file. If False, returns the tuple of
                individual entries so the caller can iterate.

        Returns:
            Either the joined BibTeX string or the raw tuple of entries.

        Raises:
            LookupError: If the subclass has not declared any
                ``bibtex_entries``.
        """
        entries = cls.bibtex_entries
        if not entries:
            raise LookupError(
                f"{cls.__name__} has no bibtex_entries declared. "
                f"Override the bibtex_entries ClassVar on the subclass."
            )
        if joined:
            return "\n\n".join(entry.strip() for entry in entries)
        return tuple(entry.strip() for entry in entries)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _infer_parameter_names_from_signature(self) -> list[str]:
        """
        Infer parameter names from required ``omega_gw_h2`` arguments
        following ``frequency``.

        A parameter is "required" if it is positional-only or
        positional-or-keyword with no default. Keyword-only arguments and
        arguments with defaults are treated as template configuration,
        not free parameters.
        """
        signature = inspect.signature(self.omega_gw_h2)
        signature_params = list(signature.parameters.values())

        if not signature_params:
            return []

        inferred: list[str] = []
        for parameter in signature_params[1:]:
            if (
                parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and parameter.default is inspect.Parameter.empty
            ):
                inferred.append(parameter.name)

        return inferred

    def _to_parameter_vector(
        self, parameters: ArrayLike | Mapping[str, Any]
    ) -> Array:
        """
        Coerce ``parameters`` to a 1-D vector in ``self.parameter_names``
        order. Accepts either a mapping (validated) or an array-like
        (last axis must match ``n_params`` if any are declared).
        """
        if isinstance(parameters, Mapping):
            missing = [n for n in self.parameter_names if n not in parameters]
            extra = [n for n in parameters if n not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    "Parameter mapping keys must match parameter_names. "
                    f"Missing={missing}, extra={extra}."
                )
            return jnp.asarray([parameters[n] for n in self.parameter_names])

        vector = jnp.asarray(parameters)
        if self.parameter_names and vector.shape[-1] != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters on last axis, "
                f"got shape {tuple(vector.shape)}."
            )
        return vector

    # ------------------------------------------------------------------
    # JAX-friendly entry point
    # ------------------------------------------------------------------

    def _omega_from_parameter_vector(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        """
        Evaluate ``omega_gw_h2`` from a parameter vector or mapping.

        Parameters are spread as positional arguments to keep the call
        traceable by ``jax`` transforms (no dict/kwargs plumbing on the
        hot path). This is the canonical entry point wrapped by autodiff
        and finite-difference helpers; callers that want to ``jax.jit``
        the template should wrap *this method* (or its public alias
        :meth:`omega_gw_h2_from_parameters`) rather than
        :meth:`omega_gw_h2` directly, since this one accepts a vector
        rather than spread parameters.
        """
        theta = self._to_parameter_vector(parameters)
        return self.omega_gw_h2(frequency, *tuple(theta), *args, **kwargs)

    def omega_gw_h2_from_parameters(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""
        Public vector/mapping entry point for :math:`\Omega_{\mathrm{GW}} h^2`.

        Use this when integrating with samplers / pipelines that hand
        parameters around as arrays or dicts. Wrap it in :func:`jax.jit`
        if you want caching across repeated calls (provided
        :attr:`jittable` is ``True``).
        """
        return self._omega_from_parameter_vector(
            frequency, parameters, *args, **kwargs
        )

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""
        Evaluate :math:`\Omega_{\mathrm{GW}} h^2`.

        Subclasses should declare model parameters as explicit positional
        arguments after ``frequency`` (no defaults), and any template-level
        configuration as keyword arguments with defaults. Example::

            def omega_gw_h2(self, frequency, log_amplitude, tilt, *, pivot=3e-3):
                ...

        Subclasses must accept the *whole* ``frequency`` array and decide
        internally how to vectorize across it. A leading batch dimension on
        parameters is not part of the contract for this entry point — use
        :func:`jax.vmap` over :meth:`omega_gw_h2_from_parameters` for
        batching across parameter sets.

        Args:
            frequency: Frequency value(s).
            *args: Required model parameters.
            **kwargs: Template configuration.

        Returns:
            Evaluated spectrum.
        """

    # ------------------------------------------------------------------
    # Derivatives — dispatch on declared backend, no try/except
    # ------------------------------------------------------------------

    def grad_theta_omega_gw_h2(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            return gradient_autodiff(
                self._omega_from_parameter_vector,
                frequency,
                theta,
                *args,
                **kwargs,
            )
        return finite_difference_grad_theta(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def hess_theta_omega_gw_h2(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate :math:`\partial^2(\Omega_{\mathrm{GW}} h^2)/\partial\theta^2`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            return hessian_autodiff(
                self._omega_from_parameter_vector,
                frequency,
                theta,
                *args,
                **kwargs,
            )
        return finite_difference_hess_theta(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def d_df_omega_gw_h2(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial f`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            freq = jnp.asarray(frequency)
            if freq.ndim == 0:
                return jax.grad(self._omega_from_parameter_vector, argnums=0)(
                    frequency, theta, *args, **kwargs
                )

            def _omega_scalar(ff: ArrayLike) -> Array:
                return self._omega_from_parameter_vector(ff, theta, *args, **kwargs)

            return jax.vmap(jax.grad(_omega_scalar))(freq)
        return finite_difference_df(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def d2_df2_omega_gw_h2(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate :math:`\partial^2(\Omega_{\mathrm{GW}} h^2)/\partial f^2`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            freq = jnp.asarray(frequency)
            if freq.ndim == 0:
                return jax.grad(
                    jax.grad(self._omega_from_parameter_vector, argnums=0),
                    argnums=0,
                )(frequency, theta, *args, **kwargs)

            def _omega_scalar(ff: ArrayLike) -> Array:
                return self._omega_from_parameter_vector(ff, theta, *args, **kwargs)

            return jax.vmap(jax.grad(jax.grad(_omega_scalar)))(freq)
        return finite_difference_d2f2(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def d2_df_dtheta_omega_gw_h2(
        self,
        frequency: ArrayLike,
        parameters: ArrayLike | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        r"""
        Evaluate the mixed derivative
        :math:`\partial^2(\Omega_{\mathrm{GW}} h^2)/(\partial f\,\partial\theta)`.
        """
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            freq = jnp.asarray(frequency)
            mixed_fn = jax.jacfwd(
                jax.grad(self._omega_from_parameter_vector, argnums=0),
                argnums=1,
            )
            if freq.ndim == 0:
                return mixed_fn(frequency, theta, *args, **kwargs)
            return jax.vmap(lambda ff: mixed_fn(ff, theta, *args, **kwargs))(freq)
        return finite_difference_d2f_dtheta(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )


# =============================================================================
# Analytic templates
# =============================================================================


class AnalyticTemplate(Template):
    """
    Abstract base for templates whose ``omega_gw_h2`` is implemented as a
    pure JAX computation (closed-form expressions, no external solvers).

    Defaults inherited from :class:`Template` already match this use case
    (``jittable=True``, ``differentiation_backend="autodiff"``); this class
    exists primarily as a typing / discoverability anchor and to make the
    "analytic vs. numerical" distinction explicit in user code.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        *args: Any,
        **kwargs: Any,
    ) -> Array:  # pragma: no cover - re-declared for abstractness
        ...


# =============================================================================
# Numerical templates
# =============================================================================


class NumericalTemplate(Template):
    """
    Abstract base for templates backed by a numerical framework (e.g.
    SIGWAY for SIGWs). These templates typically carry internal state
    (interpolation tables, kernel grids, solver configurations) and may
    not be safely traceable by :func:`jax.jit` out of the box.

    Lifecycle:

    1. ``__init__`` calls :meth:`setup` once, which subclasses use to
       build heavy state (grids, interpolators).
    2. ``__init__`` then calls :meth:`register_custom_derivatives`, which
       subclasses may override to wire up ``jax.custom_jvp`` / VJP rules.
       The default is a no-op.

    Subclasses can selectively re-enable JIT / autodiff by overriding
    :attr:`jittable` and :attr:`differentiation_backend`.

    Extra inputs that are *not* free parameters (cosmology, transfer
    functions, integrator tolerances) belong on :attr:`context`, set in
    :meth:`setup`.
    """

    jittable: ClassVar[bool] = False
    differentiation_backend: ClassVar[DifferentiationBackend] = "finite_difference"

    def __init__(
        self,
        model_name: str | None = None,
        model_type: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
        context: Any = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            model_label=model_label,
            parameter_labels=parameter_labels,
            prior_by_param=prior_by_param,
        )
        #: Free-form payload for solver configuration, transfer functions,
        #: cosmology, etc. Subclasses may overwrite this in :meth:`setup`.
        self.context: Any = context

        self.setup()
        self.register_custom_derivatives()

    def setup(self) -> None:
        """
        Build heavy internal state — interpolation grids, kernel tables,
        cached numerical artifacts. Called once at the end of
        ``__init__``. Default implementation is a no-op.

        Subclasses overriding this should keep the work idempotent so that
        an explicit re-call (e.g. after mutating :attr:`context`) is safe.
        """

    def register_custom_derivatives(self) -> None:
        """
        Optional hook to attach hand-rolled derivative rules
        (``jax.custom_jvp`` / ``jax.custom_vjp``) to
        :meth:`omega_gw_h2` or :meth:`_omega_from_parameter_vector`.

        Called once at the end of ``__init__`` after :meth:`setup`.
        Default implementation is a no-op; subclasses that *do* provide
        custom rules should typically also set
        ``differentiation_backend = "autodiff"`` so the rules are picked
        up by the derivative helpers.
        """

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: ArrayLike,
        *args: Any,
        **kwargs: Any,
    ) -> Array:  # pragma: no cover - re-declared for abstractness
        ...


def get_template_from_registry(name: str, *args: Any, **kwargs: Any) -> Template:
    """
    Module-level convenience wrapper around :meth:`Template.from_name`.

    Instantiate a registered :class:`Template` subclass by class name. For
    the registry to be populated, the modules defining each template must
    have been imported at least once — the package ``__init__`` does this
    eagerly, so a plain ``import gwb_templates`` is enough.

    Example::

        from gwb_templates import get_template_from_registry
        pl = get_template_from_registry("PowerLaw", pivot=1e-2)
    """
    return Template.from_name(name, *args, **kwargs)