r"""
Abstract base class hierarchy for GW background template spectra.

Three layers:

* :class:`Template` — minimal ABC. Defines identity (``model_type`` / ``model_name`` /
  ``model_label`` / ``model_id``), parameter bookkeeping (names, labels, priors), the
  abstract computational entry point :meth:`omega_gw_h2`, and the family of derived
  quantities (gradients, Hessian, frequency derivatives, mixed derivative). Subclass
  registration and signature validation happen via ``__init_subclass__``.
* :class:`AnalyticTemplate` — for templates whose ``omega_gw_h2`` is a pure JAX
  computation. Derivatives can go through ``jax`` autodiff.
* :class:`NumericalTemplate` — for templates backed by a numerical solver (e.g. SIGWAY).
  Carries a ``setup()`` lifecycle and a ``context`` payload, derivatives default to
  finite-difference, and exposes a :meth:`register_custom_derivatives` hook for
  templates that *can* provide hand-rolled JVP/VJP rules.

Design notes:

* Templates carry only *static config*. The method :meth:`_omega_from_parameter_vector,
  calls :meth:`omega_gw_h2` (which spreads parameters as positional arguments), is a
  JAX-friendly method that takes parameters as an array and can be wrapped in
  func:`jax.jit`. No auto-jit is performed at construction.
* Subclasses are registered in :attr:`Template._registry` and can be instantiated by
  name via :meth:`Template.from_name`.
"""

# Global imports
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp


from gwb_templates.utils import (
    finite_difference_d2f2,
    finite_difference_d2f_dtheta,
    finite_difference_df,
    finite_difference_grad_theta,
    finite_difference_hess_theta,
    gradient_autodiff,
    hessian_autodiff,
)

DifferentiationBackend = Literal["autodiff", "finite_difference"]

# Change jax config to use double precision
jax.config.update("jax_enable_x64", True)


def _check_known_keys(
    candidate: Mapping[str, Any], allowed: tuple[str, ...], label: str
) -> None:
    """Raise ``ValueError`` if ``candidate`` has keys outside ``allowed``."""
    unknown = set(candidate) - set(allowed)
    if unknown:
        raise ValueError(
            f"{label} has unknown keys: {sorted(unknown)}. Allowed: {sorted(allowed)}."
        )


# =============================================================================
# Base ABC
# =============================================================================


class Template(ABC):
    r"""
    Abstract base class for GW background template spectra.

    Concrete templates should inherit from :class:`AnalyticTemplate` (pure JAX
    implementation, autodiff-friendly) or :class:`NumericalTemplate` (numerical solver
    under the hood, finite-difference by default), not from :class:`Template` directly.

    Subclasses should implement :meth:`omega_gw_h2` with signature:

        omega_gw_h2(self, frequency, param1, param2, ..., kwarg1=..., ...)

    Required positional parameters after ``frequency`` are inferred as the model's
    parameter names. Optional keyword arguments are treated as template-level
    configuration, not free parameters.
    """

    # ── Class-level configuration. Override in subclasses where appropriate. ──

    # Whether ``omega_gw_h2`` is safe to wrap in :func:`jax.jit`. Most Analytic
    # templates should leave this ``True``; numerical templates can set it to ``False``.
    jittable: ClassVar[bool] = True

    # How to handle derivatives. ``"autodiff"`` uses :mod:`jax`; ``"finite_difference"``
    # uses the helpers in :mod:`gwb_templates.utils`.
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    # BibTeX entries the template originates from. Subclasses should override this
    # with a tuple of one or more raw BibTeX strings. Used by :meth:`get_bibtex`.
    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    # ── Per-instance defaults ──
    # Subclasses may override any of these; unset ones fall back to the generic behavior
    # documented on ``__init__``. Each is resolved via ``self.DEFAULT_*`` — an actual
    # attribute lookup on the instance — rather than as a constructor default value, so
    # a subclass's override is picked up correctly even when the subclass's own
    # ``__init__`` never mentions these parameters and just forwards ``**kwargs``. An
    # empty string / empty mapping means "unset"; passing an explicit value always wins.

    #: Default ``model_name``. Empty means unset — falls back to ``model_type`` (the
    #: class name).
    DEFAULT_MODEL_NAME: ClassVar[str] = ""

    #: Default ``model_label``. Empty means unset — falls back to ``model_name``.
    DEFAULT_MODEL_LABEL: ClassVar[str] = ""

    #: Default per-parameter display labels. Sparse: a parameter absent here (or not
    #: overridden via the constructor) falls back to its own name.
    DEFAULT_PARAMETER_LABELS: ClassVar[Mapping[str, str]] = MappingProxyType({})

    #: Default per-parameter prior specification, used when the constructor's
    #: ``prior_by_param`` is empty.
    DEFAULT_PRIOR_BY_PARAM: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    # ── Registries (class-level state) ──

    # Concrete subclass registry, keyed by class name. Populated in
    # :meth:`__init_subclass__`.
    _registry: ClassVar[dict[str, type["Template"]]] = {}

    # ------------------------------------------------------------------
    # Subclass registration & validation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register concrete subclasses and validate their ``omega_gw_h2`` signature shape.
        Abstract intermediate classes (e.g, :class:`AnalyticTemplate`) are skipped.
        """
        super().__init_subclass__(**kwargs)

        # Always validate the signature shape, so mistakes surface as early as possible.
        cls._validate_omega_signature()

        if inspect.isabstract(cls):
            return

        # NB: TO BE CHECKED!!!!
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
            my_class = cls._registry[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown Template subclass {name!r}. "
                f"Registered: {sorted(cls._registry)}."
            ) from e
        return my_class(*args, **kwargs)

    @classmethod
    def registered_templates(cls) -> Mapping[str, type["Template"]]:
        """Return a read-only view of the concrete-subclass registry."""
        return MappingProxyType(Template._registry)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        model_name: str = "",
        model_type: str = "",
        model_label: str = "",
        parameter_labels: Mapping[str, str] = MappingProxyType({}),
        prior_by_param: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """
        Initialize a GW template wrapper.

        Every argument below, when left empty, falls back to the class's own
        ``DEFAULT_*`` :class:`ClassVar` (see the class body) before falling back further
        to the generic behavior described here — so a subclass only needs to *declare
        data* (``DEFAULT_MODEL_LABEL``, ``DEFAULT_PARAMETER_LABELS``, ...) instead of
        repeating this resolution logic in its own ``__init__``.

        Args:
            model_name: runtime identifier for *this instance*. Defaults to the class's
                ``DEFAULT_MODEL_NAME`` when declared, else to ``model_type``. No
                uniqueness check is performed — pass it explicitly if you intend to use
                multiple instances of the same template simultaneously (e.g. low-freq
                vs. high-freq fits) and need to tell them apart in plots / logs.
            model_type: model family / type label. Defaults to the class name.
            model_label: display label (e.g. LaTeX-formatted) for plotting. Defaults to
                the class's ``DEFAULT_MODEL_LABEL`` when declared, else to
                ``model_name``.
            parameter_labels: sparse override map from parameter name to display label.
                Unspecified parameters fall back to the class's
                ``DEFAULT_PARAMETER_LABELS`` and then to the parameter name itself.
            prior_by_param: mapping of parameter priors. Defaults to the class's
                ``DEFAULT_PRIOR_BY_PARAM`` when empty. Stored as an immutable view.

        Raises:
            ValueError: If ``parameter_labels`` or ``prior_by_param`` (or their class-
                level ``DEFAULT_*`` counterparts) contain keys not in
                ``parameter_names``.
        """
        self.parameter_names: tuple[str, ...] = tuple(
            self._infer_parameter_names_from_signature()
        )

        # Identity
        self.model_type: str = model_type or self.__class__.__name__
        self.model_name: str = model_name or self.DEFAULT_MODEL_NAME or self.model_type
        self.model_label: str = (
            model_label or self.DEFAULT_MODEL_LABEL or self.model_name
        )

        # Sparse labels: parameter name < class defaults < explicit override
        label_map: dict[str, str] = {name: name for name in self.parameter_names}
        _check_known_keys(
            self.DEFAULT_PARAMETER_LABELS,
            self.parameter_names,
            "DEFAULT_PARAMETER_LABELS",
        )
        label_map.update(self.DEFAULT_PARAMETER_LABELS)
        _check_known_keys(parameter_labels, self.parameter_names, "parameter_labels")
        label_map.update(parameter_labels)
        self.parameter_labels: Mapping[str, str] = MappingProxyType(label_map)

        # Priors (immutable view to avoid cross-template mutation surprises)
        priors: dict[str, Any] = dict(self.DEFAULT_PRIOR_BY_PARAM)
        _check_known_keys(
            self.DEFAULT_PRIOR_BY_PARAM, self.parameter_names, "DEFAULT_PRIOR_BY_PARAM"
        )
        priors.update(prior_by_param)
        _check_known_keys(prior_by_param, self.parameter_names, "prior_by_param")
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

        Subclasses populate the source list by setting :attr:`bibtex_entries` to a tuple
        of raw BibTeX strings (one per ``@article{...}`` block). The intended usage is
        to paste the result straight into a ``.bib`` file.

        Args:
            joined: If True (default), entries are concatenated into a single string
                separated by a blank line — ready to drop into a ``.bib`` file. If
                False, returns the tuple of individual entries.

        Returns:
            Either the joined BibTeX string or the raw tuple of entries.

        Raises:
            LookupError: If the subclass has not declared any ``bibtex_entries``.
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
        Infer parameter names from required ``omega_gw_h2`` arguments following
        ``frequency``.

        A parameter is "required" if it is positional-only or positional-or-keyword with
        no default. Keyword-only arguments and arguments with defaults are treated as
        template configuration, not free parameters.
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
        self, parameters: jax.Array | Mapping[str, Any]
    ) -> jax.Array:
        """
        Coerce ``parameters`` to a 1-D vector in ``self.parameter_names`` order. Accepts
        either a mapping or an array-like (last axis must match ``n_params``).
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

        if self.parameter_names and parameters.shape[-1] != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters on last axis, "
                f"got shape {tuple(parameters.shape)}."
            )
        return parameters

    # ------------------------------------------------------------------
    # JAX-friendly entry point
    # ------------------------------------------------------------------

    def _omega_from_parameter_vector(
        self,
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        """
        Evaluate ``omega_gw_h2`` from a parameter vector or mapping.

        Parameters are spread as positional arguments to keep the call traceable by
        ``jax`` transforms (no dict/kwargs plumbing on the hot path). This is the
        canonical entry point wrapped by autodiff and finite-difference helpers; callers
        that want to ``jax.jit`` the template should wrap *this method* (or its public
        alias :meth:`omega_gw_h2_from_parameters`) rather than :meth:`omega_gw_h2`
        directly, since this one accepts a vector rather than spread parameters.
        """
        theta = self._to_parameter_vector(parameters)
        return self.omega_gw_h2(frequency, *tuple(theta), *args, **kwargs)

    def omega_gw_h2_from_parameters(
        self,
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""
        Public vector/mapping entry point for :math:`\Omega_{\mathrm{GW}} h^2`.

        Use this when integrating with samplers / pipelines that hand parameters around
        as arrays or dicts. Wrap it in :func:`jax.jit` if you want caching across
        repeated calls (provided :attr:`jittable` is ``True``).
        """
        return self._omega_from_parameter_vector(frequency, parameters, *args, **kwargs)

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: jax.Array,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""
        Evaluate :math:`\Omega_{\mathrm{GW}} h^2`.

        Subclasses should declare model parameters as explicit positional arguments and
        any template-level configuration as keyword arguments with defaults. Example:

            def omega_gw_h2(self, frequency, log_amplitude, tilt, *, pivot=3e-3):
                ...

        Subclasses must accept the *whole* ``frequency`` array and decide internally how
        to vectorize across it. A leading batch dimension on parameters is not part of
        the contract for this entry point — use :func:`jax.vmap` over
        :meth:`omega_gw_h2_from_parameters` for batching across parameter sets.

        Args:
            frequency: Frequency value(s).
            *args: Required model parameters.
            **kwargs: Template configuration.

        Returns:
            Evaluated spectrum.
        """

    # ------------------------------------------------------------------
    # Derivatives — dispatch order is: analytical override → declared backend
    # ------------------------------------------------------------------

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""
        Optional hook for hand-rolled analytic
        :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`.

        Override on subclasses where the gradient has a clean closed form.
        :meth:`grad_theta_omega_gw_h2` detects the override and dispatches accordingly.

        Args:
            frequency: Frequency value(s).
            theta: Parameter vector (already coerced to ``parameter_names`` order).
            *args, **kwargs: Forwarded from the caller.

        Returns:
            Array of shape ``frequency.shape + (n_params,)`` — parameter axis last,
            matching the autodiff / FD backends.
        """
        raise NotImplementedError

    def grad_theta_omega_gw_h2(
        self,
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""Evaluate :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial\theta`."""
        theta = self._to_parameter_vector(parameters)
        # Prefer subclass-provided analytic gradient when overridden.
        if (
            type(self)._grad_theta_omega_gw_h2_analytical
            is not Template._grad_theta_omega_gw_h2_analytical
        ):
            return self._grad_theta_omega_gw_h2_analytical(
                frequency, theta, *args, **kwargs
            )
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
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
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
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""Evaluate :math:`\partial(\Omega_{\mathrm{GW}} h^2)/\partial f`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            if frequency.ndim == 0:
                return jax.grad(self._omega_from_parameter_vector, argnums=0)(
                    frequency, theta, *args, **kwargs
                )

            def _omega_scalar(ff: jax.Array) -> jax.Array:
                return self._omega_from_parameter_vector(ff, theta, *args, **kwargs)

            return jax.vmap(jax.grad(_omega_scalar))(frequency)
        return finite_difference_df(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def d2_df2_omega_gw_h2(
        self,
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""Evaluate :math:`\partial^2(\Omega_{\mathrm{GW}} h^2)/\partial f^2`."""
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            if frequency.ndim == 0:
                return jax.grad(
                    jax.grad(self._omega_from_parameter_vector, argnums=0),
                    argnums=0,
                )(frequency, theta, *args, **kwargs)

            def _omega_scalar(ff: jax.Array) -> jax.Array:
                return self._omega_from_parameter_vector(ff, theta, *args, **kwargs)

            return jax.vmap(jax.grad(jax.grad(_omega_scalar)))(frequency)
        return finite_difference_d2f2(
            self._omega_from_parameter_vector,
            frequency,
            theta,
            *args,
            **kwargs,
        )

    def d2_df_dtheta_omega_gw_h2(
        self,
        frequency: jax.Array,
        parameters: jax.Array | Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:
        r"""
        Evaluate the mixed derivative
        :math:`\partial^2(\Omega_{\mathrm{GW}} h^2)/(\partial f\,\partial\theta)`.
        """
        theta = self._to_parameter_vector(parameters)
        if self.differentiation_backend == "autodiff":
            mixed_fn = jax.jacfwd(
                jax.grad(self._omega_from_parameter_vector, argnums=0),
                argnums=1,
            )
            if frequency.ndim == 0:
                return mixed_fn(frequency, theta, *args, **kwargs)
            return jax.vmap(lambda ff: mixed_fn(ff, theta, *args, **kwargs))(frequency)
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
    Abstract base for templates whose ``omega_gw_h2`` is implemented as a pure JAX
    computation (closed-form expressions, no external solvers).

    Defaults inherited from :class:`Template` already match this use case
    (``jittable=True``, ``differentiation_backend="autodiff"``); this class exists
    primarily to make the "analytic vs. numerical" distinction explicit.
    """

    jittable: ClassVar[bool] = True
    differentiation_backend: ClassVar[DifferentiationBackend] = "autodiff"

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: jax.Array,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:  # pragma: no cover - re-declared for abstractness
        ...


# =============================================================================
# Numerical templates
# =============================================================================


class NumericalTemplate(Template):
    """
    Abstract base for templates backed by a numerical framework (e.g. SIGWAY). These
    templates typically carry internal state (interpolation tables, kernel grids, solver
    configurations) and may not be safely traceable by :func:`jax.jit` out of the box.

    Lifecycle:

    1. ``__init__`` calls :meth:`setup` once, which subclasses use to build heavy state
        (grids, interpolators).
    2. ``__init__`` then calls :meth:`register_custom_derivatives`, which subclasses may
        override to wire up ``jax.custom_jvp`` / VJP rules. The default is a no-op.

    Subclasses can selectively re-enable JIT / autodiff by overriding :attr:`jittable`
    and :attr:`differentiation_backend`.

    Extra inputs that are *not* free parameters (cosmology, transfer functions,
    integrator tolerances) belong on :attr:`context`, set in :meth:`setup`.
    """

    jittable: ClassVar[bool] = False
    differentiation_backend: ClassVar[DifferentiationBackend] = "finite_difference"

    def __init__(
        self,
        *,
        model_name: str = "",
        model_type: str = "",
        model_label: str = "",
        parameter_labels: Mapping[str, str] = MappingProxyType({}),
        prior_by_param: Mapping[str, Any] = MappingProxyType({}),
        context: Any = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            model_label=model_label,
            parameter_labels=parameter_labels,
            prior_by_param=prior_by_param,
        )
        # Free-form payload for solver configuration, transfer functions, cosmology,
        # etc. Subclasses may overwrite this in :meth:`setup`.
        self.context: Any = context

        self.setup()
        self.register_custom_derivatives()

    def setup(self) -> None:
        """
        Build heavy internal state — interpolation grids, kernel tables, cached
        numerical artifacts. Called once at the end of ``__init__``. Default
        implementation is a no-op.

        Subclasses overriding this should keep the work idempotent so that an explicit
        re-call (e.g. after mutating :attr:`context`) is safe.
        """

    def register_custom_derivatives(self) -> None:
        """
        Optional hook to attach custom derivative rules (``jax.custom_jvp`` /
        ``jax.custom_vjp``) to :meth:`omega_gw_h2` or
        :meth:`_omega_from_parameter_vector`.

        Called once at the end of ``__init__`` after :meth:`setup`. Default
        implementation is a no-op; subclasses that *do* provide custom rules should
        typically also set ``differentiation_backend = "autodiff"`` so the rules are
        picked up by the derivative helpers.
        """

    @abstractmethod
    def omega_gw_h2(
        self,
        frequency: jax.Array,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array:  # pragma: no cover - re-declared for abstractness
        ...


def get_template_from_registry(name: str, *args: Any, **kwargs: Any) -> Template:
    """
    Module-level convenience wrapper around :meth:`Template.from_name`.

    Instantiate a registered :class:`Template` subclass by class name. For the registry
    to be populated, each template module must have been imported at least once — the
    package ``__init__`` does this, so a plain ``import gwb_templates`` is enough.

    Example::

        from gwb_templates import get_template_from_registry
        pl = get_template_from_registry("PowerLaw", pivot=1e-2)
    """
    return Template.from_name(name, *args, **kwargs)
