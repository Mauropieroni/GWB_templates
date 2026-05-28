# Global imports
from collections.abc import Callable, Mapping
from typing import Any, Optional, TypeAlias

import jax
import jax.typing as jtp
import numpy as np

from gwb_templates.utils import gradient_autodiff, hessian_autodiff

Array: TypeAlias = jax.Array
AnyArray: TypeAlias = jax.Array | np.ndarray
ArrayLike: TypeAlias = jtp.ArrayLike
TemplateFn: TypeAlias = Callable[..., Array]
IndexedDerivativeFn: TypeAlias = Callable[..., Array]


# Change jax config to use double precision
jax.config.update("jax_enable_x64", True)


class Template(object):
    """
    Container for a signal template and its first/second derivatives.

    """

    def __init__(
        self,
        model_name: str,
        template: TemplateFn,
        dtemplate: Optional[IndexedDerivativeFn] = None,
        d2template: Optional[IndexedDerivativeFn] = None,
        model_label: Optional[str] = None,
        parameter_names: Optional[Sequence[str]] = None,
        parameter_labels: Optional[Sequence[str]] = None,
        prior: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a signal model wrapper.

        Args:
            model_name: Human-readable model identifier.
            template: Base template function S(f, theta, ...).
            dtemplate: Optional indexed first-derivative function.
                If omitted, jacfwd(template) is used.
            model_label: Optional formatted label for the model.
            parameter_names: Optional ordered parameter names.
            parameter_labels: Optional formatted labels for plots.
            prior: Optional dictionary of parameter priors, with parameter names as
                keys and prior specifications as values.
        Returns:
            None.
        """

        self.model_name = model_name
        self.model_label = model_label if model_label is not None else model_name
        self.parameter_names = (
            list(parameter_names) if parameter_names is not None else []
        )
        self.parameter_labels = (
            list(parameter_labels) if parameter_labels is not None else []
        )

        self.template = template

        if dtemplate is not None:
            self.dtemplate = dtemplate
        else:
            self.dtemplate = lambda *args, **kwargs: gradient_autodiff(
                template, *args, **kwargs
            )

        if d2template is not None:
            self.d2template = d2template
        else:
            self.d2template = lambda *args, **kwargs: hessian_autodiff(
                template, *args, **kwargs
            )

        self.prior = prior if prior is not None else {}

    @property
    def Nparams(self) -> int:
        """
        Return the number of parameters in the model.

        Returns:
            Number of parameters.
        """

        return len(self.parameter_names)
