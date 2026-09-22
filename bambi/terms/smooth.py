import warnings

import formulae
import numpy as np

from bambi.priors import Prior
from bambi.terms.base import BaseTerm


class SmoothTerm(BaseTerm):
    """Term for a penalized smooth with a hierarchical curvature prior.

    Notes
    -----
    For grouped smooths, numeric curvature prior arguments can be scalars,
    vectors of length `groups_n`, or arrays broadcastable to `(groups_n, curvature_n)`.
    Vectors specify values by group and are treated as columns.
    To specify values by curvature component, use an explicit row array of shape `(1, curvature_n)`.
    These conventions also apply when the curvature coefficients themselves are fixed.
    Without `by`, numeric vectors specify values by curvature component.
    """

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.data = term.data
        self.prefix = prefix

        if self.components[0].call.stateful_transform.by_levels is not None:
            variable = self.components[0].call.kwargs["by"]
            if not isinstance(variable, formulae.terms.call_resolver.LazyVariable):
                raise ValueError("Smooth 'by' must identify one categorical variable.")

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def kind(self):
        return self.term.kind

    @property
    def shape(self):
        return self.data.shape

    @property
    def levels(self):
        return None

    @property
    def categorical(self):
        return False

    @property
    def basis(self):
        return self.components[0].call.stateful_transform.__transform_name__

    @property
    def transform(self):
        return self.components[0].call.stateful_transform

    @property
    def by_levels(self):
        return self.transform.by_levels

    @property
    def by_name(self):
        if self.by_levels is None:
            return None
        return self.components[0].call.kwargs["by"].name

    @property
    def shared(self):
        return self.transform.shared

    @property
    def basis_dimension(self):
        if self.by_levels is None:
            return self.shape[1]
        return self.transform.basis_dimension

    @property
    def null_space_dimension(self):
        return self.components[0].call.stateful_transform.null_space_dimension

    @property
    def unpenalized_prior_keys(self):
        """Prior blocks for the null-space coefficients retained after centering.

        Bambi smooth transforms name their unpenalized coefficients in basis order.
        Formulae's centering constraint removes leading null-space directions,
        so the retained directions are the final `null_space_dimension` entries.
        """
        size = self.null_space_dimension
        if size == 0:
            return ()
        return self.transform.unpenalized_prior_keys[-size:]

    @property
    def has_intercept(self):
        return "constant" in self.unpenalized_prior_keys

    @property
    def prior_keys(self):
        return [*self.unpenalized_prior_keys, "curvature"]

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is None:
            self._prior = None
            return

        if not isinstance(value, dict):
            raise ValueError("Smooth priors must be a dictionary.")

        if set(value) != set(self.prior_keys):
            raise ValueError(
                f"Smooth priors must specify exactly the keys {self.prior_keys}. "
                f"Received {sorted(value)}."
            )

        for name, prior in value.items():
            if not isinstance(prior, Prior):
                if np.asarray(prior).dtype.kind not in "biuf":
                    raise ValueError(
                        "Each smooth prior block must be a Prior or a numeric constant."
                    )

                warnings.warn(
                    f"The '{name}' smooth component is usually modeled as a random variable.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if prior.name == "Normal" and not {"mu", "sigma"}.issubset(prior.args):
                raise ValueError("Normal smooth priors require 'mu' and 'sigma'.")

        constant_prior = value.get("constant")
        if isinstance(constant_prior, Prior) and any(
            isinstance(p, Prior) for p in constant_prior.args.values()
        ):
            raise ValueError(
                "The 'constant' smooth prior should not have any random variable arguments."
            )

        linear_prior = value.get("linear")
        if isinstance(linear_prior, Prior) and any(
            isinstance(p, Prior) for p in linear_prior.args.values()
        ):
            raise ValueError(
                "The 'linear' smooth prior should not have any random variable arguments."
            )

        curvature_prior = value["curvature"]
        if not isinstance(curvature_prior, Prior):
            warnings.warn(
                "The curvature prior is usually modeled as a random variable.",
                UserWarning,
                stacklevel=2,
            )
        elif curvature_prior.name == "Normal":
            if not np.asarray(curvature_prior.args["mu"]).dtype.kind in "biuf":
                raise ValueError("The mean of the curvature prior must be a numeric constant.")

            if not isinstance(curvature_prior.args["sigma"], Prior):
                warnings.warn(
                    "The scale of the curvature prior is usually modeled as a random variable.",
                    UserWarning,
                    stacklevel=2,
                )

        # Match the basis columns regardless of the supplied dictionary's insertion order.
        self._prior = {
            name: value[name] for name in ("constant", "linear", "curvature") if name in value
        }

    def __str__(self):
        return self.make_str()
