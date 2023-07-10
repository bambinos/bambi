import formulae as fm
import numpy as np
import xarray as xr

from bambi.defaults import get_default_prior
from bambi.families import univariate, multivariate
from bambi.priors import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm, ResponseTerm
from bambi.utils import get_aliased_name, is_hsgp_term


class ConstantComponent:
    """Constant model components

    This is a component for a target parameter that has no predictors. This could be seen as
    an intercept-only model for that parameter. For example, tis is the case for sigma when
    a non-distributional Normal linear regression model is used.

    Parameters
    ----------
    name : str
        The name of the component. For example "sigma", "alpha", or "kappa".
    priors : bambi.priors.Prior
        The prior distribution for the parameter
    spec : bambi.Model
        The Bambi model
    """

    def __init__(self, name, prior, spec):
        self.name = name
        self.prior = prior
        self.spec = spec
        self.alias = None

    def update_priors(self, value):
        self.prior = value


class DistributionalComponent:
    """Distributional model components

    Parameters
    ----------
    design : formulae.DesignMatrices
        The object with all the required design matrices and information about the model terms.
    priors : dict
        A dictionary where keys are term names and values are their priors.
    response_name : str
        The name of the response or target. If ``response_kind`` is ``"data"``, it's the name of
        the response variable. If ``response_kind`` is ``"param"`` it's the name of the parameter.
        An example of the first could be "Reaction" and an example of the latter could be "sigma"
        or "kappa".
    response_kind : str
        Specifies whether the distributional component models the response (``"data"``) or an
        auxiliary parameter (``"param"``). When ``"data"`` this is actually modeling the "parent"
        parameter of the family.
    spec : bambi.Model
        The Bambi model
    """

    def __init__(self, design, priors, response_name, response_kind, spec):
        self.terms = {}
        self.alias = None
        self.design = design
        self.response_name = response_name
        self.response_kind = response_kind
        self.spec = spec
        self.prefix = "" if response_kind == "data" else response_name

        if self.design.common:
            self.add_common_terms(priors)
            self.add_hsgp_terms(priors)

        if self.design.group:
            self.add_group_specific_terms(priors)

        if self.design.response:
            self.add_response_term()

    def add_common_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                continue
            prior = priors.pop(name, priors.get("common", None))
            if isinstance(prior, Prior):
                any_hyperprior = any(isinstance(x, Prior) for x in prior.args.values())
                if any_hyperprior:
                    raise ValueError(
                        f"Trying to set hyperprior on '{name}'. "
                        "Can't set a hyperprior on common effects."
                    )

            if term.kind == "offset":
                self.terms[name] = OffsetTerm(term, self.prefix)
            else:
                self.terms[name] = CommonTerm(term, prior, self.prefix)

    def add_group_specific_terms(self, priors):
        for name, term in self.design.group.terms.items():
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior, self.prefix)

    def add_hsgp_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                prior = priors.pop(name, None)
                self.terms[name] = HSGPTerm(term, prior, self.prefix)

    def add_response_term(self):
        """Add a response (or outcome/dependent) variable to the model."""
        response = self.design.response

        if hasattr(response.term.term.components[0], "reference"):
            reference = response.term.term.components[0].reference
        else:
            reference = None

        # NOTE: This is a historical feature.
        # I'm not sure how many family specific checks we should add in this type of places now
        if reference is not None and not isinstance(self.spec.family, univariate.Bernoulli):
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        if isinstance(self.spec.family, univariate.Bernoulli):
            if response.kind == "categoric" and response.levels is None and reference is None:
                raise ValueError("Categoric response must be binary for 'bernoulli' family.")
            if response.kind == "numeric" and not all(np.isin(response.design_matrix, [0, 1])):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

        self.terms[response.name] = ResponseTerm(response, self.spec.family)

    def build_priors(self):
        for term in self.terms.values():
            if isinstance(term, GroupSpecificTerm):
                kind = "group_specific"
            elif isinstance(term, CommonTerm) and term.kind == "intercept":
                kind = "intercept"
            elif hasattr(term, "kind") and term.kind == "offset":
                continue
            elif isinstance(term, HSGPTerm):
                if term.prior is None:
                    term.prior = get_default_prior("hsgp", cov_func=term.cov)
                continue
            else:
                kind = "common"
            term.prior = prepare_prior(term.prior, kind, self.spec.auto_scale)

    def update_priors(self, priors):
        """Update priors

        Parameters
        ----------
        priors : dict
            Names are terms, values a priors.
        """
        for name, value in priors.items():
            self.terms[name].prior = value

    def predict(
        self, idata, data=None, include_group_specific=True, hsgp_dict=None, sample_new_groups=False
    ):
        linear_predictor = 0
        posterior = idata.posterior
        in_sample = data is None
        family = self.spec.family

        # Prepare dims objects
        response_name = get_aliased_name(self.spec.response_component.response_term)
        response_dim = response_name + "_obs"
        linear_predictor_dims = ("chain", "draw", response_dim)
        to_stack_dims = ("chain", "draw")
        design_matrix_dims = (response_dim, "__variables__")

        # These families drop a level in the response
        if isinstance(self.spec.family, (multivariate.Multinomial, univariate.Categorical)):
            response_levels_dim = response_name + "_reduced_dim"
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)
        # These families DON'T drop any level in the response
        elif isinstance(self.spec.family, multivariate.MultivariateFamily):
            response_levels_dim = response_name + "_dim"
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)

        if self.design.common:
            linear_predictor += self.predict_common(
                posterior, data, in_sample, to_stack_dims, design_matrix_dims, hsgp_dict
            )

        if self.design.group and include_group_specific:
            linear_predictor += self.predict_group_specific(
                posterior, data, in_sample, to_stack_dims, design_matrix_dims, sample_new_groups
            )

        # Sort dimensions
        linear_predictor = linear_predictor.transpose(*linear_predictor_dims)

        # Add coordinates for the observation number
        obs_n = len(linear_predictor[response_dim])
        linear_predictor = linear_predictor.assign_coords({response_dim: list(range(obs_n))})

        # Handle more special cases
        if hasattr(family, "transform_linear_predictor"):
            linear_predictor = family.transform_linear_predictor(
                self.spec, linear_predictor, posterior
            )

        if self.response_kind == "data":
            linkinv = family.link[family.likelihood.parent].linkinv
        else:
            linkinv = family.link[self.response_name].linkinv

        invlink_kwargs = getattr(family, "INVLINK_KWARGS", {})
        response = xr.apply_ufunc(linkinv, linear_predictor, kwargs=invlink_kwargs)

        if hasattr(family, "transform_coords"):
            response = family.transform_coords(self.spec, response)

        if hasattr(family, "transform_mean"):
            response = family.transform_mean(self.spec, response)

        return response

    def predict_common(
        self, posterior, data, in_sample, to_stack_dims, design_matrix_dims, hsgp_dict
    ):
        x_offsets = []
        linear_predictor = 0
        response_dim = design_matrix_dims[0]

        if in_sample:
            X = self.design.common.design_matrix
        else:
            X = self.design.common.evaluate_new_data(data).design_matrix

        # Add offset columns to their own design matrix and remove then from common matrix
        for term in self.offset_terms:
            term_slice = self.design.common.slices[term]
            x_offsets.append(X[:, term_slice])
            X = np.delete(X, term_slice, axis=1)

        # Add HSGP components contribution to the linear predictor
        for term_name, term in self.hsgp_terms.items():
            # Extract data for the HSGP component from the design matrix
            term_slice = self.design.common.slices[term_name]
            x_slice = X[:, term_slice]
            X = np.delete(X, term_slice, axis=1)
            term_aliased_name = get_aliased_name(term)
            hsgp_to_stack_dims = (f"{term_aliased_name}_weights_dim",)

            # Data may be scaled so the maximum Euclidean distance between two points is 1
            if term.scale_predictors:
                maximum_distance = term.maximum_distance
            else:
                maximum_distance = 1

            # NOTE:
            # The approach here differs from the one in the PyMC implementation.
            # Here we have a single dot product with many zeros, while there we have many
            # smaller dot products.
            # It is subject to change here, but I don't want to mess up dims and coords.
            if term.by_levels is not None:
                by_values = x_slice[:, -1].astype(int)
                x_slice = x_slice[:, :-1]
                x_slice_centered = (x_slice.data - term.mean[by_values]) / maximum_distance
                phi_list = []
                for i, level in enumerate(term.by_levels):
                    phi = term.hsgp[level].prior_linearized(x_slice_centered)[0].eval()
                    phi[by_values != i] = 0
                    phi_list.append(phi)
                phi = np.column_stack(phi_list)
                hsgp_to_stack_dims = (f"{term_aliased_name}_by",) + hsgp_to_stack_dims
            else:
                x_slice_centered = (x_slice - term.mean) / maximum_distance
                phi = term.hsgp.prior_linearized(x_slice_centered)[0].eval()

            # Convert 'phi' to xarray.DataArray for easier math operations
            # Notice the extra '_' in the dim name for the weights
            phi = xr.DataArray(phi, dims=(response_dim, f"{term_aliased_name}__weights_dim"))
            weights = posterior[f"{term_aliased_name}_weights"]
            weights = weights.stack({f"{term_aliased_name}__weights_dim": hsgp_to_stack_dims})

            # Compute contribution and add it to the linear predictor
            hsgp_contribution = xr.dot(phi, weights)

            # Store the contribution so it can be added later to the posterior Dataset
            hsgp_dict[term_name] = hsgp_contribution

            # Add contribution to the linear predictor
            linear_predictor += hsgp_contribution

        if self.common_terms or self.intercept_term:
            # Create DataArray
            X_terms = [get_aliased_name(term) for term in self.common_terms.values()]
            if self.intercept_term:
                X_terms.insert(0, get_aliased_name(self.intercept_term))
            b = posterior[X_terms].to_stacked_array("__variables__", to_stack_dims)

            # Add contribution due to the common terms
            X = xr.DataArray(X, dims=design_matrix_dims)
            linear_predictor += xr.dot(X, b)

        # If model contains offsets, add them directly to the linear predictor
        if x_offsets:
            linear_predictor += np.column_stack(x_offsets).sum(axis=1)[:, np.newaxis, np.newaxis]

        return linear_predictor

    def predict_group_specific(
        self, posterior, data, in_sample, to_stack_dims, design_matrix_dims, sample_new_groups
    ):
        if in_sample:
            Z = self.design.group.design_matrix
            u = posterior
        else:
            # We temporarily allow for the evaluation of new groups
            fm_eval_unseen_categories_original = fm.config["EVAL_UNSEEN_CATEGORIES"]
            fm.config["EVAL_UNSEEN_CATEGORIES"] = "silent"
            group = self.design.group.evaluate_new_data(data)
            fm.config["EVAL_UNSEEN_CATEGORIES"] = fm_eval_unseen_categories_original

            Z = group.design_matrix

            factors_with_new_levels = group.factors_with_new_levels
            if factors_with_new_levels:
                if sample_new_groups is False:
                    raise ValueError(
                        f"There are new groups for the factors {factors_with_new_levels} and "
                        "'sample_new_groups' is False."
                    )
                u = self._construct_u_with_new_groups(
                    posterior, to_stack_dims, factors_with_new_levels
                )
            else:
                u = posterior

        # Construct "u"
        # Previously, we used to use `.to_stacked_array()`.
        # Turns out the MultiIndex it used had it components sorted alphabetically, which is NOT
        # how columns are sorted in Z. This was problematic when the expression contained a
        # categoric variable.
        # I couldn't find how to do it with xarray, I think it's not possible.
        # So I'm doing it with NumPy.
        u_arrays = []
        for term in self.group_specific_terms.values():
            aliased_term_name = get_aliased_name(term)
            if term.alias:
                expr_dim = term.alias + "__expr_dim"
                factor_dim = term.alias + "__factor_dim"
            else:
                expr, factor = term.name.split("|")
                expr_dim = expr + "__expr_dim"
                factor_dim = factor + "__factor_dim"

            draws = u[aliased_term_name]

            if len(to_stack_dims) == 2:  # univariate response
                offset = 0
            elif len(to_stack_dims) == 3:  # multivariate response
                offset = 1

            if len(draws.coords) == 3 + offset:  # numeric
                u_columns = draws.to_numpy()
            elif len(draws.coords) == 4 + offset:  # categoric
                u_columns = draws.stack(column=(factor_dim, expr_dim)).to_numpy()

            u_arrays.append(u_columns)

        u_dims = ["chain", "draw", "__variables__"]
        if len(to_stack_dims) == 3:
            u_dims.insert(2, to_stack_dims[-1])

        u = np.concatenate(u_arrays, axis=-1)
        u = xr.DataArray(u, dims=u_dims)
        Z = xr.DataArray(Z, dims=design_matrix_dims)
        return xr.dot(Z, u)

    def _construct_u_with_new_groups(self, posterior, to_stack_dims, factors_with_new_levels):
        u_list = []
        names_list = []
        factor_idxs = {}
        draw_n = len(posterior.coords["draw"])
        chain_n = len(posterior.coords["chain"])
        rng = np.random.default_rng()
        seq_draw = np.arange(draw_n)
        seq_chain = np.arange(chain_n)

        if len(to_stack_dims) == 2:  # univariate response
            is_univariate = True
        elif len(to_stack_dims) == 3:  # multivariate response
            is_univariate = False

        for factor in factors_with_new_levels:
            term_names = self.group_specific_groups[factor]
            for name in term_names:
                term = self.group_specific_terms[name]
                aliased_term_name = get_aliased_name(term)
                names_list.append(aliased_term_name)

                if term.alias:
                    expr_dim = term.alias + "__expr_dim"
                    factor_dim = term.alias + "__factor_dim"
                else:
                    expr, factor = term.name.split("|")
                    expr_dim = expr + "__expr_dim"
                    factor_dim = factor + "__factor_dim"

                # For a given factor, we select the same draws to account for correlations.
                if factor in factor_idxs:
                    factor_sampled_idxs = factor_idxs[factor]
                else:
                    factor_levels = posterior.coords[factor_dim]
                    factor_sampled_idxs = rng.choice(np.arange(len(factor_levels)), size=draw_n)
                    factor_idxs[factor] = factor_sampled_idxs

                draws_original = posterior[aliased_term_name].to_numpy()

                if is_univariate:
                    # Numeric predictors
                    if draws_original.ndim == 3:
                        draws_new_group = draws_original[:, seq_draw, factor_sampled_idxs]
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }
                    # Categoric predictors
                    elif draws_original.ndim == 4:
                        draws_new_group = draws_original[:, seq_draw, :, factor_sampled_idxs]
                        # Don't know why, but the previous indexing swaps axes, we fix it
                        draws_new_group = np.swapaxes(draws_new_group, 0, 1)
                        expr_levels = posterior.coords[expr_dim].to_numpy()
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            expr_dim: expr_levels,
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }
                else:
                    response_dim = to_stack_dims[-1]
                    if draws_original.ndim == 4:
                        draws_new_group = draws_original[:, seq_draw, :, factor_sampled_idxs]
                        draws_new_group = np.swapaxes(draws_new_group, 0, 1)
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            response_dim: posterior.coords[response_dim].to_numpy(),
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }
                    elif draws_original.ndim == 5:
                        draws_new_group = draws_original[:, seq_draw, :, :, factor_sampled_idxs]
                        draws_new_group = np.swapaxes(draws_new_group, 0, 1)
                        expr_levels = posterior.coords[expr_dim].to_numpy()
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            response_dim: posterior.coords[response_dim].to_numpy(),
                            expr_dim: expr_levels,
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }

                draws_new_group = xr.DataArray(draws_new_group[..., np.newaxis], coords=coords)

                u_list.append(
                    xr.concat([posterior[aliased_term_name], draws_new_group], dim=factor_dim)
                )

        # Get a new xr.Dataset with the draws of the terms that have new groups
        u = xr.Dataset(dict(zip(names_list, u_list)))

        # Get a xr.Dataset with the draws of the terms that don't have new groups
        Z_terms = [
            get_aliased_name(term)
            for term in self.group_specific_terms.values()
            if get_aliased_name(term) not in names_list
        ]
        if Z_terms:
            u = xr.merge([u, posterior[Z_terms]])

        return u

    @property
    def group_specific_groups(self):
        groups = {}
        for term_name in self.group_specific_terms:
            factor = term_name.split("|")[1]
            if factor not in groups:
                groups[factor] = [term_name]
            else:
                groups[factor].append(term_name)
        return groups

    @property
    def intercept_term(self):
        """Return the intercept term in the model component."""
        for term in self.terms.values():
            if isinstance(term, CommonTerm) and term.kind == "intercept":
                return term
        return None

    @property
    def response_term(self):
        """Returns the response term in the model component"""
        for term in self.terms.values():
            if isinstance(term, ResponseTerm):
                return term
        return None

    @property
    def common_terms(self):
        """Return dict of all common effects in the model component."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if isinstance(v, CommonTerm) and not isinstance(v, OffsetTerm) and v.kind != "intercept"
        }

    @property
    def group_specific_terms(self):
        """Return dict of all group specific effects in model component."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, GroupSpecificTerm)}

    @property
    def offset_terms(self):
        """Return dict of all offset effects in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, OffsetTerm)}

    @property
    def hsgp_terms(self):
        """Return dict of all HSGP terms in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, HSGPTerm)}


def prepare_prior(prior, kind, auto_scale):
    """Helper function to correctly set default priors and auto scaling

    Parameters
    ----------
    prior : Prior or None
        The prior.
    kind : string
        Accepted values are: ``"intercept"``, ``"common"``, or ``"group_specific"``.
    auto_scale : bool
        Whether priors should be scaled or not. Defaults to ``True``.

    Returns
    -------
    prior : Prior
        The prior.
    """
    if prior is None:
        if auto_scale:
            prior = get_default_prior(kind)
        else:
            prior = get_default_prior(kind + "_flat")
    elif isinstance(prior, Prior):
        prior.auto_scale = False
    else:
        raise ValueError("'prior' must be instance of Prior or None.")
    return prior
