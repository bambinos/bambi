import sparse

import formulae as fm
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import xarray as xr

from bambi.defaults import get_default_prior
from bambi.families import univariate, multivariate
from bambi.priors import Prior
from bambi.terms import (
    CommonTerm,
    GroupSpecificTerm,
    HSGPTerm,
    MonotonicGroupSpecificTerm,
    MonotonicInteractionTerm,
    MonotonicTerm,
    OffsetTerm,
    ResponseTerm,
)
from bambi.utils import (
    get_aliased_name,
    is_hsgp_term,
    is_monotonic_term,
    is_monotonic_interaction_term,
    is_monotonic_group_specific_term,
    as_dataset,
)


def _reencode_mo_for_new_data(transform, component, data):
    """Re-emit mo() codes for new data using the stateful transform's stored levels.

    ``component`` is the formulae Call object; ``data`` is the new pandas DataFrame.
    """
    var_name = next(iter(component.var_names))
    codes = transform(data[var_name])  # shape (n, 1), float64
    return np.asarray(codes).squeeze().astype("int64")


class ConstantComponent:
    """Constant model components

    Represents a parameter of the response distribution that is constant for all observations.
    It is equivalent to an intercept-only model for that parameter.
    For example, this describes sigma in a homoskedastic gaussian linear regression model.

    Parameters
    ----------
    name : str
        The name of the component. For example "sigma", "alpha", or "kappa".
    priors : bambi.Prior
        The prior distribution for the parameter.
    spec : bambi.Model
        The Bambi model.
    """

    def __init__(self, name, prior, spec):
        self.alias = None
        self.name = name
        self.prior = prior
        self.spec = spec

    def update_priors(self, value):
        self.prior = value


class DistributionalComponent:
    """Distributional model components

    Parameters
    ----------
    name : str
        The name of the component.
    design : formulae.DesignMatrices
        The object with all the required design matrices and information about the model terms.
    priors : dict
        A dictionary where keys are term names and values are their priors.
    spec : bambi.Model
        The Bambi model
    is_parent : bool
        Whether it's the parent parameter.
    """

    def __init__(self, name, design, priors, spec, is_parent):
        self.terms = {}
        self.alias = None
        self.name = name
        self.design = design
        self.spec = spec
        self.is_parent = is_parent
        self.prefix = "" if is_parent else self.name

        if self.design.common:
            self.add_common_terms(priors)
            self.add_hsgp_terms(priors)
            self.add_monotonic_terms(priors)

        if self.design.group:
            self.add_group_specific_terms(priors)

    def add_common_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                continue
            if is_monotonic_term(term):
                continue
            if is_monotonic_interaction_term(term):
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
            if is_monotonic_group_specific_term(term):
                self.terms[name] = MonotonicGroupSpecificTerm(term, prior, self.prefix)
            else:
                self.terms[name] = GroupSpecificTerm(term, prior, self.prefix)

    def add_hsgp_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                prior = priors.pop(name, None)
                self.terms[name] = HSGPTerm(term, prior, self.prefix)

    def add_monotonic_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_monotonic_term(term):
                prior = priors.pop(name, None)
                self.terms[name] = MonotonicTerm(term, prior, self.prefix)
            elif is_monotonic_interaction_term(term):
                prior = priors.pop(name, None)
                self.terms[name] = MonotonicInteractionTerm(term, prior, self.prefix)
        # Validate shared-id groups have compatible structure
        groups = {}
        for name, term in self.terms.items():
            if isinstance(term, MonotonicTerm) and term.id is not None:
                groups.setdefault(term.id, []).append((name, term))
        for id_name, members in groups.items():
            k_values = {t.K for _, t in members}
            if len(k_values) > 1:
                raise ValueError(
                    f"mo() terms sharing id={id_name!r} have inconsistent K values: "
                    f"{k_values}. All terms in a shared-simplex group must have the "
                    "same number of levels."
                )

    def build_priors(self):
        for term in self.terms.values():
            if isinstance(term, MonotonicGroupSpecificTerm):
                defaults = get_default_prior("monotonic_group_specific", D=term.D)
                user_prior = term.prior or {}
                for prior_obj in user_prior.values():
                    if isinstance(prior_obj, Prior):
                        prior_obj.auto_scale = False
                term.prior = {**defaults, **user_prior}
                continue
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
            elif isinstance(term, MonotonicTerm):
                defaults = get_default_prior("monotonic", D=term.D)
                user_prior = term.prior or {}
                # Mark any user-supplied priors as already-final so the auto-scaler skips them.
                for prior_obj in user_prior.values():
                    if isinstance(prior_obj, Prior):
                        prior_obj.auto_scale = False
                term.prior = {**defaults, **user_prior}
                continue
            elif isinstance(term, MonotonicInteractionTerm):
                # Only the slope prior is configurable on interaction terms; simplex
                # priors live on the standalone mo() (or shared via id=).
                user_prior = term.prior or {}
                for prior_obj in user_prior.values():
                    if isinstance(prior_obj, Prior):
                        prior_obj.auto_scale = False
                defaults = {"slope": Prior("Normal", mu=0.0, sigma=1.0)}
                term.prior = {**defaults, **user_prior}
                continue
            else:
                kind = "common"
            term.prior = prepare_prior(term.prior, kind, self.spec.auto_scale)

        # Unify simplex priors across shared-id groups: if any member has a
        # user-supplied simplex prior, apply it to the whole group; raise on conflicts.
        groups = {}
        for term in self.terms.values():
            if isinstance(term, MonotonicTerm) and term.id is not None:
                groups.setdefault(term.id, []).append(term)
        for id_name, members in groups.items():
            simplex_len = members[0].D
            user_simplex_priors = [
                t.prior["simplex"]
                for t in members
                if t.prior["simplex"].args["a"].shape != (simplex_len,)
                or not np.array_equal(t.prior["simplex"].args["a"], np.ones(simplex_len))
            ]
            if not user_simplex_priors:
                continue
            first = user_simplex_priors[0]
            for other in user_simplex_priors[1:]:
                if not np.array_equal(other.args["a"], first.args["a"]):
                    raise ValueError(
                        f"mo() terms sharing id={id_name!r} have conflicting simplex priors."
                    )
            for t in members:
                t.prior = {**t.prior, "simplex": first}

    def update_priors(self, priors):
        """Update priors.

        Parameters
        ----------
        priors : dict
            Names are terms, values are priors
        """
        for name, value in priors.items():
            self.terms[name].prior = value

    def predict(
        self,
        idata,
        data=None,
        include_group_specific=True,
        hsgp_dict=None,
        sample_new_groups=False,
        random_seed=None,
        monotonic_dict=None,
    ):
        linear_predictor = 0
        posterior = as_dataset(idata.posterior)
        in_sample = data is None

        # Prepare dims objects
        response_name = get_aliased_name(self.spec.response_component.term)
        response_dim = "__obs__"
        linear_predictor_dims = ("chain", "draw", response_dim)
        to_stack_dims = ("chain", "draw")
        design_matrix_dims = (response_dim, "__variables__")

        # These families drop a level in the response
        if isinstance(self.spec.family, (multivariate.Multinomial, univariate.Categorical)):
            response_levels_dim = response_name + "_reduced_dim"
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)

        # These families don't drop any level in the response
        elif isinstance(self.spec.family, multivariate.MultivariateFamily):
            response_levels_dim = response_name + "_dim"
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)

        if self.design.common:
            linear_predictor += self.predict_common(
                posterior,
                data,
                in_sample,
                to_stack_dims,
                design_matrix_dims,
                hsgp_dict,
                monotonic_dict,
            )

        if include_group_specific:
            linear_predictor += self.predict_monotonic_group_specific(
                posterior, data, in_sample, monotonic_dict
            )

        if self.design.group and include_group_specific and self.group_specific_terms:
            linear_predictor += self.predict_group_specific(
                posterior=posterior,
                data=data,
                in_sample=in_sample,
                to_stack_dims=to_stack_dims,
                design_matrix_dims=design_matrix_dims,
                sample_new_groups=sample_new_groups,
                random_seed=random_seed,
            )

        # Sort dimensions
        linear_predictor = linear_predictor.transpose(*linear_predictor_dims)

        # Add coordinates for the observation number
        obs_n = len(linear_predictor[response_dim])
        linear_predictor = linear_predictor.assign_coords({response_dim: list(range(obs_n))})

        # Handle more special cases
        if hasattr(self.spec.family, "transform_linear_predictor"):
            linear_predictor = self.spec.family.transform_linear_predictor(
                self.spec, linear_predictor, posterior
            )

        # NOTE: Handle VonMises family that internally uses 'identity' but needs angles
        if self.spec.family.name == "vonmises":
            # pylint: disable=unnecessary-lambda-assignment
            invlink = lambda x: np.angle(np.exp(1j * x))
        else:
            invlink = self.spec.family.link[self.name].linkinv
        invlink_kwargs = getattr(self.spec.family, "INVLINK_KWARGS", {})
        response = xr.apply_ufunc(invlink, linear_predictor, kwargs=invlink_kwargs)

        if hasattr(self.spec.family, "transform_coords"):
            response = self.spec.family.transform_coords(self.spec, response)

        if hasattr(self.spec.family, "transform_mean"):
            response = self.spec.family.transform_mean(self.spec, response)

        return response

    def predict_common(
        self,
        posterior,
        data,
        in_sample,
        to_stack_dims,
        design_matrix_dims,
        hsgp_dict,
        monotonic_dict=None,
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
        hsgp_slices = []
        for term_name, term in self.hsgp_terms.items():
            # Extract data for the HSGP component from the design matrix
            term_slice = self.design.common.slices[term_name]
            x_slice = X[:, term_slice]
            hsgp_slices.append(term_slice)
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

        # Add monotonic mo() contributions to the linear predictor
        monotonic_slices = []
        for term_name, term in self.monotonic_terms.items():
            term_slice = self.design.common.slices[term_name]
            codes = np.asarray(X[:, term_slice]).squeeze().astype("int64")
            monotonic_slices.append(term_slice)

            term_aliased_name = get_aliased_name(term)
            simplex = posterior[term.simplex_name]
            slope = posterior[f"{term_aliased_name}_b"]

            # cumulative sum along the simplex dim, prepended with 0 so codes==0 -> 0
            simplex_np = simplex.to_numpy()  # (chain, draw, D)
            cumsum = np.cumsum(simplex_np, axis=-1)
            zero = np.zeros(cumsum.shape[:-1] + (1,))
            cumsum = np.concatenate([zero, cumsum], axis=-1)  # (chain, draw, D+1)

            # Gather along the simplex axis with the observation codes
            contribution_np = term.D * np.take(cumsum, codes, axis=-1)
            contribution = xr.DataArray(
                contribution_np,
                dims=("chain", "draw", response_dim),
            )
            mo_contribution = slope * contribution
            if monotonic_dict is not None:
                monotonic_dict[term_name] = mo_contribution
            linear_predictor += mo_contribution

        # Monotonic interaction terms (e.g. mo(x):z, mo(x):mo(y))
        monotonic_interaction_slices = []
        for term_name, term in self.monotonic_interaction_terms.items():
            term_slice = self.design.common.slices[term_name]
            x_slice = np.asarray(X[:, term_slice], dtype=float)  # (n, k)
            monotonic_interaction_slices.append(term_slice)

            term_aliased_name = get_aliased_name(term)
            # Per-row product of raw mo() codes that formulae multiplied into the slice
            code_product = np.ones(x_slice.shape[0], dtype=float)
            for mc in term.mono_components:
                # Recompute codes from this row of new data: we already have them stored
                # on the component when in-sample; for new data, re-evaluate.
                if in_sample:
                    codes_m = mc["codes"]
                else:
                    # Re-encode through the stateful transform (which has the levels stored)
                    codes_m = _reencode_mo_for_new_data(
                        mc["transform"], term.term.components[mc["idx"]], data
                    )
                    mc["codes_new"] = codes_m  # cache for the contribution math below
                code_product *= codes_m

            # Recover the "other-factor" matrix: divide each row by code_product
            safe_codes = np.where(code_product == 0, 1.0, code_product)
            other_factor = x_slice / safe_codes[:, None]
            other_factor = np.where(
                code_product[:, None] == 0, 0.0, other_factor
            )  # zero rows where any code is 0

            # Compute prod_m D_m * cumsum(simplex_m)[codes_m] per draw
            mono_factor = None
            for mc in term.mono_components:
                tx_id = mc["id"]
                # The simplex's variable name follows the same rule as MonotonicTerm.simplex_name
                if tx_id is not None:
                    simplex_var = posterior[f"simplex_{tx_id}"]
                else:
                    # Standalone (no id) simplex inside an interaction. The interaction
                    # builder names it after the interaction term + component index.
                    simplex_var = posterior[f"{term_aliased_name}_simplex_{mc['idx']}"]
                simplex_np = simplex_var.to_numpy()  # (chain, draw, D)
                cumsum = np.cumsum(simplex_np, axis=-1)
                zero = np.zeros(cumsum.shape[:-1] + (1,))
                cumsum = np.concatenate([zero, cumsum], axis=-1)  # (chain, draw, D+1)
                codes_m = mc["codes"] if in_sample else mc["codes_new"]
                gathered = mc["D"] * np.take(cumsum, codes_m, axis=-1)  # (chain, draw, n)
                mono_factor = gathered if mono_factor is None else mono_factor * gathered

            # Slope is shape (k,)  -- per-column
            slope = posterior[f"{term_aliased_name}_b"]  # dims (chain, draw, [slope_dim])
            slope_np = slope.to_numpy()  # (chain, draw, k) or (chain, draw)
            if slope_np.ndim == 2:
                slope_np = slope_np[..., None]  # (chain, draw, 1)

            # contribution[chain, draw, n] = sum_k slope[k] * mono_factor[n] * other_factor[n, k]
            # = mono_factor[chain, draw, n] * sum_k slope[chain, draw, k] * other_factor[n, k]
            other_sum = np.einsum("nk,cdk->cdn", other_factor, slope_np)
            contribution_np = mono_factor * other_sum
            contribution = xr.DataArray(
                contribution_np,
                dims=("chain", "draw", response_dim),
            )
            linear_predictor += contribution
            if monotonic_dict is not None:
                monotonic_dict[term_name] = contribution

        # Remove columns of X that are associated with HSGP or monotonic contributions.
        # All the slices _must be_ deleted at the same time. Otherwise the slice objects don't
        # reflect the right columns of X at the time they're used
        drop_slices = hsgp_slices + monotonic_slices + monotonic_interaction_slices
        if drop_slices:
            X = np.delete(X, np.r_[tuple(drop_slices)], axis=1)

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

    def predict_monotonic_group_specific(self, posterior, data, in_sample, monotonic_dict):
        """Contribution of all ``(mo(x) | g)`` terms for in-sample or new data."""
        linear_predictor = 0
        response_dim = "__obs__"
        for term_name, term in self.monotonic_group_specific_terms.items():
            term_aliased_name = get_aliased_name(term)

            if in_sample:
                codes = term.codes
                group_index = term.group_index
            else:
                codes = _reencode_mo_for_new_data(
                    term.transform, term.term.expr.components[0], data
                )
                factor_name = next(iter(term.term.factor.var_names))
                stored_groups = list(term.groups)
                recoded = pd.Categorical(data[factor_name], categories=stored_groups)
                if (np.asarray(recoded.codes) == -1).any():
                    bad = pd.Series(data[factor_name])[recoded.codes == -1].unique()
                    raise ValueError(
                        f"'(mo(x) | g)' got unseen groups for '{factor_name}': " f"{sorted(bad)}"
                    )
                group_index = np.asarray(recoded.codes).astype("int64")

            # Simplex partial sums (chain, draw, n)
            simplex_np = np.asarray(posterior[term.simplex_name])  # (chain, draw, D)
            cumsum = np.cumsum(simplex_np, axis=-1)
            zero = np.zeros(cumsum.shape[:-1] + (1,))
            cumsum = np.concatenate([zero, cumsum], axis=-1)  # (chain, draw, D+1)
            partial = term.D * np.take(cumsum, codes, axis=-1)  # (chain, draw, n)

            # Per-group slope draws
            r_g_np = np.asarray(posterior[term_aliased_name])  # (chain, draw, n_groups)
            r_g_obs = np.take(r_g_np, group_index, axis=-1)  # (chain, draw, n)

            contribution_np = partial * r_g_obs
            contribution = xr.DataArray(contribution_np, dims=("chain", "draw", response_dim))
            linear_predictor += contribution
            if monotonic_dict is not None:
                monotonic_dict[term_name] = contribution
        return linear_predictor

    def predict_group_specific(
        self,
        posterior,
        data,
        in_sample,
        to_stack_dims,
        design_matrix_dims,
        sample_new_groups,
        random_seed,
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
                    posterior=posterior,
                    to_stack_dims=to_stack_dims,
                    factors_with_new_levels=factors_with_new_levels,
                    random_seed=random_seed,
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

            to_stack_dims_len = len(to_stack_dims)
            assert 2 <= to_stack_dims_len <= 3

            if to_stack_dims_len == 2:  # univariate response
                offset = 0
            else:  # multivariate response
                offset = 1

            coords_len = len(draws.coords)
            assert 3 + offset <= coords_len <= 4 + offset

            if coords_len == 3 + offset:  # numeric
                u_columns = draws.to_numpy()
            else:  # categoric
                u_columns = draws.stack(column=(factor_dim, expr_dim)).to_numpy()

            u_arrays.append(u_columns)

        u_dims = ["chain", "draw", "__variables__"]
        if to_stack_dims_len == 3:
            u_dims.insert(2, to_stack_dims[-1])

        u = np.concatenate(u_arrays, axis=-1)
        u = xr.DataArray(u, dims=u_dims)

        # Remove columns that belong to MonotonicGroupSpecificTerms (their
        # contribution is computed separately in predict_monotonic_group_specific).
        mono_gs_slices = [
            self.design.group.slices[name] for name in self.monotonic_group_specific_terms
        ]
        if mono_gs_slices:
            Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
            Z = np.delete(Z, np.r_[tuple(mono_gs_slices)], axis=1)
            Z = sp_sparse.csr_matrix(Z)

        # NOTE: xarray supports sparse matrices from the 'sparse' package, not from SciPy.
        Z = xr.DataArray(sparse.COO.from_scipy_sparse(Z), dims=design_matrix_dims)
        # Ensure the result's `.data` is a dense NumPy array.
        return xr.dot(Z, u).as_numpy()

    def _construct_u_with_new_groups(
        self, posterior, to_stack_dims, factors_with_new_levels, random_seed
    ):
        u_list = []
        names_list = []
        factor_idxs = {}
        draw_n = len(posterior.coords["draw"])
        chain_n = len(posterior.coords["chain"])
        rng = np.random.default_rng(random_seed)
        seq_draw = np.arange(draw_n)
        seq_chain = np.arange(chain_n)

        to_stack_dims_len = len(to_stack_dims)
        assert 2 <= to_stack_dims_len <= 3
        is_univariate = to_stack_dims_len == 2

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
                draws_original_ndim = draws_original.ndim

                if is_univariate:
                    assert 3 <= draws_original_ndim <= 4

                    if draws_original.ndim == 3:
                        # Numeric predictors
                        draws_new_group = draws_original[:, seq_draw, factor_sampled_idxs]
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }
                    else:
                        # Categoric predictors
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
                    assert 4 <= draws_original_ndim <= 5
                    response_dim = to_stack_dims[-1]

                    if draws_original_ndim == 4:
                        draws_new_group = draws_original[:, seq_draw, :, factor_sampled_idxs]
                        draws_new_group = np.swapaxes(draws_new_group, 0, 1)
                        coords = {
                            "chain": seq_chain,
                            "draw": seq_draw,
                            response_dim: posterior.coords[response_dim].to_numpy(),
                            factor_dim: ["__NEW_FACTOR_GROUP__"],
                        }
                    else:
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

        # Get an xr.Dataset with the draws of the terms that don't have new groups
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
    def common_terms(self):
        """Return dict of all common effects in the model component."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if isinstance(v, CommonTerm) and not isinstance(v, OffsetTerm) and v.kind != "intercept"
        }

    @property
    def group_specific_terms(self):
        """Return dict of all *regular* group-specific effects in model component."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, GroupSpecificTerm)}

    @property
    def monotonic_group_specific_terms(self):
        """Return dict of all group-specific monotonic ``(mo(x)|g)`` terms."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, MonotonicGroupSpecificTerm)}

    @property
    def offset_terms(self):
        """Return dict of all offset effects in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, OffsetTerm)}

    @property
    def hsgp_terms(self):
        """Return dict of all HSGP terms in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, HSGPTerm)}

    @property
    def monotonic_terms(self):
        """Return dict of all monotonic mo() terms in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, MonotonicTerm)}

    @property
    def monotonic_interaction_terms(self):
        """Return dict of all monotonic-interaction terms in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, MonotonicInteractionTerm)}


class ResponseComponent:
    def __init__(self, response, spec):
        self.term = None
        self.response = response
        self.spec = spec
        self._init_response()

    def _init_response(self):
        response = self.response

        if hasattr(response.term.term.components[0], "reference"):
            reference = response.term.term.components[0].reference
        else:
            reference = None

        # This is a historical feature.
        # It's not clear how many family specific checks should be added here
        if reference is not None and not isinstance(self.spec.family, univariate.Bernoulli):
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        if isinstance(self.spec.family, univariate.Bernoulli):
            if response.kind == "categoric" and response.levels is None and reference is None:
                raise ValueError("Categoric response must be binary for 'bernoulli' family.")
            if response.kind == "numeric" and not all(np.isin(response.design_matrix, (0, 1))):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

        self.term = ResponseTerm(response, self.spec.family)


def prepare_prior(prior, kind, auto_scale):
    """Helper function to correctly set default priors and auto scaling

    Parameters
    ----------
    prior : Prior or None
        The prior.
    kind : string
        Accepted values are: `"intercept"`, `"common"`, or `"group_specific"`.
    auto_scale : bool
        Whether priors should be scaled or not. Defaults to `True`.

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
        raise ValueError("'prior' must be instance of Prior or `None`.")
    return prior
