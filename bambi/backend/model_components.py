import numpy as np

from pytensor import tensor as pt

from bambi.backend.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, InterceptTerm, ResponseTerm
from bambi.backend.utils import get_distribution_from_prior
from bambi.families.multivariate import MultivariateFamily
from bambi.families.univariate import Categorical, Cumulative, StoppingRatio
from bambi.utils import get_aliased_name


ORDINAL_FAMILIES = (Cumulative, StoppingRatio)


class ConstantComponent:
    def __init__(self, component):
        self.component = component
        self.output = 0

    def build(self, pymc_backend, bmb_model):
        response_aliased_name = get_aliased_name(bmb_model.response_component.response_term)
        if self.component.alias:
            label = self.component.alias
        else:
            label = f"{response_aliased_name}_{self.component.name}"

        # NOTE: This could be handled in a different manner in the future, only applies to
        # thresholds and assumes we always do it when using ordinal families.
        extra_args = {}
        if isinstance(bmb_model.family, ORDINAL_FAMILIES):
            threshold_dim = label + "_dim"
            threshold_values = np.arange(len(bmb_model.response_component.response_term.levels) - 1)
            extra_args["dims"] = threshold_dim
            pymc_backend.model.add_coords({threshold_dim: threshold_values})

        with pymc_backend.model:
            # Set to a constant value
            if isinstance(self.component.prior, (int, float)):
                self.output = self.component.prior
            # Set to a distribution
            else:
                dist = get_distribution_from_prior(self.component.prior)
                self.output = dist(label, **self.component.prior.args, **extra_args)


class DistributionalComponent:
    def __init__(self, component):
        self.component = component
        self.output = 0
        self.has_intercept = self.component.intercept_term is not None
        self.design_matrix_without_intercept = None
        self.terms = {}

    def build(self, pymc_backend, bmb_model):
        # Coordinates for the response are added first
        self.add_response_coords(pymc_backend, bmb_model)
        with pymc_backend.model:
            self.build_intercept(bmb_model)
            self.build_offsets()
            self.build_common_terms(pymc_backend, bmb_model)
            self.build_hsgp_terms(pymc_backend, bmb_model)
            self.build_group_specific_terms(pymc_backend, bmb_model)

    def build_intercept(self, bmb_model):
        if self.has_intercept:
            self.output += InterceptTerm(self.component.intercept_term).build(bmb_model)

    def build_offsets(self):
        """Add intercept term to the PyMC model.

        We have linear predictors of the form 'X @ b + Z @ u'. This is technically part of
        'X @ b' but it is added separately for convenience reasons.
        """
        for offset in self.component.offset_terms.values():
            self.output += offset.data.squeeze()

    def build_common_terms(self, pymc_backend, bmb_model):
        """Add common (fixed) terms to the PyMC model.

        We have linear predictors of the form 'X @ b + Z @ u'.
        This creates the 'b' parameter vector in PyMC, computes `X @ b`, and adds it to ``self.mu``.

        Parameters
        ----------
        spec : bambi.Model
            The model.
        """
        if self.component.common_terms:
            coefs = []
            columns = []
            for term in self.component.common_terms.values():
                common_term = CommonTerm(term)
                # Add coords
                for name, values in common_term.coords.items():
                    pymc_backend.model.add_coords({name: values})

                # Build
                coef, data = common_term.build(bmb_model)
                coefs.append(coef)
                columns.append(data)

            # Column vector of coefficients and design matrix
            coefs = pt.concatenate(coefs)

            # Design matrix
            data = np.column_stack(columns)

            # If there's an intercept, center the data
            # Also store the design matrix without the intercept to uncenter the intercept later
            if self.has_intercept and bmb_model.center_predictors:
                self.design_matrix_without_intercept = data
                data = data - data.mean(0)

            # Add term to linear predictor
            self.output += pt.dot(data, coefs)

    def build_hsgp_terms(self, pymc_backend, bmb_model):
        """Add HSGP (Hilbert-Space Gaussian Process approximation) terms to the PyMC model.

        The linear predictor 'X @ b + Z @ u' can be augmented with non-parametric HSGP terms
        'f(x)'. This creates the 'f(x)' and adds it ``self.output``.
        """
        for term in self.component.hsgp_terms.values():
            hsgp_term = HSGPTerm(term)
            for name, values in hsgp_term.coords.items():
                if name not in pymc_backend.model.coords:
                    pymc_backend.model.add_coords({name: values})
            self.output += hsgp_term.build(bmb_model)

    def build_group_specific_terms(self, pymc_backend, bmb_model):
        """Add group-specific (random or varying) terms to the PyMC model.

        We have linear predictors of the form 'X @ b + Z @ u'.
        This creates the 'u' parameter vector in PyMC, computes `Z @ u`, and adds it to
        ``self.output``.
        """
        for term in self.component.group_specific_terms.values():
            group_specific_term = GroupSpecificTerm(term, bmb_model.noncentered)

            # Add coords
            for name, values in group_specific_term.coords.items():
                if name not in pymc_backend.model.coords:
                    pymc_backend.model.add_coords({name: values})

            # Build
            coef, predictor = group_specific_term.build(bmb_model)

            # Add to the linear predictor
            # The loop through predictor columns is not the most beautiful alternative.
            # But it's the fastest. Doing matrix multiplication, pm.math.dot(data, coef), is slower.
            if predictor.ndim > 1:
                for col in range(predictor.shape[1]):
                    self.output += coef[:, col] * predictor[:, col]
            elif isinstance(bmb_model.family, (MultivariateFamily, Categorical)):
                self.output += coef * predictor[:, np.newaxis]
            else:
                self.output += coef * predictor

    def build_response(self, pymc_backend, bmb_model):
        # Extract the response term from the Bambi family
        response_term = bmb_model.response_component.response_term

        # Create and build the response term
        response_term = ResponseTerm(response_term, bmb_model.family)
        response_term.build(pymc_backend, bmb_model)

    def add_response_coords(self, pymc_backend, bmb_model):
        response_term = bmb_model.response_component.response_term
        response_name = get_aliased_name(response_term)
        dim_name = f"{response_name}_obs"
        dim_value = np.arange(response_term.shape[0])
        pymc_backend.model.add_coords({dim_name: dim_value})


# # NOTE: Here for historical reasons, not supposed to work now at least for now
# def add_lkj(backend, terms, eta=1):
#     """Add correlated prior for group-specific effects.

#     This function receives a list of group-specific terms that share their `grouper`, constructs
#     a multivariate Normal prior with LKJ prior on the correlation matrix, and adds the necessary
#     variables to the model. It uses a non-centered parametrization.

#     Parameters
#     ----------
#     terms: list
#         A list of terms that share a common grouper (i.e. ``1|Group`` and ``Variable|Group`` in
#         formula notation).
#     eta: num
#         The value for the eta parameter in the LKJ distribution.

#     Parameters
#     ----------
#     mu
#         The contribution to the linear predictor of the roup-specific terms in ``terms``.
#     """

#     # Parameters
#     # grouper: The name of the grouper.build_group_specific_distribution
#     # rows: Sum of the number of columns in all the "Xi" matrices for a given grouper.
#     #       Same than the order of L
#     # cols: Number of groups in the grouper variable
#     mu = 0
#     grouper = terms[0].name.split("|")[1]
#     rows = int(np.sum([term.predictor.shape[1] for term in terms]))
#     cols = int(terms[0].grouper.shape[1])  # not the most beautiful, but works

#     # Construct sigma
#     # Horizontally stack the sigma values for all the hyperpriors
#     sigma = np.hstack([term.prior.args["sigma"].args["sigma"] for term in terms])

#     # Reconstruct the hyperprior for the standard deviations, using one variable
#     sigma = pm.HalfNormal.dist(sigma=sigma, shape=rows)

#     # Obtain Cholesky factor for the covariance
#     # pylint: disable = unused-variable, disable=unpacking-non-sequence
#     (lkj_decomp, corr, sigma,) = pm.LKJCholeskyCov(
#         "_LKJCholeskyCov_" + grouper,
#         n=rows,
#         eta=eta,
#         sd_dist=sigma,
#         compute_corr=True,
#         store_in_trace=False,
#     )

#     coefs_offset = pm.Normal("_LKJ_" + grouper + "_offset", mu=0, sigma=1, shape=(rows, cols))
#     coefs = pt.dot(lkj_decomp, coefs_offset).T

#     ## Separate group-specific terms
#     start = 0
#     for term in terms:
#         label = term.name
#         dims = list(term.coords)

#         # Add coordinates to the model, only if they are not added yet.
#         for name, values in term.coords.items():
#             if name not in backend.model.coords:
#                 backend.model.add_coords({name: values})

#         predictor = term.predictor.squeeze()
#         delta = term.predictor.shape[1]

#         if delta == 1:
#             idx = start
#         else:
#             idx = slice(start, start + delta)

#         # Add prior for the parameter
#         coef = pm.Deterministic(label, coefs[:, idx], dims=dims)
#         coef = coef[term.group_index]

#         # Add standard deviation of the hyperprior distribution
#         group_dim = [dim for dim in dims if dim.endswith("_group_expr")]
#         pm.Deterministic(label + "_sigma", sigma[idx], dims=group_dim)

#         # Account for the contribution of the term to the linear predictor
#         if predictor.ndim > 1:
#             for col in range(predictor.shape[1]):
#                 mu += coef[:, col] * predictor[:, col]
#         else:
#             mu += coef * predictor
#         start += delta

#     # TO DO: Add correlations
#     return mu
