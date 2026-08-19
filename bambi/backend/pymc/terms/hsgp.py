import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.terms.info import HSGPTermInfo
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.families.types import ParamSpec
from bambi.priors import Prior


def exp_quad(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.ExpQuad(input_dim, ls=ell)


def matern32(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern32(input_dim, ls=ell)


def matern52(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern52(input_dim, ls=ell)


GP_KERNELS = {
    "ExpQuad": {"fn": exp_quad, "params": ("sigma", "ell")},
    "Matern32": {"fn": matern32, "params": ("sigma", "ell")},
    "Matern52": {"fn": matern52, "params": ("sigma", "ell")},
}


def build_hsgp_term(term_info: HSGPTermInfo, param_spec: ParamSpec, model: pm.Model) -> pt.Variable:
    """Build and return the contribution of an HSGP term.

    Parameters
    ----------
    term_info : HSGPTermInfo
        Static information for the HSGP term to build.
    param_spec : ParamSpec
        Dimensionality metadata for the conditional parameter.
    model : pymc.Model
        The model that owns the term's variables and coordinates.
    """
    term = term_info.term
    model.add_coords(term_info.coords)
    covariance_functions = build_covariance_function(term, model)

    # Prepare dims
    coeff_dims = (f"{term.label}_weights_dim",)
    contribution_dims = ("__obs__",)

    # training_hsgp_data initializes fixed HSGP quantities from the training data.
    if term.scale_predictors:
        training_hsgp_data = term.data_centered / term.maximum_distance
    else:
        training_hsgp_data = term.data_centered

    # input_data holds raw inputs. hsgp_data is centered and scaled in the model graph
    # with quantities estimated from the training data.
    input_data = pm.Data(f"{term.label}_data", term.data, model=model)

    # Build HSGP object(s) and retain them on the term.
    if term.by_levels is not None:
        coeff_dims = coeff_dims + (f"{term.label}_by",)
        phi_list, sqrt_psd_list = [], []
        term.hsgp = {}

        by_data = pm.Data(f"{term.label}_by_idx", term.by, dims="__obs__", model=model)
        hsgp_data = input_data - pt.as_tensor_variable(term.mean)[by_data]
    else:
        hsgp_data = input_data - term.mean

    if term.scale_predictors:
        hsgp_data = hsgp_data / term.maximum_distance

    if term.by_levels is not None:
        for i, level in enumerate(term.by_levels):
            cov_func = covariance_functions[i]
            # Notes:
            # 'm' doesn't change by group
            # We need to use list() in 'm' and 'L' because arrays are not instance of Sequence
            hsgp = pm.gp.HSGP(
                m=list(term.m),
                L=list(term.L[i]),
                drop_first=term.drop_first,
                cov_func=cov_func,
            )
            # training_hsgp_data fixes the HSGP centering constants for this group; hsgp_data
            # keeps the basis dependent on the mutable input_data.
            _, sqrt_psd = hsgp.prior_linearized(training_hsgp_data[term.by == i])
            phi, _ = hsgp.prior_linearized(hsgp_data)
            sqrt_psd_list.append(sqrt_psd)
            phi_list.append(phi)

            # Store it for later usage
            term.hsgp[level] = hsgp
        sqrt_psd = pt.stack(sqrt_psd_list, axis=1)
    else:
        (cov_func,) = covariance_functions
        term.hsgp = pm.gp.HSGP(
            m=list(term.m),
            L=list(term.L[0]),
            drop_first=term.drop_first,
            cov_func=cov_func,
        )
        # Get the basis and basis-weight scale for the mutable hsgp_data.
        phi, sqrt_psd = term.hsgp.prior_linearized(hsgp_data)

    # Build weights coefficient
    # Handle the case where the outcome is multivariate
    if param_spec.ndim == 1:
        # Append the dims of the response variables to the coefficient and contribution dims
        # In general:
        # coeff_dims: ('weights_dim', ) -> ('weights_dim', f'{response}_dim')
        # contribution_dims: ('__obs__', ) -> ('__obs__', f'{response}_dim')
        response_dims = tuple(
            model.__bambi_attrs__["response_coords_data"]
            | model.__bambi_attrs__["response_coords_reduced"]
        )
        coeff_dims = coeff_dims + response_dims
        contribution_dims = contribution_dims + response_dims

        # Append a dimension to sqrt_psd: ('weights_dim', ) -> ('weights_dim', 1)
        sqrt_psd = sqrt_psd[:, np.newaxis]

    with model:
        if term.centered:
            coeffs = pm.Normal(f"{term.label}_weights", sigma=sqrt_psd, dims=coeff_dims)
        else:
            coeffs_raw = pm.Normal(f"{term.label}_weights_raw", dims=coeff_dims)
            coeffs = pm.Deterministic(
                f"{term.label}_weights", coeffs_raw * sqrt_psd, dims=coeff_dims
            )

    # Build deterministic for the HSGP contribution
    # If there are groups, we do as many dot products as groups
    if term.by_levels is not None:
        contribution_list = []
        for i in range(len(term.by_levels)):
            contribution_list.append(pt.dot(phi_list[i], coeffs[:, i]))
        contribution_by_group = pt.stack(contribution_list, axis=1)
        contribution = contribution_by_group[pt.arange(hsgp_data.shape[0]), by_data]
    # If there are no groups, it's a single dot product
    else:
        contribution = pt.dot(phi, coeffs)  # "@" operator is not working as expected

    with model:
        return pm.Deterministic(term.label, contribution, dims=contribution_dims)


def build_covariance_function(term, model):
    cov_dict = GP_KERNELS[term.cov]
    create_covariance_function = cov_dict["fn"]
    param_names = cov_dict["params"]
    params = {}

    # Set dimensions and behavior for priors that are actually fixed (floats or ints)
    if term.by_levels is not None and not term.share_cov:
        dims = (f"{term.label}_by",)
        recycle = True
    else:
        dims = tuple()
        recycle = False

    # Build priors and parameters
    for param_name in param_names:
        prior = term.prior[param_name]
        param_dims = dims
        if isinstance(prior, Prior):
            dist = get_distribution_from_prior(prior)
            # varying lengthscale parameter
            if param_name == "ell" and not term.iso and term.shape[1] > 1:
                param_dims = (f"{term.label}_var",) + param_dims
            with model:
                value = dist(f"{term.label}_{param_name}", **prior.args, dims=param_dims)
        else:
            # The value is constant
            if recycle:
                value = np.full(term.groups_n, prior)
            else:
                value = prior

        params[param_name] = value

    if term.share_cov:
        params["input_dim"] = term.shape[1]
    else:
        # squeeze makes sure the array is 0d when term.groups_n is 1
        params["input_dim"] = np.repeat(term.shape[1], term.groups_n).squeeze()

    if term.groups_n == 1 or term.share_cov:
        # All groups use the same covariance function
        covariance_function = create_covariance_function(**params)
        output = [covariance_function] * term.groups_n
    else:
        # Each group gets its own covariance function
        output = []
        for i in range(len(term.by_levels)):
            params_level = {}
            for key, value in params.items():
                entry = value[..., i]
                if isinstance(entry, np.ndarray) and entry.ndim == 0:
                    entry = entry.item()
                elif isinstance(entry, np.generic):
                    entry = entry.item()
                params_level[key] = entry
            covariance_function = create_covariance_function(**params_level)
            output.append(covariance_function)

    return output
