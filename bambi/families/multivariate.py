# pylint: disable=unused-argument
import numpy as np
import xarray as xr
import pytensor.tensor as pt

from bambi.families.family import Family
from bambi.utils import extract_argument_names, extra_namespace, get_aliased_name


class MultivariateFamily(Family):
    KIND = "Multivariate"


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    UFUNC_KWARGS = {"axis": -1}

    def transform_linear_predictor(self, model, linear_predictor):
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_dim"
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        return linear_predictor

    def transform_coords(self, model, mean):
        # The mean has the reference level in the dimension, a new name is needed
        response_name = get_aliased_name(model.response_component.response_term)
        response_levels_dim = response_name + "_dim"
        response_levels_dim_complete = response_name + "_mean_dim"
        levels_complete = model.response_component.response_term.levels
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: levels_complete})
        return mean

    def posterior_predictive(self, model, posterior, **kwargs):
        response_name = get_aliased_name(model.response_component.response_term)
        response_dim = response_name + "_obs"
        response_levels_dim_complete = response_name + "_mean_dim"

        mean = posterior[response_name + "_mean"].to_numpy()
        shape = mean.shape

        # Stack chains and draws
        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        # Q: What is the right 'n' for out of sample data?
        #    right now it assumes that "N" is the same as before..
        #    It could be improved in the future!
        pps = np.empty(mean.shape, dtype=int)
        n = model.response_component.response_term.data.sum(1)

        # random.multinomial only accepts
        # * n : integer
        # * p : vector
        for i in range(obs_n):
            for j in range(draws_n):
                pps[j, i, :] = np.random.multinomial(n[i], mean[j, i, :])

        # Final shape is of (chain, draw, obs_n, response_n)
        pps = pps.reshape(shape)
        pps = xr.DataArray(
            pps,
            coords={
                "chain": np.arange(shape[0]),
                "draw": np.arange(shape[1]),
                response_dim: np.arange(obs_n),
                response_levels_dim_complete: model.response_component.response_term.levels,
            },
        )
        return pps

    def get_coords(self, response):
        # For the moment, it always uses the first column as reference.
        name = response.name + "_dim"
        labels = self.get_levels(response)
        return {name: labels[1:]}

    def get_levels(self, response):
        labels = extract_argument_names(response.name, list(extra_namespace))
        if labels:
            return labels
        return [str(level) for level in range(response.data.shape[1])]

    @staticmethod
    def transform_backend_kwargs(kwargs):
        kwargs["n"] = kwargs["observed"].sum(axis=1)
        return kwargs

    @staticmethod
    def transform_backend_nu(nu, data):
        # Add column of zeros to the linear predictor for the reference level (the first one)
        shape = (data.shape[0], 1)

        # The first line makes sure the intercept-only models work
        nu = np.ones(shape) * nu  # (response_levels, ) -> (n, response_levels)
        nu = pt.concatenate([np.zeros(shape), nu], axis=1)
        return nu
