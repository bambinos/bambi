# pylint: disable=unused-argument
import pytensor.tensor as pt
import numpy as np
import xarray as xr

from bambi.families.family import Family
from bambi.utils import extract_argument_names, extra_namespace


class MultivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        return NotImplemented

    def posterior_predictive(self, model, posterior, linear_predictor):
        return NotImplemented


class Categorical(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        # This is only 'softmax' for now.
        # Because of reference encoding, we need to padd with 0s for the first level of the
        # response variable
        # (1, 0): 1 new levels on the left, 0 new level on the right
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        # The mean has the reference level in the dimension, a new name is needed
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        posterior[response_var] = mean
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        # https://stackoverflow.com/questions/34187130
        def draw_categorical_samples(probability_matrix, items):
            """
            probability_matrix is a matrix of shape (n_chain * n_draw, n_levels)
            """
            cumsum = probability_matrix.cumsum(axis=1)
            idx = np.random.rand(probability_matrix.shape[0])[:, np.newaxis]
            idx = (cumsum < idx).sum(axis=1)
            return items[idx]

        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        mean = mean.to_numpy()
        shape = mean.shape
        # Stack chains and draws
        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        pps = np.empty((draws_n, obs_n), dtype=int)
        response_levels = np.arange(len(model.response.levels))

        for idx in range(obs_n):
            pps[:, idx] = draw_categorical_samples(mean[:, idx, :], response_levels)

        pps = pps.reshape((shape[0], shape[1], obs_n))
        pps = xr.DataArray(
            pps,
            coords={
                "chain": np.arange(shape[0]),
                "draw": np.arange(shape[1]),
                response_dim: np.arange(obs_n),
            },
        )
        return pps

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    def get_coords(self, response):
        name = response.name + "_dim"
        return {name: [level for level in response.levels if level != response.reference]}

    def get_reference(self, response):
        return get_reference_level(response.term)

    @staticmethod
    def transform_backend_nu(nu, data):
        # Add column of zeros to the linear predictor for the reference level (the first one)
        shape = (data.shape[0], 1)

        # The first line makes sure the intercept-only models work
        nu = np.ones(shape) * nu  # (response_levels, ) -> (n, response_levels)
        nu = pt.concatenate([np.zeros(shape), nu], axis=1)
        return nu


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        # The mean has the reference level in the dimension, a new name is needed
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        posterior[response_var] = mean
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        mean = mean.to_numpy()
        shape = mean.shape

        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        pps = np.empty(mean.shape, dtype=int)
        n = model.response.data.sum(1)

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
                response_levels_dim_complete: model.response.levels,
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


# pylint: disable = protected-access
def get_reference_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
