# pylint: disable=unused-argument
import numpy as np

from .family import Family


class MultivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        return NotImplemented

    def posterior_predictive(self, model, posterior, linear_predictor):
        return NotImplemented


class Categorical(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        # This is only 'softmax' for now.
        # Last axis is the one where the response coord is inserted
        # We need to append zeros for the reference category
        shape = linear_predictor.shape
        linear_predictor = np.concatenate([np.zeros(shape[:-1] + (1,)), linear_predictor], axis=-1)

        mean = self.link.linkinv(linear_predictor, axis=-1)

        obs_n = mean.shape[2]
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        response_levels_dim = model.response.name + "_mean_dim"
        dims = ("chain", "draw", response_dim, response_levels_dim)
        posterior[response_var] = (dims, mean)

        posterior = posterior.assign_coords({response_levels_dim: model.response.levels})
        posterior = posterior.assign_coords({response_dim: list(range(obs_n))})
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        # https://stackoverflow.com/questions/34187130
        def draw_categorical_samples(probability_matrix, items):
            cumsum = probability_matrix.cumsum(axis=0)
            idx = np.random.rand(probability_matrix.shape[1])
            idx = (cumsum < idx).sum(axis=0)
            return items[idx]

        shape = linear_predictor.shape
        linear_predictor = np.concatenate([np.zeros(shape[:-1] + (1,)), linear_predictor], axis=-1)

        mean = self.link.linkinv(linear_predictor, axis=-1)
        shape = mean.shape

        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        pps = np.empty((draws_n, obs_n), dtype=int)
        response_levels = np.arange(len(model.response.levels))
        for idx in range(obs_n):
            pps[:, idx] = draw_categorical_samples(mean[:, idx, :].T, response_levels)

        return pps.reshape((shape[0], shape[1], obs_n))


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        # This is only 'softmax' for now.
        # Last axis is the one where the response coord is inserted
        # We need to append zeros for the reference category
        shape = linear_predictor.shape
        linear_predictor = np.concatenate([np.zeros(shape[:-1] + (1,)), linear_predictor], axis=-1)

        mean = self.link.linkinv(linear_predictor, axis=-1)

        obs_n = mean.shape[2]
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        response_levels_dim = model.response.name + "_mean_dim"
        dims = ("chain", "draw", response_dim, response_levels_dim)
        posterior[response_var] = (dims, mean)

        posterior = posterior.assign_coords({response_levels_dim: model.response.levels})
        posterior = posterior.assign_coords({response_dim: list(range(obs_n))})
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        shape = linear_predictor.shape
        linear_predictor = np.concatenate([np.zeros(shape[:-1] + (1,)), linear_predictor], axis=-1)

        mean = self.link.linkinv(linear_predictor, axis=-1)
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
        return pps
