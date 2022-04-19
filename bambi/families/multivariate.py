# pylint: disable=unused-argument
import numpy as np

from .family import Family


class MultivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        """Predict mean response"""
        mean = self.link.linkinv(linear_predictor)
        obs_n = mean.shape[-1]
        name = model.response.name + "_mean"
        coord_name = name + "_dim_0"

        # Drop var/dim if already present
        if name in posterior.data_vars:
            posterior = posterior.drop_vars(name)

        if coord_name in posterior.dims:
            posterior = posterior.drop_dims(coord_name)

        coords = ("chain", "draw", coord_name)
        posterior[name] = (coords, mean)
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior


class Categorical(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        # This is only 'softmax' for now.
        # Second axis is the one where the response coord is inserted
        # We need to append zeros for the reference category
        shape = linear_predictor.shape
        linear_predictor = np.dstack(
            (np.zeros(shape=(shape[0], shape[1], 1, shape[3])), linear_predictor)
        )

        mean = self.link.linkinv(linear_predictor, axis=2)

        obs_n = mean.shape[-1]
        name = model.response.name + "_mean"
        coord_name = model.response.name + "_obs"

        # Drop var/dim if already present
        if name in posterior.data_vars:
            posterior = posterior.drop_vars(name)

        if coord_name in posterior.dims:
            posterior = posterior.drop_dims(coord_name)

        response_coord_name = model.response.name + "_mean_coord"
        coords = ("chain", "draw", response_coord_name, coord_name)
        posterior[name] = (coords, mean)
        posterior = posterior.assign_coords({response_coord_name: model.response.levels})
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        # https://stackoverflow.com/questions/34187130
        def draw_categorical_samples(probability_matrix, items):
            cumsum = probability_matrix.cumsum(axis=0)
            idx = np.random.rand(probability_matrix.shape[1])
            idx = (cumsum < idx).sum(axis=0)
            return items[idx]

        shape = linear_predictor.shape
        linear_predictor = np.dstack(
            (np.zeros(shape=(shape[0], shape[1], 1, shape[3])), linear_predictor)
        )

        mean = self.link.linkinv(linear_predictor, axis=2)
        idxs = np.random.randint(low=0, high=draw_n, size=draws)
        mean = mean[:, idxs, :, :]
        shape = mean.shape

        mean = mean.reshape((np.prod(mean.shape[:2]), mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[-1]

        pps = np.empty((draws_n, obs_n), dtype=int)
        response_levels = np.arange(len(model.response.levels))
        for idx in range(obs_n):
            pps[:, idx] = draw_categorical_samples(mean[..., idx].T, response_levels)

        return pps.reshape((shape[0], shape[1], obs_n))


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        # This is only 'softmax' for now.
        # Second axis is the one where the response coord is inserted
        # We need to append zeros for the reference category
        shape = linear_predictor.shape
        linear_predictor = np.dstack(
            (np.zeros(shape=(shape[0], shape[1], 1, shape[3])), linear_predictor)
        )

        mean = self.link.linkinv(linear_predictor, axis=2)

        obs_n = mean.shape[-1]
        name = model.response.name + "_mean"
        coord_name = model.response.name + "_obs"

        # Drop var/dim if already present
        if name in posterior.data_vars:
            posterior = posterior.drop_vars(name)

        if coord_name in posterior.dims:
            posterior = posterior.drop_dims(coord_name)

        response_coord_name = model.response.name + "_mean_coord"
        coords = ("chain", "draw", response_coord_name, coord_name)
        posterior[name] = (coords, mean)

        # NOTE: Improve this. It would be better to have a better way to grab this list.
        posterior = posterior.assign_coords(
            {response_coord_name: [str(level) for level in range(model.response.data.shape[1])]}
        )
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor, draws, draw_n):
        return None
