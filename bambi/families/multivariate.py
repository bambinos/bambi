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
            posterior = posterior.drop_vars(name).drop_dims(coord_name)

        coords = ("chain", "draw", coord_name)
        posterior[name] = (coords, mean)
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior


class Categorical(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        # This is only 'softmax' for now.
        # Second axis is the one where the response coord is inserted
        # We need to append zeros for  the reference category
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
            posterior = posterior.drop_vars(name).drop_dims(coord_name)

        response_coord_name = model.response.name + "_mean_coord"
        coords = ("chain", "draw", response_coord_name, coord_name)
        posterior[name] = (coords, mean)
        posterior = posterior.assign_coords({response_coord_name: model.response.levels})
        posterior = posterior.assign_coords({coord_name: list(range(obs_n))})
        return posterior
