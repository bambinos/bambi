import numpy as np

from bambi.backend.pymc.types import Coords


def predictor_data_name(base_name: str, dims: tuple[str, ...], model) -> str:
    data_name = f"{base_name}_data"

    if _is_data_name_available(data_name, dims, model):
        return data_name

    index = 2
    while True:
        indexed_data_name = f"{base_name}_{index}_data"
        if _is_data_name_available(indexed_data_name, dims, model):
            return indexed_data_name
        index += 1


def shape_common_data(data: np.ndarray, coords: Coords) -> np.ndarray:
    # PyMC fixes mutable-data dtypes at construction; prediction grids may be fractional.
    data = np.asarray(data, dtype=float)

    if not coords:
        # Since we don't have coords, this must be a single numeric column.
        # Data only has the "__obs__" dim.
        if data.ndim == 2 and data.shape[1] == 1:
            return data.flatten()
        if data.ndim > 1:
            raise ValueError("Common term data without coordinates must be one-dimensional.")
        # It's already one dimensional.
        return data

    # Coords describe all non-observation dimensions for the term.
    # Formulae gives common terms as flat design-matrix columns,
    # while PyMC data is registered with named dimensions.
    # This restores the named coordinate shape.
    coords_shape = tuple(len(coord) for coord in coords.values())
    coords_cardinality = np.prod(coords_shape)

    # A vector is expantded into a single column matrix only when coords imply a single column.
    if data.ndim == 1:
        if coords_cardinality != 1:
            raise ValueError("Cannot reshape one-dimensional common term data to multiple levels.")
        return data[:, np.newaxis]

    # Convert design-matrix columns into an array of shape given by coords, including __obs__.
    if data.ndim == 2 and data.shape[1] == coords_cardinality:
        # (__obs__, *coords)
        return data.reshape((data.shape[0], *coords_shape))

    raise ValueError("Common term data shape does not match its coordinates.")


def _is_data_name_available(data_name: str, dims: tuple[str, ...], model) -> bool:
    if data_name not in model:
        return True
    return tuple(model.named_vars_to_dims.get(data_name, ())) == dims
