import importlib
import inspect
import operator

import pymc as pm


class InferenceMethods:
    """Obtain a dictionary of available inference methods for Bambi
    models and or the default kwargs of each inference method.
    """

    def __init__(self):
        # In order to access inference methods, a bayeux model must be initialized
        self.bayeux_model = bayeux_model()
        self.bayeux_methods = self._get_bayeux_methods(bayeux_model())
        self.pymc_methods = self._pymc_methods()

    def _get_bayeux_methods(self, model):
        # If bayeux is not installed, return an empty MCMC list.
        if model is None:
            return {"mcmc": []}
        # Bambi only supports bayeux MCMC methods
        mcmc_methods = model.methods.get("mcmc")
        return {"mcmc": mcmc_methods}

    def _pymc_methods(self):
        return {"mcmc": ["mcmc"], "vi": ["vi"]}

    def _remove_parameters(self, fn_signature_dict):
        # Remove 'pm.sample' parameters that are irrelevant for Bambi users
        params_to_remove = [
            "progressbar",
            "progressbar_theme",
            "var_names",
            "nuts_sampler",
            "return_inferencedata",
            "idata_kwargs",
            "callback",
            "mp_ctx",
            "model",
        ]
        return {k: v for k, v in fn_signature_dict.items() if k not in params_to_remove}

    def get_kwargs(self, method):
        """Get the default kwargs for a given inference method.

        Parameters
        ----------
        method : str
            The name of the inference method.

        Returns
        -------
        dict
            The default kwargs for the inference method.
        """
        if method in self.bayeux_methods.get("mcmc"):
            bx_method = operator.attrgetter(method)(
                self.bayeux_model.mcmc  # pylint: disable=no-member
            )
            return bx_method.get_kwargs()
        elif method in self.pymc_methods.get("mcmc"):
            return self._remove_parameters(get_default_signature(pm.sample))
        elif method in self.pymc_methods.get("vi"):
            return get_default_signature(pm.ADVI.fit)
        else:
            raise ValueError(
                f"Inference method '{method}' not found in the list of available"
                " methods. Use `bmb.inference_methods.names` to list the available methods."
            )

    @property
    def names(self):
        return {"pymc": self.pymc_methods, "bayeux": self.bayeux_methods}


def bayeux_model():
    """Dummy bayeux model for obtaining inference methods.

    A dummy model is needed because algorithms are dynamically determined at
    runtime, based on the libraries that are installed. A model can give
    programmatic access to the available algorithms via the `methods` attribute.

    Returns
    -------
    bayeux.Model
        A dummy model with a simple quadratic likelihood function.
    """
    if importlib.util.find_spec("bayeux") is None:
        return None

    import bayeux as bx  # pylint: disable=import-outside-toplevel

    return bx.Model(lambda x: -(x**2), 0.0)


def get_default_signature(fn):
    """Get the default parameter values of a function.

    This function inspects the signature of the provided function and returns
    a dictionary containing the default values of its parameters.

    Parameters
    ----------
    fn : callable
        The function for which default argument values are to be retrieved.

    Returns
    -------
    dict
        A dictionary mapping argument names to their default values.

    """
    defaults = {}
    for key, val in inspect.signature(fn).parameters.items():
        if val.default is not inspect.Signature.empty:
            defaults[key] = val.default
    return defaults


inference_methods = InferenceMethods()
