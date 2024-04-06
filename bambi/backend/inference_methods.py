import importlib
import operator


class InferenceMethods:
    """Obtain a dictionary of available inference methods for Bambi
    models, and or the kwargs that each inference method accepts.
    """

    def __init__(self):
        self.bayeux_model = bayeux_model()
    
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
        # TODO: Somehow add the ability to retrieve PyMC kwargs of 
        # TODO: `pymc.sampling.mcmc.sample`
        # Bambi only supports bayeux MCMC methods
        if method not in self.bayeux_model.methods["mcmc"]:
            raise ValueError(
                f"Inference method '{method}' not found in the list of available"
                 " methods"
                )

        bx_method = operator.attrgetter(method)(self.bayeux_model.mcmc)
        return bx_method.get_kwargs()
    
    @property
    def names(self):
        # TODO: Add PyMC MCMC methods
        return self.bayeux_model.methods.get("mcmc")


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
        return {"mcmc": []}

    import bayeux as bx  # pylint: disable=import-outside-toplevel
    return bx.Model(lambda x: -(x**2), 0.0)


inference_methods = InferenceMethods()