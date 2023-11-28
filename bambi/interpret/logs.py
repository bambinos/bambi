import logging

from bambi import config


def log_interpret_defaults(func):
    """
    Decorator for functions that compute default values.

    Logs output to console if 'bmb.config["INTERPRET_VERBOSE"] = True' and when
    default values are computed for the variable of interest, i.e., 'contrast'
    or 'wrt' of 'comparisons' and 'slopes', as well as the 'conditional'
    parameter of 'comparisons', 'predictions', and 'slopes'.
    """
    logger = logging.getLogger("__bambi_interpret__")

    def wrapper(*args, **kwargs):

        if not config["INTERPRET_VERBOSE"]:
            return func(*args, **kwargs)

        arg_name = None
        covariate_name = None

        if func.__name__ == "set_default_variable_values":
            variables = {"comparisons": "contrast", "slopes": "wrt"}
            arg_name = variables.get(args[0].kind)
            covariate_name = args[0].name

        if arg_name:
            logger.info("Default computed for %s variable: %s", arg_name, covariate_name)

        return func(*args, **kwargs)

    return wrapper
