import logging

from bambi import config


def log_interpret_defaults(func):
    """Decorator for functions that compute default values.

    Logs output to console if `bmb.config["INTERPRET_VERBOSE"] = True` and when
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
        elif func.__name__ == "create_grid":
            conditional = args[0]
            if not conditional.user_passed:
                covariate_name = ", ".join(conditional.covariates.values())
                arg_name = "conditional"
        elif func.__name__ == "set_default_values":
            data_dict = kwargs.get("data_dict", args[1])
            keys_before = list(data_dict.keys())
            keys_after = list(func(*args, **kwargs).keys())
            covariate_name = ", ".join([key for key in keys_after if key not in keys_before])

            if len(covariate_name) > 0:
                arg_name = "unspecified"

        if arg_name:
            logger.info("Default computed for %s variable: %s", arg_name, covariate_name)

        return func(*args, **kwargs)

    return wrapper
