from bambi.interpret import logger


def log_interpret_defaults(func):
    """
    Decorator for functions that compute default values.

    Logs outpout to console if `bmb.interpret.logger.messages = True` and when
    default values are computed for the variable of interest, i.e., `contrast`
    or `wrt` of `comparisons` and `slopes`, as well as the `conditional`
    parameter of `comparisons`, `predictions`, and `slopes`.
    """
    interpret_logger = logger.get_logger("interpret")

    def wrapper(*args, **kwargs):

        if not logger.messages:
            return func(*args, **kwargs)

        result = None
        name_key = None
        term_name = None

        if func.__name__ in ["set_default_values", "make_group_panel_values"]:
            data_dict = kwargs.get("data_dict", args[1])
            keys_before = set(data_dict.keys())
            result = func(*args, **kwargs)
            term_name = ", ".join(set(result.keys()) - keys_before)  # keys after

            if len(term_name) > 0:
                name_key = "unspecified" if func.__name__ == "set_default_values" else "group/panel"

        elif func.__name__ == "make_main_values":
            term_name = args[1]
            name_key = "main"

        elif func.__name__ == "set_default_variable_values":
            variables = {"comparisons": "contrast", "slopes": "wrt"}
            name_key = variables.get(args[0].kind)
            term_name = args[0].name

        if name_key:
            interpret_logger.info("Default computed for %s variable: %s", name_key, term_name)

        return func(*args, **kwargs)

    return wrapper
