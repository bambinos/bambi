import logging

from bambi.interpret import logger


def log_interpret_defaults(func):
    interpret_logger = logger.get_logger()
    interpret_logger.setLevel(logging.INFO)

    def wrapper(*args, **kwargs):
        if not logger.messages:
            return func(*args, **kwargs)

        if func.__name__ == "set_default_values":
            data_dict = kwargs.get("data_dict", None)

            if data_dict is None:
                data_dict = args[1]

            pre_keys = set(data_dict.keys())
            result = func(*args, **kwargs)
            post_keys = set(result.keys())
            added_keys = post_keys - pre_keys

            if len(added_keys) > 0:
                logging.info(f"Default values computed for: {added_keys}")

            return result

        # TODO: add other "default value" functions here to log their output

    return wrapper
