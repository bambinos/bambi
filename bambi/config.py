class Config:
    """Configuration variables for Bambi

    It works with a pre-specified set of configuration variables and options for those.
    When a user tries to set a configuration variable to a non-supported value, it raises an error.
    """

    __FIELDS = {"INTERPRET_VERBOSE": (True, False)}

    def __init__(self, config_dict: dict = None):
        config_dict = {} if config_dict is None else config_dict
        # When an option is not specified at instantiation time, it uses the first value specified
        # in __FIELDS.
        for field, choices in Config.__FIELDS.items():
            if field in config_dict:
                value = config_dict[field]
            else:
                value = choices[0]
            self[field] = value

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __setattr__(self, key, value):
        if key in Config.__FIELDS:
            if value in Config.__FIELDS[key]:
                super().__setattr__(key, value)
            else:
                raise ValueError(
                    f"{value} is not a valid value for '{key}'"
                    f"Valid options are: {Config.__FIELDS[key]}"
                )
        else:
            raise KeyError(f"'{key}' is not a valid configuration option")

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):  # pragma: no cover
        lines = []
        for field, choices in Config.__FIELDS.items():
            lines.append(f"{field}: {self[field]} (available: {list(choices)})")
        header = ["Bambi configuration"]
        header.append("-" * len(header[0]))
        return "\n".join(header + lines)

    def __repr__(self):  # pragma: no cover
        return str(self)


config = Config()
