class TransformsRegistry:
    def __init__(self):
        self.additive_predictors = {}
        self.parameters = {}
        self.data = {}

    def transform_predictor(self, family, parameter):
        """Register transformation function for additive predictors."""
        family = self._family_class(family)

        def decorator(function):
            self.additive_predictors[(family, parameter)] = function
            return function

        return decorator

    def transform_parameters(self, family):
        """Register transformation function for parameters of the observational model."""
        family = self._family_class(family)

        def decorator(function):
            self.parameters[family] = function
            return function

        return decorator

    def transform_data(self, family):
        """Register transformation function for observational model data."""
        family = self._family_class(family)

        def decorator(function):
            self.data[family] = function
            return function

        return decorator

    def get_predictor_transform(self, family, parameter):
        return self.additive_predictors.get((self._family_class(family), parameter), None)

    def get_parameter_transform(self, family):
        return self.parameters.get(self._family_class(family), None)

    def get_data_transform(self, family):
        return self.data.get(self._family_class(family), None)

    def _family_class(self, family):
        return family if isinstance(family, type) else type(family)


transforms_registry = TransformsRegistry()
