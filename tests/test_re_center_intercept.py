import numpy as np
import pytest

import bambi as bmb
from bambi.utils import get_aliased_name


@pytest.mark.usefixtures("mock_pymc_sample")
class TestReCenterIntercept:
    @staticmethod
    def _check_recentering(model, idata):
        for pymc_component in model.backend.distributional_components.values():
            bambi_component = pymc_component.component
            if not (
                bambi_component.intercept_term
                and bambi_component.common_terms
                and pymc_component.design_matrix_without_intercept is not None
            ):
                continue

            common_names = [get_aliased_name(t) for t in bambi_component.common_terms.values()]

            for name in common_names:
                idata.posterior[name] = idata.posterior[name] * 0 + 1.0

            intercept_before = idata.posterior["Intercept"].values.copy()

            idata_corrected = model._re_center_intercept(idata)
            intercept_after = idata_corrected.posterior["Intercept"].values

            x_uncentered = pymc_component.design_matrix_without_intercept
            expected_shift = x_uncentered.mean(0).sum()

            actual_shift = intercept_before - intercept_after
            np.testing.assert_allclose(actual_shift, expected_shift)

    def test_numerical(self, integer_data_fixture):
        model, idata = integer_data_fixture
        self._check_recentering(model, idata)

    def test_categorical_and_interactions(self, mtcars_fixture):
        model, idata = mtcars_fixture
        self._check_recentering(model, idata)

    def test_categorical_numerical(self, data_inhaler):
        model = bmb.Model("rating ~ treat + period + carry", data_inhaler, family="categorical")
        idata = model.fit(tune=200, draws=200, chains=2)
        self._check_recentering(model, idata)

    def test_categorical_categoricals(self, food_choice):
        model, idata = food_choice
        self._check_recentering(model, idata)

    def test_center_predictors_false(self, data_inhaler):
        model = bmb.Model(
            "rating ~ treat + period + carry",
            data_inhaler,
            family="categorical",
            center_predictors=False,
        )
        idata = model.fit(tune=200, draws=200, chains=2)

        idata_corrected = model._re_center_intercept(idata)
        assert idata_corrected is idata
