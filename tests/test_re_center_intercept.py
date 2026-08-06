from copy import deepcopy

import numpy as np
import pytest

import bambi as bmb
from bambi.utils import as_dataset, get_aliased_name


@pytest.mark.usefixtures("mock_pymc_sample")
class TestReCenterIntercept:
    @staticmethod
    def _check_recentering(model, idata):
        idata = deepcopy(idata)

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

            intercept_name = get_aliased_name(bambi_component.intercept_term)
            intercept_before = idata.posterior[intercept_name].values.copy()

            idata_corrected = model._re_center_intercept(idata)
            intercept_after = idata_corrected.posterior[intercept_name].values

            expected_shift = pymc_component.design_matrix_without_intercept.mean(0).sum()

            actual_shift = intercept_after - intercept_before
            np.testing.assert_allclose(actual_shift, expected_shift)
            np.testing.assert_array_equal(idata.posterior[intercept_name].values, intercept_before)

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

    def test_offsets_match_sampled_offsets(self, data_sleepstudy):
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data_sleepstudy)
        idata = model.fit(tune=100, draws=100, chains=2, omit_offsets=False)

        offset_names = [name for name in idata.posterior.data_vars if name.endswith("_offset")]
        assert set(offset_names) == {"1|Subject_offset", "Days|Subject_offset"}

        idata_dropped = idata.copy()
        idata_dropped["posterior"] = as_dataset(idata["posterior"]).drop_vars(offset_names)

        idata_corrected = model._re_center_intercept(idata_dropped)

        for name in offset_names:
            np.testing.assert_allclose(
                idata_corrected.posterior[name].values, idata.posterior[name].values
            )

    def test_offsets_reconstructed_without_centering(self, data_sleepstudy):
        model = bmb.Model(
            "Reaction ~ Days + (Days | Subject)", data_sleepstudy, center_predictors=False
        )
        idata = model.fit(tune=100, draws=100, chains=2)

        assert not any(name.endswith("_offset") for name in idata.posterior.data_vars)

        idata_corrected = model._re_center_intercept(idata)

        assert "1|Subject_offset" in idata_corrected.posterior
        assert "Days|Subject_offset" in idata_corrected.posterior

    def test_offsets_reconstructed_with_sigma_alias(self, data_sleepstudy):
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data_sleepstudy)
        model.set_alias({"sigma": "tau"})
        idata = model.fit(tune=100, draws=100, chains=2)

        idata_corrected = model._re_center_intercept(idata)

        for base_name in ["1|Subject", "Days|Subject"]:
            assert f"{base_name}_tau" in idata_corrected.posterior
            assert f"{base_name}_offset" in idata_corrected.posterior

    def test_no_offsets_when_centered_parametrization(self, data_sleepstudy):
        model = bmb.Model("Reaction ~ Days + (Days | Subject)", data_sleepstudy, noncentered=False)
        idata = model.fit(tune=100, draws=100, chains=2)

        idata_corrected = model._re_center_intercept(idata)

        assert not any(name.endswith("_offset") for name in idata_corrected.posterior.data_vars)
