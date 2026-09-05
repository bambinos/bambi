import pandas as pd
import pytest

import bambi as bmb


@pytest.mark.parametrize(
    ("deprecated_name", "replacement_name"),
    [
        ("components", "parameters"),
        ("distributional_components", "conditional_parameters"),
        ("constant_components", "marginal_parameters"),
    ],
)
def test_deprecated_component_properties(deprecated_name, replacement_name):
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    model = bmb.Model("y ~ x", data)

    with pytest.warns(FutureWarning, match=f"'{deprecated_name}'.*'{replacement_name}'") as record:
        value = getattr(model, deprecated_name)

    assert value == getattr(model, replacement_name)
    assert record[0].filename == __file__
