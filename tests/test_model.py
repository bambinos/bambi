import pytest
from bambi.models import Term
from os.path import dirname, join
import pandas as pd

@pytest.fixture(scope="module")
def diabetes_data():
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'diabetes.txt'), sep='\t')
    data['old'] = (data['AGE'] > 40).astype(int)
    return data

def test_term_init(diabetes_data):
    term = Term('BMI', diabetes_data)
    # Test that all defaults are properly initialized
    assert term.label == 'BMI'
    assert term.transformations == []
    assert term.categorical == False
    assert term.random == False
    assert term.split_by is None
    assert term.data_source.shape == (442, 12)
    assert term.drop_first == False
    assert term.levels is None
    assert term.values.shape == (442, 1)

def test_term_split(diabetes_data):
    split_by = Term('old', diabetes_data, categorical=True)
    term = Term('BMI', diabetes_data, split_by=split_by)
    assert term.values.shape == (442, 1, 2)
    term = Term('BMI', diabetes_data, categorical=True, split_by=split_by)
    assert term.values.shape == (442, 163, 2)

def test_transformation(diabetes_data):

    stdize = lambda x: (x - x.mean()) / x.std()

    # Test predefined transformations
    scaled = stdize(diabetes_data['BMI'].values)
    term = Term('BMI', diabetes_data)
    term.transform('scale')
    assert (term.values.ravel() == scaled).all()

    # Test transformation using callable
    def add_ten(data):
        return data + 10
    term = Term('BMI', diabetes_data)
    term.transform(add_ten)
    assert ((diabetes_data['BMI'].values + 10) == term.values.ravel()).all()

    # # Test groupby
    scaled = diabetes_data.groupby('old')['BMI'].apply(stdize).values
    term = Term('BMI', diabetes_data)
    term.transform('scale', groupby='old')
    assert (term.values.ravel() == scaled).all()
