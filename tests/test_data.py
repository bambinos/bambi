import os

from urllib.parse import urlunsplit

import pandas as pd
import pytest

import bambi as bmb
from bambi.data.datasets import DATASETS, FileMetadata


@pytest.fixture(autouse=True)
def no_remote_data(monkeypatch, tmpdir):
    """Run tests without relying on remote data"""

    filename = os.path.join(str(tmpdir), os.path.basename("test_remote.csv"))
    with open(filename, "w") as fdo:
        fdo.write("x,y\n0,1")
    url = urlunsplit(("file", "", filename, "", ""))

    monkeypatch.setitem(
        DATASETS,
        "test_remote",
        FileMetadata(
            filename=filename,
            url=url,
            checksum="2c5501a9f5d7b6998fc7e6a4651030b9765032b2e5a1d7331f5b1f3df6c632a5",
            description="aquella solitaria vaca cubana",
        ),
    )
    monkeypatch.setitem(
        DATASETS,
        "bad_checksum",
        FileMetadata(
            filename=filename, url=url, checksum="bad!", description="aquella solitaria vaca cubana"
        ),
    )


def test_clear_data_home():
    resource = DATASETS["test_remote"]
    assert os.path.exists(resource.filename)
    bmb.clear_data_home(data_home=os.path.dirname(resource.filename))
    assert not os.path.exists(resource.filename)


def test_load_data():
    df = bmb.load_data("test_remote")
    df_ = pd.DataFrame({"x": [0], "y": [1]})
    assert df.equals(df_)


def test_bad_checksum():
    resource = DATASETS["test_remote"]
    bmb.clear_data_home(data_home=os.path.dirname(resource.filename))
    with pytest.raises(IOError):
        bmb.load_data("bad_checksum")


def test_missing_dataset():
    with pytest.raises(ValueError):
        bmb.load_data("does not exist")


def test_list_datasets():
    dataset_string = bmb.load_data()
    # make sure all the names of the data sets are in the dataset description
    for key in ("my_data",):
        assert key in dataset_string
